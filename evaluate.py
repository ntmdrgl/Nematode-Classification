from utils.dataset import create_testing_loader
from utils.training import predict
from models import create_model

import csv
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import hydra
from omegaconf import DictConfig, OmegaConf

EIGENWORMS_CLASS_NAMES = {
    "1": "N2 (wild type)",
    "2": "goa-1",
    "3": "unc-1",
    "4": "unc-38",
    "5": "unc-63",
}

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset_name = Path(cfg.data_path).stem.lower()
    checkpoint_dir = Path("checkpoints") / f"{cfg.model}_{dataset_name}"
    checkpoint_path = checkpoint_dir / f"{cfg.model}_{dataset_name}_best_val_loss.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_cfg = OmegaConf.create(checkpoint["config"])
    num_classes = saved_cfg.model_params.num_classes
    stats = checkpoint["normalization"]

    # Only load test samples; reuse training statistics from the checkpoint.
    with np.load(cfg.data_path, allow_pickle=False) as data:
        X_test = torch.as_tensor(data["X_test"], dtype=torch.float32)
        y_test = torch.as_tensor(data["y_test"], dtype=torch.long)
        class_labels = data["classes"].astype(str).tolist()

    if len(class_labels) != num_classes:
        raise ValueError("Dataset class labels do not match the checkpoint's class count.")
    class_names = [EIGENWORMS_CLASS_NAMES[label] for label in class_labels]
    if len(y_test) == 0:
        raise ValueError("The testing dataset is empty.")
    if y_test.min().item() < 0 or y_test.max().item() >= num_classes:
        raise ValueError("Test labels must be class indices from 0 to num_classes - 1.")

    test_loader = create_testing_loader(X_test, y_test, stats, cfg.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(saved_cfg.model, saved_cfg.model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # predict() sets eval mode and disables gradient tracking.
    probabilities, labels = predict(model, test_loader, device)
    predictions = probabilities.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    matrix = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
    overall_accuracy = 100.0 * float(matrix.trace()) / int(matrix.sum())

    per_class = []
    for index, class_name in enumerate(class_names):
        total = int(matrix[index].sum())
        correct = int(matrix[index, index])
        per_class.append(
            {
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy_percent": 100.0 * correct / total if total else None,
            }
        )

    results_dir = Path("results") / f"{saved_cfg.model}_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scope", "class_name",
                "correct", "total", "accuracy_percent",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "correct": int(matrix.trace()),
                "total": len(labels),
                "accuracy_percent": overall_accuracy,
            }
        )
        for item in per_class:
            writer.writerow({"scope": "per_class", **item})

    figure = Figure(figsize=(7, 6))
    axes = figure.subplots()
    image = axes.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axes, label="Test samples (count)")
    axes.set(
        title=f"{saved_cfg.model.upper()} test-set confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    for row in range(num_classes):
        for column in range(num_classes):
            axes.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=150)

    print(f"Test accuracy: {overall_accuracy:.2f}%")
    for item in per_class:
        accuracy = item["accuracy_percent"]
        description = f"{accuracy:.2f}%" if accuracy is not None else "N/A (no samples)"
        print(f"{item['class_name']} accuracy: {description}")
    print(f"Saved evaluation results: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
