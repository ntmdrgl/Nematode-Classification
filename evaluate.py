from utils.evaluation import load_evaluation
from utils.training import predict

import csv
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model, test_loader, class_names, saved_cfg, dataset_name = load_evaluation(cfg)
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # predict() sets eval mode and disables gradient tracking.
    probabilities, labels = predict(model, test_loader, device)
    predictions = probabilities.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    matrix = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
    top_classes = probabilities.topk(min(5, num_classes), dim=1).indices.cpu().numpy()
    top5_correct = (top_classes == labels[:, None]).any(axis=1)
    class_f1 = f1_score(
        labels, predictions, labels=np.arange(num_classes), average=None, zero_division=0,
    )
    overall = {
        "scope": "overall",
        "total": len(labels),
        "top1_accuracy_percent": 100.0 * float(matrix.trace()) / int(matrix.sum()),
        "top5_accuracy_percent": 100.0 * float(top5_correct.mean()),
        # Macro F1 gives every dataset class equal weight; undefined F1 is zero.
        "f1_score": float(class_f1.mean()),
    }

    per_class = []
    for index, class_name in enumerate(class_names):
        total = int(matrix[index].sum())
        mask = labels == index
        per_class.append(
            {
                "class_name": class_name,
                "total": total,
                "top1_accuracy_percent": 100.0 * int(matrix[index, index]) / total if total else None,
                "top5_accuracy_percent": 100.0 * float(top5_correct[mask].mean()) if total else None,
                "f1_score": float(class_f1[index]),
            }
        )

    results_dir = Path(to_absolute_path("results")) / f"{saved_cfg.model}_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scope", "class_name",
                "total", "top1_accuracy_percent", "top5_accuracy_percent", "f1_score",
            ],
        )
        writer.writeheader()
        writer.writerow(overall)
        for item in per_class:
            writer.writerow({"scope": "per_class", **item})

    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix * 100.0, row_totals,
        out=np.zeros_like(matrix, dtype=float), where=row_totals != 0,
    )
    with (results_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_class / predicted_class (row %)", *class_names])
        writer.writerows([name, *row] for name, row in zip(class_names, percentages))

    # Grow the figure so every class label remains readable, including OpenWorm strains.
    figure = Figure(figsize=(max(7, 3 + num_classes * 0.22), max(6, 2 + num_classes * 0.22)))
    axes = figure.subplots()
    image = axes.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    figure.colorbar(image, ax=axes, label="Percentage of true-class samples",
                    format=PercentFormatter(xmax=100), fraction=0.046, pad=0.04)
    ticks = np.arange(num_classes)
    axes.set(
        title=f"{saved_cfg.model.upper()} test-set confusion matrix (row percentages)",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=ticks,
        yticks=ticks,
        xticklabels=[class_names[i] for i in ticks],
        yticklabels=[class_names[i] for i in ticks],
    )
    axes.tick_params(axis="x", labelrotation=90)
    axes.tick_params(axis="both", labelsize=7 if num_classes > 20 else 9)
    if num_classes <= 20:
        for row in range(num_classes):
            for column in range(num_classes):
                axes.text(
                    column, row, f"{percentages[row, column]:.1f}%",
                    ha="center", va="center",
                    color="white" if percentages[row, column] > 50 else "black",
                )
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=150)

    print(f"Test top-1 accuracy: {overall['top1_accuracy_percent']:.2f}%")
    print(f"Test top-5 accuracy: {overall['top5_accuracy_percent']:.2f}%")
    print(f"Test macro F1: {overall['f1_score']:.4f}")
    print(f"Saved evaluation results: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
