"""Export CAM-derived local class probabilities for every test recording.

Wang et al., section IV: M_c(t) = sum_k W[c, k] * features[k, t].
Here local probabilities are softmax(M(t) + classifier.bias) across classes.
They are not calibrated frame labels, and averaging them does not reproduce
the recording probability: softmax must follow averaging the local logits.
For the 2D dilated FCN, scores are averaged over the embedding axis and
interpolated to input length before softmax. Interpolation adds no temporal
detail; recording predictions always use the original GAP output.
"""

import csv
from pathlib import Path
import re

import hydra
from hydra.utils import to_absolute_path
from matplotlib.figure import Figure
import numpy as np
from omegaconf import DictConfig
import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils.evaluation import load_evaluation


def temporal_predictions(model, inputs):
    """Apply a GAP classifier at every final feature-map time point.

    Returns probabilities shaped (batch, time, classes) and recording logits.
    The caller must set model.eval() and disable gradient tracking.
    """
    features = model.features(inputs)
    if features.ndim == 3:
        local_logits = model.classifier(features.transpose(1, 2)).transpose(1, 2)
        recording_logits = model.classifier(model.pool(features).squeeze(-1))
    elif features.ndim == 4:
        # The classifier's 1x1 convolution acts independently at each location.
        local_logits = model.classifier(features).mean(dim=2)
        recording_logits = model.classifier(model.pool(features)).flatten(1)
    else:
        raise ValueError("CAM requires 1D or 2D convolutional feature maps.")
    if local_logits.shape[-1] != inputs.shape[-1]:
        local_logits = F.interpolate(
            local_logits, size=inputs.shape[-1], mode="linear", align_corners=False
        )
    return local_logits.softmax(dim=1).transpose(1, 2), recording_logits


def save_classifier(model, output_dir, class_names):
    """Save exact classifier tensors and a readable weight table."""
    state = {
        key: value.detach().cpu() for key, value in model.classifier.state_dict().items()
    }
    torch.save({"state_dict": state, "class_names": class_names}, output_dir / "classifier_weights.pt")
    # Linear weights are (classes, filters); 1x1 Conv2d adds two singleton axes.
    weights = state["weight"].flatten(start_dim=1).numpy()
    bias = state.get("bias")
    with (output_dir / "classifier_weights.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "bias", *[f"feature_{i}" for i in range(weights.shape[1])]])
        for index, name in enumerate(class_names):
            writer.writerow([name, float(bias[index]) if bias is not None else 0.0, *weights[index]])


def save_sample(output_dir, sample_index, probabilities, true_class, predicted_class, class_names):
    """Save a probability plot grouped by the true recording class."""
    sample_name = f"sample_{sample_index:04d}"
    time_points = np.arange(probabilities.shape[0])

    # Prefixing the class index avoids collisions and Windows reserved names.
    folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", class_names[true_class])[:80].rstrip(" .")
    class_dir = output_dir / f"{true_class:03d}_{folder_name}"
    class_dir.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 4.5))
    axes = figure.subplots()
    if len(class_names) <= 10:
        plotted_classes = range(len(class_names))
    else:
        top_classes = np.argsort(probabilities.mean(axis=0))[-5:][::-1].tolist()
        plotted_classes = list(dict.fromkeys([true_class, predicted_class, *top_classes]))
    for index in plotted_classes:
        axes.plot(time_points, probabilities[:, index], label=class_names[index], linewidth=0.8)
    axes.set(
        title=(
            f"Test sample {sample_index:04d} | True: {class_names[true_class]} | "
            f"Recording prediction: {class_names[predicted_class]}"
        ),
        xlabel="Time point (zero-based index)",
        ylabel="Local class probability" if len(class_names) <= 10 else "Local class probability (selected classes)",
        ylim=(0, 1),
        xlim=(0, max(len(time_points) - 1, 1)),
    )
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(plotted_classes), 4))
    figure.tight_layout()
    plot_path = class_dir / f"{sample_name}.png"
    figure.savefig(plot_path, dpi=150)
    figure.clear()
    return plot_path


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model, test_loader, class_names, saved_cfg, dataset_name = load_evaluation(cfg)
    if saved_cfg.model not in ("fcn", "dilated_fcn"):
        raise ValueError("CAM requires an FCN or dilated_fcn GAP classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output_dir = Path(to_absolute_path("results")) / f"{saved_cfg.model}_{dataset_name}" / "cam"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_classifier(model, output_dir, class_names)

    # This index is the original zero-based test-set row: the loader never shuffles.
    sample_index = 0
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "true_class", "predicted_class", "recording_probability", "plot"])
        with torch.inference_mode(), tqdm(total=len(test_loader.dataset), desc="Exporting test CAM plots") as progress:
            for inputs, labels in test_loader:
                probabilities, recording_logits = temporal_predictions(model, inputs.to(device))
                if not torch.isfinite(probabilities).all() or not torch.isfinite(recording_logits).all():
                    raise RuntimeError("Non-finite CAM predictions; check inputs and normalization.")
                probabilities = probabilities.cpu().numpy()
                recording_probabilities = recording_logits.softmax(dim=-1).cpu().numpy()
                for local_index, true_class in enumerate(labels.tolist()):
                    predicted_class = int(recording_probabilities[local_index].argmax())
                    plot_path = save_sample(
                        output_dir, sample_index, probabilities[local_index],
                        true_class, predicted_class, class_names,
                    )
                    writer.writerow([
                        sample_index, class_names[true_class], class_names[predicted_class],
                        float(recording_probabilities[local_index, predicted_class]),
                        str(plot_path.relative_to(output_dir)),
                    ])
                    sample_index += 1
                    progress.update(1)
    print(f"Saved classifier weights and {sample_index} sample plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
