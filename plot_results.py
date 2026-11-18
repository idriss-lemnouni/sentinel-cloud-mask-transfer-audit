"""Render cloud screening and index changes from the measured audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

METHODS = {"SCL": ("Sen2Cor SCL", "#3862b2"), "probability": ("s2cloudless", "#007b75")}
INK = "#172e40"
NOTE = "#4a6175"


def heading(figure, title, note):
    figure.set_facecolor("#fcfdfe")
    figure.text(0.055, 0.93, title, fontsize=18, weight="bold", color=INK)
    figure.text(0.055, 0.045, note, fontsize=9, color=NOTE)


def graph():
    figure, axis = plt.subplots(figsize=(10.5, 5))
    figure.subplots_adjust(top=0.79, bottom=0.2, left=0.12, right=0.95)
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d0dce7")
    return figure, axis


def render(report, example, out):
    out.mkdir(parents=True, exist_ok=True)
    figure, axis = graph()
    heading(figure, "Cloud screening by reference class",
            f"{len(report['scenes'])} held-out ROIs · correct means retained for clear pixels and masked for other classes")
    for index, (method, (label, color)) in enumerate(METHODS.items()):
        counts = report["test"][method]["pixels_by_reference_class"]
        rates = [100 * row["correct"] / row["pixels"] if row["pixels"] else np.nan for row in counts.values()]
        bars = axis.bar(np.arange(4) + (index - 0.5) * 0.32, rates, width=0.30, color=color, label=label)
        axis.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=3, color=INK)
    axis.set_xticks(range(4), ["Clear", "Thick cloud", "Thin cloud", "Shadow"])
    axis.set(ylabel="Correct screening (%)", ylim=(0, 111))
    axis.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2)
    figure.savefig(out / "screening-by-class.png", dpi=175)
    plt.close(figure)

    figure, axis = graph()
    heading(figure, "Screening shifts vegetation-index means",
            "Each mark is one held-out ROI · candidate clear-pixel mean minus expert clear-pixel mean")
    axis.axhline(0, color=NOTE, linestyle="--", linewidth=1)
    indices = ("NDVI", "NDMI", "SAVI")
    for j, (method, (label, color)) in enumerate(METHODS.items()):
        for i, name in enumerate(indices):
            values = [row["indices"][method][name]["mean_shift"] for row in report["scenes"]]
            values = [value for value in values if value is not None]
            offsets = np.linspace(-0.055, 0.055, len(values))
            axis.scatter(i + (j - 0.5) * 0.26 + offsets, values, color=color, alpha=0.85,
                         s=48, label=label if i == 0 else None)
    axis.set_xticks(range(3), indices)
    axis.set_ylabel("Mean index shift (unitless)")
    axis.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2)
    figure.savefig(out / "vegetation-index-shifts.png", dpi=175)
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(13.5, 4.9))
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.78, wspace=0.09)
    heading(figure, "Where screening disagrees with the expert",
            f"CloudSEN12+ · Aybar et al. · CC0 · {report['protocol']['sample'].split('__')[0]}")
    valid, expert = example["valid"], example["truth"] != 0
    rgb = np.where(valid[..., None], np.clip(example["rgb"] / 0.36, 0, 1), 0.88)
    palette = ["#e3e8ee", "#f2f4f7", "#de664c", "#4a7ec8"]
    cmap = ListedColormap(palette)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", color=INK)
    axes[1].imshow(np.where(valid, example["truth"], np.nan), vmin=0, vmax=3,
                   cmap=ListedColormap(["#e3e8ee", "#f2a15b", "#d79dba", "#614f79"]))
    axes[1].set_title("Expert labels", color=INK)
    for axis, method in zip(axes[2:], ("scl", "probability")):
        predicted = example[method]
        false_clear = expert & ~predicted
        false_obscured = ~expert & predicted
        errors = np.where(false_clear, 2, np.where(false_obscured, 3, 1))
        axis.imshow(np.where(valid, errors, 0), vmin=0, vmax=3, cmap=cmap, interpolation="nearest")
        axis.set_title(METHODS["SCL" if method == "scl" else method][0] + " errors", color=INK)
    for axis in axes:
        axis.axis("off")
    labels = [("#f2a15b", "Thick cloud"), ("#d79dba", "Thin cloud"), ("#614f79", "Shadow"),
              (palette[2], "Obscured pixel retained"), (palette[3], "Clear pixel removed")]
    figure.legend(handles=[Patch(facecolor=color, label=text) for color, text in labels], ncol=5,
                  frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.11), fontsize=9)
    figure.savefig(out / "held-out-screening-map.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--metrics", type=Path, required=True)
    cli.add_argument("--example", type=Path, required=True)
    cli.add_argument("--out", type=Path, default=Path("figures/results"))
    args = cli.parse_args()
    with np.load(args.example, allow_pickle=False) as example:
        render(json.loads(args.metrics.read_text(encoding="utf-8")), example, args.out)
