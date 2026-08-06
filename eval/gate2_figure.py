"""
Standalone Gate 2 bar chart figure.
Reads training/gates/gate2_results.json and produces a grouped bar chart of mean
Shannon cross-attention entropy per V-A input for the manuscript.

Usage:
    python eval/gate2_figure.py

Output:
    docs/figures/gate2_entropy_bar.pdf
    docs/figures/gate2_entropy_bar.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless -- must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

GATE2_RESULTS = os.path.join(_REPO_ROOT, "training", "gates", "gate2_results.json")
DOCS_FIGURES = os.path.join(_REPO_ROOT, "docs", "figures")

# Canonical display order (left to right on x-axis)
DISPLAY_ORDER = [
    "Q1-triumphant",
    "Q2-tense",
    "Q3-melancholic",
    "Q4-serene",
    "center-neutral",
]


def make_gate2_bar_chart(results_path: str = GATE2_RESULTS,
                          output_dir: str = DOCS_FIGURES) -> None:
    """Read gate2_results.json and write gate2_entropy_bar.{pdf,png}."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"gate2_results.json not found at {results_path}. "
            "Run training/gates/gate2_attention.py first."
        )
    print(f"[AffectScore] Loading gate2_results.json from {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    entropies = [data["entropy_per_input"][k] for k in DISPLAY_ORDER]
    threshold = data["threshold_bits"]
    entropy_range = data["entropy_range_bits"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DISPLAY_ORDER))

    ax.bar(x, entropies, width=0.5, color="steelblue", alpha=0.85,
           label="Mean entropy (bits)")

    ax.axhline(threshold, linestyle="--", color="tomato", linewidth=1.5,
               label=f"Gate threshold ({threshold:.3f} bits)")

    y_min, y_max = min(entropies), max(entropies)
    x_annot = len(DISPLAY_ORDER) - 0.1
    ax.annotate(
        "",
        xy=(x_annot, y_max),
        xytext=(x_annot, y_min),
        arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5),
    )
    ax.text(
        x_annot + 0.15,
        (y_min + y_max) / 2,
        f"range = {entropy_range:.4f} bits",
        color="darkgreen",
        va="center",
        fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("-", "\n") for k in DISPLAY_ORDER], fontsize=9
    )
    ax.set_ylabel("Mean Shannon entropy (bits)")
    ax.set_title("Cross-attention entropy across V-A inputs")
    ax.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "gate2_entropy_bar.pdf")
    png_path = os.path.join(output_dir, "gate2_entropy_bar.png")

    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[AffectScore] Figure saved: {pdf_path}")
    print(f"[AffectScore] Figure saved: {png_path}")


if __name__ == "__main__":
    make_gate2_bar_chart()
