"""Visualization utilities: radar charts for probe scores."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def create_radar_chart(
    probe_scores: Dict[str, float],
    model_name: str = "Model",
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 8),
) -> Optional[object]:
    """Create a radar chart showing scores across probe categories.

    Args:
        probe_scores: Mapping of probe name to score in [0, 1].
        model_name: Title label for the chart.
        output_path: If provided, saves the figure to this path.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object, or None if matplotlib not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    categories = list(probe_scores.keys())
    values = [probe_scores[c] for c in categories]
    n = len(categories)

    if n < 3:
        # Radar chart needs at least 3 axes; fall back to bar chart
        return _create_bar_chart(probe_scores, model_name, output_path, figsize)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]  # close the polygon
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="#2196F3")
    ax.plot(angles, values, "o-", linewidth=2, color="#1565C0")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_format_label(c) for c in categories], size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="gray")
    ax.set_title(f"DreamBench: {model_name}", size=14, pad=20)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def create_comparison_radar(
    all_scores: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 10),
) -> Optional[object]:
    """Overlay multiple models on one radar chart.

    Args:
        all_scores: Mapping of model_name -> {probe_name: score}.
        output_path: If provided, saves the figure.
        figsize: Figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not all_scores:
        return None

    # Use the union of all probe names
    all_probes = sorted(set(p for scores in all_scores.values() for p in scores))
    n = len(all_probes)
    if n < 3:
        return None

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, (model_name, scores) in enumerate(all_scores.items()):
        values = [scores.get(p, 0.0) for p in all_probes]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.fill(angles, values, alpha=0.1, color=color)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=model_name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_format_label(c) for c in all_probes], size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("DreamBench: Model Comparison", size=14, pad=20)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _create_bar_chart(
    probe_scores: Dict[str, float],
    model_name: str,
    output_path: Optional[Path],
    figsize: tuple,
) -> Optional[object]:
    """Fallback bar chart for < 3 probe categories."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    categories = list(probe_scores.keys())
    values = [probe_scores[c] for c in categories]

    ax.barh(categories, values, color="#2196F3")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(f"DreamBench: {model_name}")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _format_label(probe_name: str) -> str:
    """Convert probe_name to a readable label."""
    return probe_name.replace("_", " ").title()
