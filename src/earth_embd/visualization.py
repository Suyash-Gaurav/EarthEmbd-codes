"""
Publication-oriented plotting helpers for EarthEM.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def set_publication_style() -> None:
    """Apply a clean, consistent matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 15,
    title: str = "Random Forest Feature Importance",
    figsize: tuple = (9, 6),
) -> plt.Figure:
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Embedding Dimension")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pca_scatter(
    coords: np.ndarray,
    labels: Sequence,
    title: str = "PCA Projection",
    label_names: Optional[Dict] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    unique = sorted(set(labels))
    for lab in unique:
        mask = np.array(labels) == lab
        name = label_names.get(lab, str(lab)) if label_names else str(lab)
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.75, s=40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str = "Inter-annual Cosine Similarity",
    figsize: tuple = (9, 7),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)
    sns.heatmap(
        similarity_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        mask=mask,
        vmin=0.85,
        vmax=1.0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Cosine similarity"},
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_climate_similarity_boxplot(
    group_sims: Dict[str, List[float]],
    title: str = "Within-group Similarity by Climate Category",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    data = []
    for cat in ["drought", "normal", "good"]:
        for v in group_sims.get(cat, []):
            data.append({"Climate": cat.capitalize(), "Similarity": v})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="Climate", y="Similarity", ax=ax, palette="Set2")
    sns.stripplot(
        data=df, x="Climate", y="Similarity", ax=ax, color="black", alpha=0.5, size=4
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_separation_change(
    years: Sequence[int],
    separation_ratios: Sequence[float],
    title: str = "Cluster Separation Ratio Over Time",
    figsize: tuple = (7, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(years, separation_ratios, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Separation Ratio")
    ax.set_title(title)
    ax.set_xticks(list(years))
    fig.tight_layout()
    return fig
