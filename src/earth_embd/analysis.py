"""
Core analytical routines for EarthEM:
- supervised classification
- temporal cosine similarity & ANOVA
- cluster separation and shift statistics
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# RQ1 — Crop-type classification
# ---------------------------------------------------------------------------

def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest classifier with a stratified train/test split.

    Returns
    -------
    model : fitted RandomForestClassifier
    metrics : dict containing accuracy, predictions, report, etc.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=False
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importances": pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# RQ2 — Temporal similarity
# ---------------------------------------------------------------------------

def pairwise_cosine_similarity(
    embedding_matrix: np.ndarray,
    labels: Sequence,
) -> pd.DataFrame:
    """Return a square DataFrame of pairwise cosine similarities."""
    sim = cosine_similarity(embedding_matrix)
    return pd.DataFrame(sim, index=labels, columns=labels)


def climate_group_similarities(
    similarity_df: pd.DataFrame,
    climate_map: Dict[int, str],
) -> Dict[str, List[float]]:
    """
    Collect within-group and between-group pairwise similarities
    according to a climate label dictionary (year → climate category).
    """
    groups = {"drought": [], "normal": [], "good": [], "between": []}

    years = list(similarity_df.index)
    for i, y1 in enumerate(years):
        for y2 in years[i + 1 :]:
            sim = similarity_df.loc[y1, y2]
            c1 = climate_map.get(int(y1), "unknown")
            c2 = climate_map.get(int(y2), "unknown")
            if c1 == c2 and c1 in groups:
                groups[c1].append(sim)
            else:
                groups["between"].append(sim)
    return groups


def anova_climate_groups(
    group_sims: Dict[str, List[float]],
) -> Tuple[float, float]:
    """
    One-way ANOVA on the within-group similarity lists
    (drought, normal, good). Returns (F-statistic, p-value).
    """
    samples = [
        group_sims["drought"],
        group_sims["normal"],
        group_sims["good"],
    ]
    # Guard against empty groups
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2:
        return np.nan, np.nan
    return f_oneway(*samples)


# ---------------------------------------------------------------------------
# RQ3 — Cluster separation & shift
# ---------------------------------------------------------------------------

def compute_cluster_separation(
    embeddings: pd.DataFrame,
    labels: pd.Series,
    positive_label: int = 0,
    negative_label: int = 1,
) -> Dict[str, float]:
    """
    Compute Euclidean distance between class centroids and a simple
    separation ratio (between-centroid distance / pooled within-class std).
    """
    pos = embeddings[labels == positive_label]
    neg = embeddings[labels == negative_label]

    pos_centroid = pos.mean()
    neg_centroid = neg.mean()
    centroid_dist = np.linalg.norm(pos_centroid - neg_centroid)

    pos_var = pos.var().mean()
    neg_var = neg.var().mean()
    pooled_std = np.sqrt((pos_var + neg_var) / 2.0)
    separation_ratio = centroid_dist / pooled_std if pooled_std > 0 else np.nan

    return {
        "centroid_distance": float(centroid_dist),
        "separation_ratio": float(separation_ratio),
        "pos_centroid": pos_centroid,
        "neg_centroid": neg_centroid,
    }


def distance_to_centroid(
    samples: pd.DataFrame,
    centroid: pd.Series,
) -> np.ndarray:
    """Euclidean distance of each sample to a given centroid."""
    return np.linalg.norm(samples.values - centroid.values, axis=1)


def two_sample_ttest(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float]:
    """Independent two-sample t-test. Returns (statistic, p-value)."""
    return ttest_ind(a, b, equal_var=False)


def run_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans, StandardScaler]:
    """Standardize features and fit KMeans. Returns labels, model, scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, model, scaler


def pca_projection(
    X: pd.DataFrame,
    n_components: int = 2,
) -> Tuple[np.ndarray, PCA]:
    """Fit PCA and return the projected coordinates plus the fitted model."""
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    return coords, pca
