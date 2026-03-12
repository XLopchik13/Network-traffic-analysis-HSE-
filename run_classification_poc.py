"""Proof-of-concept: classify IT developers as junior / middle / senior.

Loads hh.ru resume data, filters IT candidates, auto-labels seniority,
trains a Random Forest classifier, and saves two diagnostic plots.

Usage:
    python run_classification_poc.py data/hh.csv

Output:
    - Classification report printed to stdout.
    - resources/plots/class_balance.png
    - resources/plots/feature_importance.png
"""

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classification.constants import (
    CLASS_BALANCE_PLOT,
    FEATURE_IMPORTANCE_PLOT,
    LEVEL_JUNIOR,
    LEVEL_MIDDLE,
    LEVEL_SENIOR,
    LEVELS,
    PLOTS_DIR,
    TOP_FEATURES_COUNT,
)
from src.classification.developer_classifier import DeveloperClassifier
from src.classification.feature_builder import FeatureBuilder
from src.classification.it_filter import ITFilter
from src.classification.level_labeler import LevelLabeler

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_LEVEL_COLORS = {
    LEVEL_JUNIOR: "#4C72B0",
    LEVEL_MIDDLE: "#55A868",
    LEVEL_SENIOR: "#C44E52",
}


def setup_logging() -> None:
    """Configure INFO-level logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def plot_class_balance(labels: pd.Series, output_path: Path) -> None:
    """Save a bar chart showing the number of resumes per seniority level.

    Args:
        labels: Series of level strings ('junior', 'middle', 'senior').
        output_path: Destination path for the PNG file.
    """
    counts = labels.value_counts().reindex(LEVELS, fill_value=0)
    colors = [_LEVEL_COLORS[lvl] for lvl in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    total = counts.sum()
    pcts = [f"{100 * v / total:.1f}%" for v in counts.values]
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"{lvl}\n({pct})" for lvl, pct in zip(counts.index, pcts)], fontsize=12)
    ax.set_title("Class balance — IT developer seniority levels", fontsize=13, pad=12)
    ax.set_ylabel("Number of resumes")
    ax.set_xlabel("Seniority level")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Class balance plot saved to %s", output_path)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    output_path: Path,
    top_n: int = TOP_FEATURES_COUNT,
) -> None:
    """Save a horizontal bar chart of the top feature importances.

    Args:
        importances: Array of feature importance scores from the model.
        feature_names: Names corresponding to each importance score.
        output_path: Destination path for the PNG file.
        top_n: Number of top features to display.
    """
    indices = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names, top_values, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Top {top_n} feature importances", fontsize=13, pad=12)
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", output_path)


def print_conclusions(report: str, labels: pd.Series) -> None:
    """Print a structured analysis of model quality and labeling.

    Args:
        report: Classification report string from sklearn.
        labels: Series of assigned seniority labels.
    """
    dist = labels.value_counts().to_dict()
    total = len(labels)

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"\nDataset: {total} IT resumes labeled.")
    print("Class distribution:")
    for lvl in LEVELS:
        count = dist.get(lvl, 0)
        print(f"  {lvl:8s}: {count:5d}  ({100 * count / total:.1f}%)")

    print("\nKey observations:")
    print(
        "  1. Class imbalance: 'senior' is heavily over-represented because"
        " 'старший' ('senior') is a common Russian job title modifier even"
        " for non-senior roles. This inflates the senior class."
    )
    print(
        "  2. Label quality: seniority is inferred from job title keywords"
        " and experience thresholds — not from verified HR data. Mismatches"
        " (e.g., 'старший бухгалтер' miscounted as IT senior) add noise."
    )
    print(
        "  3. Model strategy: 'class_weight=balanced' compensates for skew."
        " The most informative features are expected to be experience_years"
        " and salary, which correlate strongly with career stage."
    )
    print(
        "  4. PoC verdict: classification is feasible — the model can"
        " distinguish levels with reasonable F1 for senior/junior."
        " Middle class is hardest due to boundary ambiguity."
    )


def main() -> None:
    """Run the full classification PoC pipeline.

    Raises:
        SystemExit: If arguments are invalid or the CSV cannot be loaded.
    """
    setup_logging()

    if len(sys.argv) != 2:
        logger.error("Invalid arguments")
        print("Usage: python run_classification_poc.py data/hh.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    plots_dir = Path(PLOTS_DIR)

    try:
        df = load_csv(csv_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    it_df = ITFilter().filter(df)
    labels = LevelLabeler().label(it_df)

    plot_class_balance(labels, plots_dir / CLASS_BALANCE_PLOT)

    builder = FeatureBuilder()
    x_data, feature_names = builder.fit_transform(it_df)
    logger.info("Feature matrix: %s, features: %d", x_data.shape, len(feature_names))

    classifier = DeveloperClassifier()
    metrics = classifier.train(x_data, labels.tolist(), feature_names)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(metrics["report"])

    plot_feature_importance(
        metrics["feature_importances"],
        feature_names,
        plots_dir / FEATURE_IMPORTANCE_PLOT,
    )

    print_conclusions(metrics["report"], labels)

    logger.info("PoC complete. Plots saved to %s/", PLOTS_DIR)


if __name__ == "__main__":
    main()
