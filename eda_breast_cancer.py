#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) and Visualization
Dataset : Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
Libs    : Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn (PCA only)

The dataset ships with scikit-learn, so the script runs offline with no download.

Usage
-----
    python eda_breast_cancer.py                  # saves everything to ./eda_output
    python eda_breast_cancer.py --out results    # custom output folder
    python eda_breast_cancer.py --show           # also open plot windows

Outputs
-------
    <out>/figures/*.png      all visualizations (150 dpi)
    <out>/tables/*.csv       summary tables used in the report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PALETTE = {"Benign": "#2a9d8f", "Malignant": "#e63946"}
HUE_ORDER = ["Benign", "Malignant"]
KEY_FEATURES = ["mean radius", "mean texture", "mean area", "mean concavity"]
PAIR_FEATURES = ["mean radius", "mean texture", "mean concavity", "mean smoothness"]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_fig(fig: plt.Figure, path: Path, show: bool = False) -> None:
    """Tight-layout, save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  [saved] {path}")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 1. Load and prepare data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Load the dataset and add a readable diagnosis label.

    In scikit-learn's copy of the data, target 0 = malignant and 1 = benign.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    return df.drop(columns="target")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "diagnosis"]


# --------------------------------------------------------------------------- #
# 2. Initial inspection and descriptive statistics
# --------------------------------------------------------------------------- #
def initial_inspection(df: pd.DataFrame, tables: Path) -> None:
    section("1. INITIAL INSPECTION")
    print(f"Shape            : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values   : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows   : {int(df.duplicated().sum())}")
    print("\nData types:")
    print(df.dtypes.value_counts().to_string())
    print("\nClass balance:")
    counts = df["diagnosis"].value_counts()
    print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(2)}))

    desc = df[feature_columns(df)].describe().T
    desc["skew"] = df[feature_columns(df)].skew()
    desc["kurtosis"] = df[feature_columns(df)].kurt()
    desc.round(4).to_csv(tables / "descriptive_statistics.csv")
    print("\nDescriptive statistics (first 10 features):")
    print(desc.round(3).head(10).to_string())

    grouped = df.groupby("diagnosis")[KEY_FEATURES].agg(["mean", "std"]).round(3)
    grouped.to_csv(tables / "group_summary_key_features.csv")
    print("\nGroup means by diagnosis (key features):")
    print(df.groupby("diagnosis")[KEY_FEATURES].mean().round(2).to_string())


# --------------------------------------------------------------------------- #
# 3. Visualizations
# --------------------------------------------------------------------------- #
def plot_class_distribution(df: pd.DataFrame, figs: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(data=df, x="diagnosis", order=HUE_ORDER, hue="diagnosis",
                  palette=PALETTE, legend=False, ax=ax)
    total = len(df)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height() / total:.1%})",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Figure 1: Class Distribution of Diagnosis")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of patients")
    ax.set_ylim(0, df["diagnosis"].value_counts().max() * 1.15)
    save_fig(fig, figs / "01_class_distribution.png", show)


def plot_histograms(df: pd.DataFrame, figs: Path, show: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), KEY_FEATURES):
        sns.histplot(data=df, x=feat, hue="diagnosis", hue_order=HUE_ORDER,
                     palette=PALETTE, kde=True, bins=30, alpha=0.55, ax=ax)
        ax.set_title(feat.title())
        ax.set_xlabel(feat)
        ax.set_ylabel("Frequency")
    fig.suptitle("Figure 2: Distribution of Key Features by Diagnosis",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, figs / "02_histograms_key_features.png", show)


def plot_boxplots(df: pd.DataFrame, figs: Path, show: bool) -> None:
    feats = ["mean radius", "mean perimeter", "mean concavity", "mean concave points"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    for ax, feat in zip(axes, feats):
        sns.boxplot(data=df, x="diagnosis", y=feat, order=HUE_ORDER, hue="diagnosis",
                    palette=PALETTE, legend=False, width=0.55, fliersize=3, ax=ax)
        ax.set_title(feat.title())
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel(feat)
    fig.suptitle("Figure 3: Box Plots - Size and Shape Features by Diagnosis",
                 fontsize=14, fontweight="bold", y=1.03)
    save_fig(fig, figs / "03_boxplots_by_diagnosis.png", show)


def plot_correlation_heatmap(df: pd.DataFrame, figs: Path, tables: Path,
                             show: bool) -> pd.DataFrame:
    mean_cols = [c for c in feature_columns(df) if c.startswith("mean ")]
    corr = df[mean_cols].corr()
    corr.round(3).to_csv(tables / "correlation_mean_features.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8},
                cbar_kws={"label": "Pearson correlation (r)"}, ax=ax)
    ax.set_title("Figure 4: Correlation Heatmap of the 10 'Mean' Features")
    save_fig(fig, figs / "04_correlation_heatmap.png", show)
    return df[feature_columns(df)].corr()


def high_correlation_pairs(corr_full: pd.DataFrame, threshold: float = 0.95,
                           tables: Path | None = None) -> pd.DataFrame:
    section("MULTICOLLINEARITY CHECK (|r| >= %.2f)" % threshold)
    upper = corr_full.where(np.triu(np.ones(corr_full.shape, dtype=bool), k=1))
    pairs = (upper.stack().rename("r").reset_index()
             .rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
    pairs = pairs[pairs["r"].abs() >= threshold].sort_values("r", ascending=False)
    print(f"{len(pairs)} feature pairs have |r| >= {threshold}")
    print(pairs.head(10).round(4).to_string(index=False))
    if tables is not None:
        pairs.round(4).to_csv(tables / "high_correlation_pairs.csv", index=False)
    return pairs


def plot_target_correlation(df: pd.DataFrame, figs: Path, tables: Path,
                            show: bool) -> pd.Series:
    y = (df["diagnosis"] == "Malignant").astype(int)
    corr = df[feature_columns(df)].corrwith(y).sort_values()
    corr.round(4).to_csv(tables / "correlation_with_malignancy.csv", header=["r"])

    fig, ax = plt.subplots(figsize=(8, 9))
    colors = np.where(corr > 0, PALETTE["Malignant"], PALETTE["Benign"])
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Figure 5: Correlation of Each Feature with Malignancy")
    ax.set_xlabel("Pearson correlation with malignant (1) vs benign (0)")
    ax.set_ylabel("Feature")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE["Malignant"]),
                       plt.Rectangle((0, 0), 1, 1, color=PALETTE["Benign"])],
              labels=["Higher in malignant", "Higher in benign"], loc="lower right")
    save_fig(fig, figs / "05_correlation_with_target.png", show)
    return corr


def plot_pairplot(df: pd.DataFrame, figs: Path, show: bool) -> None:
    g = sns.pairplot(df, vars=PAIR_FEATURES, hue="diagnosis", hue_order=HUE_ORDER,
                     palette=PALETTE, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 22},
                     corner=True)
    g.figure.suptitle("Figure 6: Pair Plot of Selected Features", y=1.02,
                      fontsize=14, fontweight="bold")
    g.figure.savefig(figs / "06_pairplot.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(g.figure)
    print(f"  [saved] {figs / '06_pairplot.png'}")


def plot_scatter_regression(df: pd.DataFrame, figs: Path, show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pairs = [("mean radius", "mean area"), ("mean concavity", "mean concave points")]
    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=df, x=x, y=y, hue="diagnosis", hue_order=HUE_ORDER,
                        palette=PALETTE, alpha=0.7, s=35, ax=ax)
        sns.regplot(data=df, x=x, y=y, scatter=False, color="black",
                    line_kws={"linestyle": "--", "linewidth": 1.5}, ax=ax)
        r = df[x].corr(df[y])
        ax.set_title(f"{x.title()} vs {y.title()} (r = {r:.2f})")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(title="Diagnosis")
    fig.suptitle("Figure 7: Relationships Between Highly Correlated Features",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, figs / "07_scatter_regression.png", show)


def plot_violin_standardised(df: pd.DataFrame, figs: Path, show: bool) -> None:
    """Standardise features (z-score) so they share a scale, then plot violins."""
    cols = [c for c in feature_columns(df) if c.startswith("worst ")]
    z = pd.DataFrame(StandardScaler().fit_transform(df[cols]), columns=cols)
    z["diagnosis"] = df["diagnosis"].values
    long = z.melt(id_vars="diagnosis", var_name="feature", value_name="z-score")

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=long, x="feature", y="z-score", hue="diagnosis",
                   hue_order=HUE_ORDER, palette=PALETTE, split=True, inner="quart",
                   density_norm="width", ax=ax)
    ax.set_title("Figure 8: Standardised 'Worst' Features by Diagnosis (Violin Plot)")
    ax.set_xlabel("Feature (z-score standardised)")
    ax.set_ylabel("Standardised value (z-score)")
    ax.tick_params(axis="x", rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(title="Diagnosis", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_fig(fig, figs / "08_violin_standardised.png", show)


def detect_outliers(df: pd.DataFrame, figs: Path, tables: Path,
                    show: bool) -> pd.DataFrame:
    section("OUTLIER ANALYSIS (1.5 x IQR rule)")
    rows = []
    for col in feature_columns(df):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({"feature": col, "outliers": n, "percent": round(n / len(df) * 100, 2)})
    out = pd.DataFrame(rows).sort_values("outliers", ascending=False)
    out.to_csv(tables / "outlier_counts.csv", index=False)
    print(out.head(10).to_string(index=False))

    top = out.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["outliers"], color="#457b9d")
    for y, v in enumerate(top["outliers"]):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=9)
    ax.set_title("Figure 9: Features with the Most IQR Outliers (Top 12)")
    ax.set_xlabel("Number of outlier observations (n = 569)")
    ax.set_ylabel("Feature")
    save_fig(fig, figs / "09_outlier_counts.png", show)
    return out


def plot_pca(df: pd.DataFrame, figs: Path, tables: Path, show: bool) -> PCA:
    """Standardise all 30 features, then project to 2-D with PCA."""
    section("PCA (30 features standardised)")
    cols = feature_columns(df)
    X = StandardScaler().fit_transform(df[cols])
    pca = PCA().fit(X)
    scores = pca.transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"PC1 = {evr[0]:.1%}, PC2 = {evr[1]:.1%}, PC1+PC2 = {cum[1]:.1%}")
    n95 = int(np.argmax(cum >= 0.95)) + 1
    print(f"Components needed for 95% variance: {n95}")

    pd.DataFrame({"component": range(1, len(evr) + 1),
                  "explained_variance_ratio": evr.round(4),
                  "cumulative": cum.round(4)}).to_csv(tables / "pca_variance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=df["diagnosis"], hue_order=HUE_ORDER,
                    palette=PALETTE, alpha=0.75, s=40, ax=axes[0])
    axes[0].set_title("PCA Projection (PC1 vs PC2)")
    axes[0].set_xlabel(f"PC1 ({evr[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({evr[1]:.1%} variance)")
    axes[0].legend(title="Diagnosis")

    axes[1].bar(range(1, 11), evr[:10], color="#457b9d", label="Individual")
    axes[1].plot(range(1, 11), cum[:10], "o-", color="#e76f51", label="Cumulative")
    axes[1].axhline(0.95, color="grey", linestyle="--", linewidth=1)
    axes[1].set_title("Explained Variance by Principal Component")
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("Explained variance ratio")
    axes[1].set_xticks(range(1, 11))
    axes[1].legend()
    fig.suptitle("Figure 10: Principal Component Analysis", fontsize=14,
                 fontweight="bold", y=1.02)
    save_fig(fig, figs / "10_pca.png", show)
    return pca


# --------------------------------------------------------------------------- #
# 4. Statistical testing
# --------------------------------------------------------------------------- #
def group_tests(df: pd.DataFrame, tables: Path) -> pd.DataFrame:
    """Mann-Whitney U test + Cohen's d for every feature (malignant vs benign)."""
    section("STATISTICAL TESTS: Malignant vs Benign")
    m = df[df["diagnosis"] == "Malignant"]
    b = df[df["diagnosis"] == "Benign"]
    rows = []
    for col in feature_columns(df):
        u, p = stats.mannwhitneyu(m[col], b[col], alternative="two-sided")
        pooled = np.sqrt(((len(m) - 1) * m[col].var() + (len(b) - 1) * b[col].var())
                         / (len(m) + len(b) - 2))
        d = (m[col].mean() - b[col].mean()) / pooled
        rows.append({"feature": col, "mean_malignant": m[col].mean(),
                     "mean_benign": b[col].mean(), "cohens_d": d, "p_value": p})
    res = pd.DataFrame(rows)
    res["abs_d"] = res["cohens_d"].abs()
    res = res.sort_values("abs_d", ascending=False).drop(columns="abs_d")
    res.round(6).to_csv(tables / "group_tests.csv", index=False)
    print(res.head(8).round(4).to_string(index=False))
    print(f"\nFeatures with p < 0.05: {(res['p_value'] < 0.05).sum()} of {len(res)}")
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="eda_output", help="output folder")
    parser.add_argument("--show", action="store_true", help="display plots interactively")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # headless backend

    out = Path(args.out)
    figs, tables = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df.to_csv(out / "breast_cancer_dataset.csv", index=False)

    initial_inspection(df, tables)

    section("VISUALIZATIONS")
    plot_class_distribution(df, figs, args.show)
    plot_histograms(df, figs, args.show)
    plot_boxplots(df, figs, args.show)
    corr_full = plot_correlation_heatmap(df, figs, tables, args.show)
    high_correlation_pairs(corr_full, 0.95, tables)
    plot_target_correlation(df, figs, tables, args.show)
    plot_pairplot(df, figs, args.show)
    plot_scatter_regression(df, figs, args.show)
    plot_violin_standardised(df, figs, args.show)
    detect_outliers(df, figs, tables, args.show)
    plot_pca(df, figs, tables, args.show)
    group_tests(df, tables)

    section("DONE")
    print(f"Figures -> {figs.resolve()}")
    print(f"Tables  -> {tables.resolve()}")


if __name__ == "__main__":
    main()
