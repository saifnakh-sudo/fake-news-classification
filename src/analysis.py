"""News classification and analysis for the GDG project.

The numeric classifier learns from the labels in the project dataset.
Original submissions are retained separately in originals/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEXT_FEATURES = [
    "Title_Word_Count", "Title_Char_Count", "Body_Word_Count",
    "Body_Char_Count", "Body_Avg_Word_Length", "Clickbait_Score",
]
SHARE_FEATURES = ["Shares_Per_Char", "Log_Shares"]
REQUIRED_COLUMNS = ["Title", "Corps_Textee", "URL_Source", "Nb_Partages", "Etiquette"]


def load_data(path: Path) -> pd.DataFrame:
    """Validate the workshop schema without treating missing numbers as text."""
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    for col in ["Title", "Corps_Textee", "URL_Source"]:
        df[col] = df[col].fillna("").astype(str)
    df["Nb_Partages"] = pd.to_numeric(df["Nb_Partages"], errors="raise")
    shares = df["Nb_Partages"].to_numpy(dtype=float)
    if not np.isfinite(shares).all() or (shares < 0).any():
        raise ValueError("Nb_Partages must contain finite, non-negative numbers.")
    if df["Etiquette"].isna().any() or not df["Etiquette"].isin(["Real", "Fake"]).all():
        raise ValueError("Etiquette must contain only Real or Fake.")
    return df.drop_duplicates().reset_index(drop=True)


def clickbait_score(text: str) -> int:
    """Simple workshop heuristic; it does not determine article truthfulness."""
    lower = text.lower()
    return (int("!" in text) + int("?" in text) + int("video" in lower)
            + int("exclusif" in lower) + 2 * int(text.isupper()))


def make_features(df: pd.DataFrame, share_cap: float | None = None) -> pd.DataFrame:
    title = df["Title"]
    body = df["Corps_Textee"]
    features = pd.DataFrame(index=df.index)
    features["Title_Word_Count"] = title.str.split().str.len()
    features["Title_Char_Count"] = title.str.len()
    features["Body_Word_Count"] = body.str.split().str.len()
    features["Body_Char_Count"] = body.str.len()
    # Preserve the original workshop definition, including spaces/punctuation.
    features["Body_Avg_Word_Length"] = (
        features["Body_Char_Count"] / (features["Body_Word_Count"] + 1)
    )
    features["Clickbait_Score"] = title.map(clickbait_score)
    shares = df["Nb_Partages"]
    if share_cap is not None:
        shares = shares.clip(upper=share_cap)
    features["Shares_Per_Char"] = shares / (features["Body_Char_Count"] + 1)
    features["Log_Shares"] = np.log1p(shares)
    return features


def save_figure(output: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(output / filename, dpi=160, bbox_inches="tight")
    plt.close()


def run_analysis(data_path: Path, output: Path) -> dict:
    """Run classification, exploratory analysis, and a leakage-free regression."""
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.figsize": (10, 6), "axes.titleweight": "bold"})
    df = load_data(data_path)
    target = df["Etiquette"].map({"Real": 0, "Fake": 1})
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=SEED, stratify=target,
    )
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    # Fit the clipping threshold on the training set only.
    q1, q3 = train["Nb_Partages"].quantile([0.25, 0.75])
    share_cap = float(q3 + 3 * (q3 - q1))
    x_train = make_features(train, share_cap)
    x_test = make_features(test, share_cap)
    y_train = target.loc[train_idx]
    y_test = target.loc[test_idx]
    classifier = RandomForestClassifier(n_estimators=100, random_state=SEED)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=["Real", "Fake"],
        output_dict=True, zero_division=0,
    )

    plt.figure()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual workshop label")
    plt.title("Random Forest classification — educational holdout")
    plt.figtext(0.5, -0.02, "Template-style data; these results do not measure real-world fact checking.",
                ha="center", fontsize=10)
    save_figure(output, "confusion_matrix.png")

    importance = pd.DataFrame({"feature": x_train.columns, "importance": classifier.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    plt.figure()
    sns.barplot(data=importance, x="importance", y="feature", color="#3782b8")
    plt.title("What the classifier learned — feature importance")
    plt.xlabel("Random Forest impurity-based importance")
    plt.ylabel("")
    save_figure(output, "feature_importance.png")
    importance.to_csv(output / "feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="Etiquette", order=["Real", "Fake"], color="#3782b8", ax=axes[0])
    axes[0].set(title="Dataset labels", xlabel="Workshop label", ylabel="Articles")
    body_lengths = df.assign(body_length=df["Corps_Textee"].str.len())
    sns.histplot(data=body_lengths, x="body_length", hue="Etiquette", bins=20, ax=axes[1])
    axes[1].set(title="Article length by label", xlabel="Body characters", ylabel="Articles")
    save_figure(output, "dataset_overview.png")

    plt.figure()
    sns.boxplot(data=df, x="Etiquette", y="Nb_Partages", order=["Real", "Fake"], color="#91b9d7")
    plt.yscale("symlog", linthresh=1)
    plt.title("Sharing activity by workshop label — original values")
    plt.xlabel("Workshop label")
    plt.ylabel("Shares (symmetric log scale)")
    save_figure(output, "sharing_distribution.png")

    count_vectorizer = CountVectorizer(stop_words="english", max_features=20)
    counts = count_vectorizer.fit_transform(df.loc[df["Etiquette"] == "Fake", "Title"])
    frequencies = np.asarray(counts.sum(axis=0)).ravel()
    keywords = sorted(zip(count_vectorizer.get_feature_names_out(), frequencies),
                      key=lambda item: (-item[1], item[0]))[:10]
    pd.DataFrame(keywords, columns=["term", "count"]).to_csv(output / "fake_title_keywords.csv", index=False)

    # This is a short-text/high-sharing heuristic, not evidence of actual bots.
    threshold = float(df["Nb_Partages"].quantile(0.80))
    short_high_share = (df["Nb_Partages"] > threshold) & (df["Corps_Textee"].str.len() < 500)
    plt.figure()
    sns.scatterplot(data=body_lengths, x="body_length", y="Nb_Partages", hue="Etiquette", alpha=0.6)
    plt.axhline(threshold, color="#c55d43", linestyle="--", label="80th percentile of shares")
    plt.axvline(500, color="#555555", linestyle=":", label="500-character heuristic")
    plt.yscale("symlog", linthresh=1)
    plt.title("Exploring short articles with high sharing activity")
    plt.xlabel("Body characters")
    plt.ylabel("Shares (symmetric log scale)")
    plt.legend()
    save_figure(output, "sharing_patterns.png")

    # Exclude every share-derived input when predicting the number of shares.
    regressor = RandomForestRegressor(n_estimators=100, random_state=SEED)
    regressor.fit(x_train[TEXT_FEATURES], train["Nb_Partages"])
    share_predictions = regressor.predict(x_test[TEXT_FEATURES])
    regression_baseline = DummyRegressor(strategy="median")
    regression_baseline.fit(x_train[TEXT_FEATURES], train["Nb_Partages"])
    regression_baseline_predictions = regression_baseline.predict(x_test[TEXT_FEATURES])

    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    text_vectors = tfidf.fit_transform(df["Corps_Textee"])
    # A fixed number of clusters can exceed the distinct text vectors here.
    distinct_vectors = len({
        (tuple(text_vectors[i].indices), tuple(np.round(text_vectors[i].data, 10)))
        for i in range(text_vectors.shape[0])
    })
    n_clusters = min(4, distinct_vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(text_vectors)
    terms = tfidf.get_feature_names_out()
    cluster_keywords = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        indices = np.argsort(center)[::-1]
        cluster_keywords[str(i)] = [str(terms[j]) for j in indices if center[j] > 0][:10]
    plt.figure()
    sns.countplot(data=df.assign(cluster=clusters), x="cluster", hue="Etiquette")
    plt.title("Exploratory TF-IDF / K-Means text groups")
    plt.xlabel("Cluster (arbitrary ID)")
    plt.ylabel("Articles")
    save_figure(output, "text_clusters.png")

    versions = {"python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}
    metrics = {
        "dataset": {"file": data_path.name, "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "rows_after_exact_deduplication": len(df),
                    "label_counts": {str(k): int(v) for k, v in df["Etiquette"].value_counts().items()},
                    "unique_titles": int(df["Title"].nunique()),
                    "unique_bodies": int(df["Corps_Textee"].nunique()),
                    "shared_bodies_across_train_test": len(set(train["Corps_Textee"]) & set(test["Corps_Textee"]))},
        "split": {"random_state": SEED, "stratified": True, "train_rows": len(train), "test_rows": len(test)},
        "classification": {"accuracy": float(accuracy_score(y_test, predictions)),
                           "fake_class_f1": float(f1_score(y_test, predictions, zero_division=0)),
                           "majority_baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
                           "confusion_matrix": matrix.tolist(), "classification_report": report,
                           "train_only_share_cap": share_cap, "features": list(x_train.columns)},
        "sharing_regression": {"mae": float(mean_absolute_error(test["Nb_Partages"], share_predictions)),
                               "median_baseline_mae": float(mean_absolute_error(test["Nb_Partages"], regression_baseline_predictions)),
                               "features": TEXT_FEATURES, "target": "Uncapped Nb_Partages",
                               "note": "Exploratory retrospective regression; not validated as a forecast."},
        "exploration": {"short_high_share_articles": int(short_high_share.sum()),
                        "n_text_clusters": n_clusters, "cluster_keywords": cluster_keywords},
        "environment": versions,
        "limitation": "Template-style workshop data with overlapping text across the random split. Metrics are educational, not real-world validation.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = (
        "Educational holdout results\n"
        f"Dataset: {len(df)} articles; train: {len(train)}; test: {len(test)}\n"
        f"Accuracy: {metrics['classification']['accuracy']:.4f}\n"
        f"Fake-class F1: {metrics['classification']['fake_class_f1']:.4f}\n"
        f"Majority baseline accuracy: {metrics['classification']['majority_baseline_accuracy']:.4f}\n\n"
        + classification_report(y_test, predictions, labels=[0, 1], target_names=["Real", "Fake"], zero_division=0)
        + f"\nSharing regression MAE: {metrics['sharing_regression']['mae']:.2f}\n"
        + f"Median baseline MAE: {metrics['sharing_regression']['median_baseline_mae']:.2f}\n\n"
        + metrics["limitation"] + "\n"
    )
    (output / "classification_report.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fake_news.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_analysis(args.data.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
