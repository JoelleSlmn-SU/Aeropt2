#!/usr/bin/env python3
"""
Train a mesh pass/fail classifier using SMOTE + Tomek Links.

Expected dataset format
-----------------------
One JSON object per line:

{
    "features": {
        "metric_1": 0.123,
        "metric_2": 4.56
    },
    "label": 0
}

Labels must be 0 or 1. By default:
    0 = failed mesh
    1 = successful mesh

The saved joblib file is compatible with the existing visualisation script:
    {
        "model": fitted_model,
        "feature_order": [...]
    }

Important
---------
SMOTE-Tomek is applied only to the training partition. The test partition
remains untouched so that reported performance is not artificially inflated.
"""

from __future__ import annotations

import os

# Prevent libgomp warnings caused by an empty/invalid cluster environment value.
_raw_omp = os.environ.get("OMP_NUM_THREADS", "").strip()
if not _raw_omp.isdigit() or int(_raw_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split


def load_dataset(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load valid labelled rows from a JSON-lines dataset."""
    rows: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] Skipping invalid JSON on line {line_number}: {exc}"
                )
                continue

            features = row.get("features")
            label = row.get("label")

            if not isinstance(features, dict) or not features:
                print(
                    f"[WARN] Skipping line {line_number}: "
                    "'features' is missing or empty."
                )
                continue

            if label not in (0, 1):
                print(
                    f"[WARN] Skipping line {line_number}: "
                    f"label must be 0 or 1, got {label!r}."
                )
                continue

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid labelled rows were found in {jsonl_path}")

    return rows


def determine_feature_order(rows: list[dict[str, Any]]) -> list[str]:
    """
    Build a deterministic union of all feature names.

    Sorting makes the feature order reproducible between training runs.
    """
    feature_names: set[str] = set()

    for row in rows:
        features = row.get("features") or {}
        feature_names.update(str(name) for name in features.keys())

    if not feature_names:
        raise RuntimeError("No feature names were found in the dataset.")

    return sorted(feature_names)


def _to_float(value: Any) -> float:
    """Convert a feature value to float, returning NaN when conversion fails."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan

    return converted if np.isfinite(converted) else np.nan


def build_feature_matrix(
    rows: list[dict[str, Any]],
    feature_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset rows into a feature matrix and label vector."""
    X = np.asarray(
        [
            [
                _to_float((row.get("features") or {}).get(feature_name, np.nan))
                for feature_name in feature_order
            ]
            for row in rows
        ],
        dtype=float,
    )

    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    return X, y


def validate_dataset(y: np.ndarray, test_size: float) -> None:
    """Check that stratified splitting and SMOTE are possible."""
    counts = Counter(y.tolist())

    if set(counts) != {0, 1}:
        raise RuntimeError(
            "The dataset must contain both class 0 and class 1. "
            f"Observed distribution: {dict(counts)}"
        )

    if min(counts.values()) < 2:
        raise RuntimeError(
            "At least two samples are required in each class before splitting. "
            f"Observed distribution: {dict(counts)}"
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError("--test_size must be between 0 and 1.")


def choose_smote_neighbours(y_train: np.ndarray, requested_k: int) -> int:
    """
    Select a valid SMOTE k-neighbours value.

    SMOTE requires:
        k_neighbors < number of minority training samples
    """
    counts = Counter(y_train.tolist())
    minority_count = min(counts.values())

    if minority_count < 2:
        raise RuntimeError(
            "The training split contains fewer than two minority-class samples. "
            "Reduce --test_size or add more minority-class data."
        )

    return max(1, min(int(requested_k), minority_count - 1))


def probability_for_label(
    model: Pipeline,
    X: np.ndarray,
    label: int,
) -> np.ndarray:
    """Return predicted probabilities for a requested class label."""
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)

    matching = np.where(classes == int(label))[0]
    if matching.size != 1:
        raise RuntimeError(
            f"Could not locate label {label} in model classes {classes.tolist()}."
        )

    return probabilities[:, int(matching[0])]


def compute_metrics(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    failure_label: int,
) -> dict[str, Any]:
    """Evaluate the fitted model on the untouched test partition."""
    y_pred = model.predict(X_test)
    p_failure = probability_for_label(model, X_test, failure_label)
    y_failure = (y_test == failure_label).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "failure_label": int(failure_label),
        "failure_precision": float(
            precision_score(
                y_test,
                y_pred,
                pos_label=failure_label,
                zero_division=0,
            )
        ),
        "failure_recall": float(
            recall_score(
                y_test,
                y_pred,
                pos_label=failure_label,
                zero_division=0,
            )
        ),
        "failure_f1": float(
            f1_score(
                y_test,
                y_pred,
                pos_label=failure_label,
                zero_division=0,
            )
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["label_0", "label_1"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if np.unique(y_failure).size == 2:
        metrics["failure_roc_auc"] = float(
            roc_auc_score(y_failure, p_failure)
        )
        metrics["failure_average_precision"] = float(
            average_precision_score(y_failure, p_failure)
        )
    else:
        metrics["failure_roc_auc"] = None
        metrics["failure_average_precision"] = None

    return metrics



def save_evaluation_plots(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    outdir: Path,
    positive_label: int,
    threshold: float = 0.5,
) -> None:
    """Save confusion-matrix, ROC and precision-recall plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    probabilities = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)

    pos_match = np.where(classes == int(positive_label))[0]
    if pos_match.size != 1:
        raise RuntimeError(
            f"Positive label {positive_label} is not present in model classes "
            f"{classes.tolist()}."
        )

    pos_index = int(pos_match[0])
    p_pos = probabilities[:, pos_index]
    y_pos = (y_test == int(positive_label)).astype(int)

    # Threshold the requested positive-class probability explicitly.
    negative_label = 1 - int(positive_label)
    y_pred = np.where(
        p_pos >= float(threshold),
        int(positive_label),
        negative_label,
    )

    # Confusion matrix in the original label convention.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail (0)", "Pass (1)"],
    ).plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    # ROC curve for the requested positive label.
    if np.unique(y_pos).size == 2:
        fpr, tpr, _ = roc_curve(y_pos, p_pos)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve: label {positive_label}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(outdir / "roc_curve.png", dpi=220)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_pos, p_pos)
        ap_score = average_precision_score(y_pos, p_pos)

        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.plot(recall, precision, label=f"Average precision = {ap_score:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-recall curve: label {positive_label}")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(outdir / "pr_curve.png", dpi=220)
        plt.close(fig)
    else:
        print(
            "[WARN] ROC and precision-recall plots were skipped because the "
            "test set contains only one class."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a mesh classifier using SMOTE-Tomek."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset.jsonl or a directory containing dataset.jsonl.",
    )
    parser.add_argument(
        "--model",
        default="model.joblib",
        help="Output path for the trained joblib model.",
    )
    parser.add_argument(
        "--outdir",
        default="classifier_plots",
        help="Directory in which evaluation plots are saved.",
    )
    parser.add_argument(
        "--positive_label",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Positive class used for ROC/PR plotting. Defaults to "
            "--failure_label."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the requested positive label.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "Optional output path for a JSON training report. "
            "Defaults to <model_stem>_report.json."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Use the same value in the visualisation script.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help=(
            "Fraction reserved as untouched test data. "
            "Use the same value in the visualisation script."
        ),
    )
    parser.add_argument(
        "--failure_label",
        type=int,
        choices=[0, 1],
        default=0,
        help="Label representing a failed mesh.",
    )
    parser.add_argument(
        "--smote_k",
        type=int,
        default=5,
        help="Requested SMOTE neighbour count; reduced automatically if needed.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Optional maximum random-forest tree depth.",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=1,
        help="Minimum samples required at a random-forest leaf.",
    )
    args = parser.parse_args()

    dataset_arg = Path(args.dataset).expanduser().resolve()
    jsonl_path = (
        dataset_arg / "dataset.jsonl"
        if dataset_arg.is_dir()
        else dataset_arg
    )

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    model_path = Path(args.model).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else model_path.with_name(f"{model_path.stem}_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    positive_label = (
        int(args.failure_label)
        if args.positive_label is None
        else int(args.positive_label)
    )

    rows = load_dataset(jsonl_path)
    feature_order = determine_feature_order(rows)
    X, y = build_feature_matrix(rows, feature_order)

    validate_dataset(y, args.test_size)

    print(f"[INFO] Dataset: {jsonl_path}")
    print(f"[INFO] Valid samples: {len(y)}")
    print(f"[INFO] Features: {len(feature_order)}")
    print(f"[INFO] Full class distribution: {dict(Counter(y.tolist()))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y,
    )

    print(
        "[INFO] Original training distribution: "
        f"{dict(Counter(y_train.tolist()))}"
    )
    print(
        "[INFO] Untouched test distribution: "
        f"{dict(Counter(y_test.tolist()))}"
    )

    smote_k = choose_smote_neighbours(y_train, args.smote_k)
    print(f"[INFO] Using SMOTE k_neighbors={smote_k}")

    sampler = SMOTETomek(
        smote=SMOTE(
            sampling_strategy="auto",
            random_state=int(args.seed),
            k_neighbors=smote_k,
        ),
        sampling_strategy="auto",
        random_state=int(args.seed),
    )

    classifier = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=args.max_depth,
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote_tomek", sampler),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    metrics = compute_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        failure_label=int(args.failure_label),
    )

    # Recover the actual class distribution after imputation and resampling
    # for reporting only. The fitted model itself already performed these steps.
    X_train_imputed = model.named_steps["imputer"].transform(X_train)
    _, y_train_resampled = model.named_steps["smote_tomek"].fit_resample(
        X_train_imputed,
        y_train,
    )
    resampled_distribution = dict(Counter(y_train_resampled.tolist()))

    model_package = {
        "model": model,
        "feature_order": feature_order,
        "failure_label": int(args.failure_label),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "training_distribution": dict(Counter(y_train.tolist())),
        "resampled_training_distribution": resampled_distribution,
    }
    joblib.dump(model_package, model_path)

    save_evaluation_plots(
        model=model,
        X_test=X_test,
        y_test=y_test,
        outdir=outdir,
        positive_label=positive_label,
        threshold=float(args.threshold),
    )

    report = {
        "dataset": str(jsonl_path),
        "model": str(model_path),
        "plot_directory": str(outdir),
        "positive_label": int(positive_label),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "feature_count": len(feature_order),
        "sample_count": int(len(y)),
        "full_distribution": dict(Counter(y.tolist())),
        "training_distribution_before_resampling": dict(
            Counter(y_train.tolist())
        ),
        "training_distribution_after_smote_tomek": resampled_distribution,
        "test_distribution_untouched": dict(Counter(y_test.tolist())),
        "smote_k_neighbors": int(smote_k),
        "metrics": metrics,
    }

    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(
        "[INFO] Resampled training distribution: "
        f"{resampled_distribution}"
    )
    print("\nConfusion matrix, rows=true and columns=predicted, labels=[0, 1]:")
    print(np.asarray(metrics["confusion_matrix_labels_0_1"]))

    print("\nFailure-class test metrics:")
    print(f"  Precision: {metrics['failure_precision']:.4f}")
    print(f"  Recall:    {metrics['failure_recall']:.4f}")
    print(f"  F1:        {metrics['failure_f1']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")

    if metrics["failure_average_precision"] is not None:
        print(
            "  PR average precision: "
            f"{metrics['failure_average_precision']:.4f}"
        )
    if metrics["failure_roc_auc"] is not None:
        print(f"  ROC AUC:   {metrics['failure_roc_auc']:.4f}")

    print(f"\n[OK] Saved model:  {model_path}")
    print(f"[OK] Saved report: {report_path}")
    print(f"[OK] Saved plots:  {outdir}")


if __name__ == "__main__":
    main()