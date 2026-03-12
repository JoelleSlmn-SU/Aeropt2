#!/usr/bin/env python3
"""
Visualise a mesh-failure classifier:
- Confusion matrix
- ROC curve
- Precision-Recall curve (recommended for imbalanced data)

Assumes:
- model.joblib produced by train_classifier.py (contains {"model": clf, "feature_order": [...]})
- dataset.jsonl in the same schema as train_classifier.py expects
"""
import json
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

def load_dataset(jsonl_path: Path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            feats = r.get("features") or {}
            lab = r.get("label", None)
            if not isinstance(feats, dict) or len(feats) == 0:
                continue
            if lab not in (0, 1):
                continue
            rows.append(r)
    return rows

def build_feature_matrix(rows, feature_order):
    X = np.array([[float((r.get("features") or {}).get(k, 0.0)) for k in feature_order] for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    return X, y

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset.jsonl OR directory containing it")
    ap.add_argument("--model", required=True, help="Path to model.joblib from train_classifier.py")
    ap.add_argument("--outdir", default=".", help="Where to save figures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5, help="p(class=1) threshold")
    ap.add_argument("--positive_label", type=int, default=0,
                    help="Which class is the 'failure' class for reporting (0 or 1).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_arg = Path(args.dataset)
    jsonl_path = (dataset_arg / "dataset.jsonl") if dataset_arg.is_dir() else dataset_arg
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {jsonl_path}")

    pack = joblib.load(args.model)
    clf = pack["model"]
    feature_order = pack["feature_order"]

    rows = load_dataset(jsonl_path)
    X, y = build_feature_matrix(rows, feature_order)

    # Stratified split (same spirit as train_classifier.py)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # probabilities for class 1
    proba = clf.predict_proba(Xte)
    one_idx = int(np.where(clf.classes_ == 1)[0][0])
    p1 = proba[:, one_idx]
    yhat = (p1 >= float(args.threshold)).astype(int)

    # If your “failure” class is label 0, flip for ROC/PR plotting convenience
    # (ROC/PR typically treat positive label = 1)
    if int(args.positive_label) == 0:
        y_pos = (yte == 0).astype(int)
        p_pos = 1.0 - p1  # P(failure) = 1 - P(success)
        pos_name = "Failure (label=0)"
        # For confusion matrix: keep original labels
    else:
        y_pos = (yte == 1).astype(int)
        p_pos = p1
        pos_name = "Failure (label=1)"

    # --- Confusion Matrix (using original labels) ---
    cm = confusion_matrix(yte, yhat, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure(figsize=(5.2, 4.6))
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (threshold={args.threshold:.2f})")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=220)
    plt.close()

    # --- ROC Curve (positive = failure) ---
    fpr, tpr, _ = roc_curve(y_pos, p_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5.6, 4.6))
    plt.plot(fpr, tpr, label=f"{pos_name} ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=220)
    plt.close()

    # --- Precision-Recall (positive = failure) ---
    prec, rec, _ = precision_recall_curve(y_pos, p_pos)
    ap_score = average_precision_score(y_pos, p_pos)

    plt.figure(figsize=(5.6, 4.6))
    plt.plot(rec, prec, label=f"{pos_name} AP = {ap_score:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=220)
    plt.close()

    print("Saved:")
    print(" -", outdir / "confusion_matrix.png")
    print(" -", outdir / "roc_curve.png")
    print(" -", outdir / "pr_curve.png")

if __name__ == "__main__":
    main()