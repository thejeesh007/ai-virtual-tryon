"""
Step 12: accuracy / precision / recall for the body-measurement models.

Precision and recall are CLASSIFICATION metrics (they need a discrete label
per example -- true positive / false positive / false negative). RF, HGB,
and the CNN are all REGRESSION models: they output a continuous cm value,
not a class. Reporting "precision" on a raw cm prediction isn't meaningful
-- there's no natural positive/negative outcome to count.

This script gives two honest things instead of a fabricated third:

  1. The correct regression metrics (MAE, RMSE, R^2) per measurement, per
     model -- what these models actually are.
  2. A real classification task DERIVED from this project's own size
     system: bucket predicted and actual chest_cm into the same S/M/L/XL/
     XXL/XXXL boundaries fabric_ease_reference.py's STANDARD_BODY_
     MEASUREMENTS_CM already uses (midpoints between consecutive sizes'
     chest values), then score "which size bucket would this prediction
     have put someone in" as an actual multi-class classification problem
     -- accuracy, precision, recall, F1 all apply cleanly here, because
     there's now a real discrete label (the size bucket) being predicted
     and compared.

Usage:
    python bodym_classification_metrics.py
        Requires baseline_rf.joblib / baseline_hgb.joblib / cnn_best.pt to
        already exist (run bodym_baseline.py --model rf, bodym_baseline.py
        --model hgb, bodym_cnn.py first). Whichever is missing is skipped.
        Prints a full report and saves it to
        bodym_output/classification_metrics_report.csv
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
)
from torch.utils.data import DataLoader

from bodym_data import OUTPUT_DIR, TARGET_COLS, load_merged, get_or_build_mask_cache, BodyMArrayDataset
from bodym_baseline import build_features
from bodym_cnn import DualBranchRegressor

# Same boundaries the rest of this project already uses for tops sizing
# (fabric_ease_reference.py's STANDARD_BODY_MEASUREMENTS_CM chest column:
# S=91, M=97, L=103, XL=109, XXL=115, XXXL=121) -- bucket edges are the
# midpoint between each consecutive pair, not an arbitrary choice.
SIZE_LABELS = ["S", "M", "L", "XL", "XXL", "XXXL"]
SIZE_CHEST_CENTERS = [91, 97, 103, 109, 115, 121]
SIZE_EDGES = [-np.inf] + [
    (SIZE_CHEST_CENTERS[i] + SIZE_CHEST_CENTERS[i + 1]) / 2
    for i in range(len(SIZE_CHEST_CENTERS) - 1)
] + [np.inf]


def to_size_bucket(chest_cm):
    idx = np.digitize(chest_cm, SIZE_EDGES[1:-1])
    return np.array(SIZE_LABELS)[idx]


def regression_report(preds, targets, target_cols):
    rows = {}
    for i, col in enumerate(target_cols):
        p, t = preds[:, i], targets[:, i]
        rows[col] = {
            "MAE_cm": mean_absolute_error(t, p),
            "RMSE_cm": mean_squared_error(t, p) ** 0.5,
            "R2": r2_score(t, p),
        }
    return pd.DataFrame(rows).T


def classification_report_for_chest(pred_chest, true_chest, model_name):
    pred_bucket = to_size_bucket(pred_chest)
    true_bucket = to_size_bucket(true_chest)

    acc = accuracy_score(true_bucket, pred_bucket)
    prec_macro = precision_score(true_bucket, pred_bucket, average="macro", zero_division=0)
    rec_macro = recall_score(true_bucket, pred_bucket, average="macro", zero_division=0)
    f1_macro = f1_score(true_bucket, pred_bucket, average="macro", zero_division=0)
    prec_weighted = precision_score(true_bucket, pred_bucket, average="weighted", zero_division=0)
    rec_weighted = recall_score(true_bucket, pred_bucket, average="weighted", zero_division=0)
    f1_weighted = f1_score(true_bucket, pred_bucket, average="weighted", zero_division=0)

    print(f"\n=== {model_name}: size-bucket classification (from predicted chest_cm) ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (macro/weighted): {prec_macro:.3f} / {prec_weighted:.3f}")
    print(f"Recall    (macro/weighted): {rec_macro:.3f} / {rec_weighted:.3f}")
    print(f"F1        (macro/weighted): {f1_macro:.3f} / {f1_weighted:.3f}")
    print("\nPer-size breakdown:")
    print(classification_report(true_bucket, pred_bucket, labels=SIZE_LABELS, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(true_bucket, pred_bucket, labels=SIZE_LABELS)
    print(pd.DataFrame(cm, index=SIZE_LABELS, columns=SIZE_LABELS))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
        "precision_weighted": prec_weighted, "recall_weighted": rec_weighted, "f1_weighted": f1_weighted,
    }


def predict_rf_or_hgb(model_name, testA, testB):
    path = os.path.join(OUTPUT_DIR, f"baseline_{model_name}.joblib")
    if not os.path.exists(path):
        print(f"No {model_name} model found at {path}, skipping (train it first).")
        return None

    bundle = joblib.load(path)
    model, target_cols, scalar_cols = bundle["model"], bundle["target_cols"], bundle["scalar_cols"]

    out = {}
    for name, df in [("testA", testA), ("testB", testB)]:
        X = build_features(df, scalar_cols=scalar_cols)
        y = df[target_cols].values.astype(float)
        preds = model.predict(X)
        out[name] = (preds, y, target_cols)
    return out


def predict_cnn(testA, testB):
    path = os.path.join(OUTPUT_DIR, "cnn_best.pt")
    if not os.path.exists(path):
        print(f"No CNN checkpoint found at {path}, skipping (train it first).")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target_cols, scalar_cols = ckpt["target_cols"], ckpt["scalar_cols"]
    target_mean, target_std = ckpt["target_mean"], ckpt["target_std"]

    model = DualBranchRegressor(n_scalars=len(scalar_cols), n_targets=len(target_cols)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out = {}
    for name, df in [("testA", testA), ("testB", testB)]:
        front, side = get_or_build_mask_cache(df, name)
        ds = BodyMArrayDataset(df, front, side, target_cols=target_cols, scalar_cols=scalar_cols)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for f, s, sc, t in loader:
                f, s, sc = f.to(device), s.to(device), sc.to(device)
                p = model(f, s, sc).cpu().numpy()
                p = p * target_std + target_mean
                all_preds.append(p)
                all_targets.append(t.numpy())
        out[name] = (np.concatenate(all_preds), np.concatenate(all_targets), target_cols)
    return out


def main():
    print("Loading testA/testB...")
    testA = load_merged("testA")
    testB = load_merged("testB")

    results = {}
    results["rf"] = predict_rf_or_hgb("rf", testA, testB)
    results["hgb"] = predict_rf_or_hgb("hgb", testA, testB)
    results["cnn"] = predict_cnn(testA, testB)

    classification_rows = []
    for model_name, res in results.items():
        if res is None:
            continue

        for split in ["testA", "testB"]:
            preds, targets, target_cols = res[split]
            print(f"\n{'=' * 60}\n{model_name.upper()} -- {split} -- regression metrics\n{'=' * 60}")
            print(regression_report(preds, targets, target_cols).round(3))

        # Classification (size bucket) is scored on testA -- the
        # in-domain split, same one used for the headline accuracy
        # numbers elsewhere in this project's reporting.
        preds, targets, target_cols = res["testA"]
        chest_idx = target_cols.index("chest")
        row = classification_report_for_chest(preds[:, chest_idx], targets[:, chest_idx], model_name.upper())
        row["split"] = "testA"
        classification_rows.append(row)

    if classification_rows:
        out_path = os.path.join(OUTPUT_DIR, "classification_metrics_report.csv")
        pd.DataFrame(classification_rows).to_csv(out_path, index=False)
        print(f"\nSaved size-bucket classification summary to {out_path}")


if __name__ == "__main__":
    main()
