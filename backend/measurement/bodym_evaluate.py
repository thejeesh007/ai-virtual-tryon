"""
Step 10: evaluate the baseline (RF) and CNN models side by side on testA
and testB, and report the testA/testB MAE gap (domain-gap indicator).

Usage:
    python bodym_evaluate.py
        Requires bodym_output/baseline_rf.joblib and/or
        bodym_output/cnn_best.pt to already exist (run bodym_baseline.py /
        bodym_cnn.py first). Whichever is missing is skipped with a note.
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bodym_data import OUTPUT_DIR, TARGET_COLS, SCALAR_COLS, load_merged, get_or_build_mask_cache, BodyMArrayDataset
from bodym_baseline import build_features
from bodym_cnn import DualBranchRegressor


def evaluate_baseline(testA, testB):
    path = os.path.join(OUTPUT_DIR, "baseline_rf.joblib")
    if not os.path.exists(path):
        print("No baseline model found, skipping (run bodym_baseline.py first).")
        return None

    bundle = joblib.load(path)
    model = bundle["model"]
    target_cols = bundle["target_cols"]
    scalar_cols = bundle["scalar_cols"]

    rows = {}
    for name, df in [("testA", testA), ("testB", testB)]:
        X = build_features(df, scalar_cols=scalar_cols)
        y = df[target_cols].values.astype(float)
        preds = model.predict(X)
        rows[name] = np.abs(preds - y).mean(axis=0)

    return pd.DataFrame(rows, index=target_cols).T


def evaluate_cnn(testA, testB):
    path = os.path.join(OUTPUT_DIR, "cnn_best.pt")
    if not os.path.exists(path):
        print("No CNN checkpoint found, skipping (run bodym_cnn.py first).")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    target_cols = ckpt["target_cols"]
    scalar_cols = ckpt["scalar_cols"]
    target_mean = ckpt["target_mean"]
    target_std = ckpt["target_std"]

    model = DualBranchRegressor(n_scalars=len(scalar_cols), n_targets=len(target_cols)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = {}
    for name, df in [("testA", testA), ("testB", testB)]:
        front, side = get_or_build_mask_cache(df, name)
        ds = BodyMArrayDataset(df, front, side, target_cols=target_cols,
                                scalar_cols=scalar_cols)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for f, s, sc, t in loader:
                f, s, sc = f.to(device), s.to(device), sc.to(device)
                preds = model(f, s, sc).cpu().numpy()
                preds = preds * target_std + target_mean  # de-normalize
                all_preds.append(preds)
                all_targets.append(t.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        rows[name] = np.abs(preds - targets).mean(axis=0)

    return pd.DataFrame(rows, index=target_cols).T


def main():
    print("Loading testA/testB...")
    testA = load_merged("testA")
    testB = load_merged("testB")

    baseline_table = evaluate_baseline(testA, testB)
    cnn_table = evaluate_cnn(testA, testB)

    if baseline_table is not None:
        print("\n=== Baseline (RandomForest) MAE per measurement (cm) ===")
        print(baseline_table.round(2))
        print("Mean MAE:", baseline_table.mean(axis=1).round(2).to_dict())
        gap = baseline_table.loc["testB"].mean() - baseline_table.loc["testA"].mean()
        print(f"testA/testB domain gap (mean MAE): {gap:+.2f} cm")

    if cnn_table is not None:
        print("\n=== CNN (DualBranchRegressor) MAE per measurement (cm) ===")
        print(cnn_table.round(2))
        print("Mean MAE:", cnn_table.mean(axis=1).round(2).to_dict())
        gap = cnn_table.loc["testB"].mean() - cnn_table.loc["testA"].mean()
        print(f"testA/testB domain gap (mean MAE): {gap:+.2f} cm")

    if baseline_table is not None and cnn_table is not None:
        combined = pd.concat(
            {"baseline": baseline_table, "cnn": cnn_table}, axis=0
        )
        combined.to_csv(os.path.join(OUTPUT_DIR, "results_table.csv"))
        print(f"\nSaved combined results table to {os.path.join(OUTPUT_DIR, 'results_table.csv')}")


if __name__ == "__main__":
    main()
