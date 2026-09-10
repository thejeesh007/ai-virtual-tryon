"""
Step 7: baseline model. Width profiles (front + side silhouette) + height_cm
-> RandomForestRegressor. This is the comparison point / safety net for the
CNN in bodym_cnn.py.

Usage:
    python bodym_baseline.py
        Builds features for train/val/testA/testB, fits the RF, prints an
        MAE table, and saves the fitted model + feature config to
        bodym_output/baseline_rf.joblib.
"""

import argparse
import os
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from bodym_data import (
    OUTPUT_DIR, TARGET_COLS, SCALAR_COLS, load_merged, subject_split,
)

MODELS = {
    "rf": lambda: MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    ),
    "hgb": lambda: MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, random_state=42)
    ),
}

N_SLICES = 20


def extract_width_profile(mask_path, n_slices=N_SLICES):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    h = binary.shape[0]
    widths = []
    for i in range(n_slices):
        row = binary[int(i * h / n_slices), :]
        cols = np.where(row > 0)[0]
        widths.append(cols.max() - cols.min() if len(cols) > 0 else 0)
    return widths


def build_features(df, scalar_cols=SCALAR_COLS):
    front_feats = np.array([extract_width_profile(p) for p in df["mask_path"]])
    side_feats = np.array([extract_width_profile(p) for p in df["mask_left_path"]])
    scalars = df[scalar_cols].values.astype(float)
    return np.hstack([front_feats, side_feats, scalars])


def mae_table(model, splits_features, splits_targets, target_cols=TARGET_COLS):
    rows = {}
    for name, X in splits_features.items():
        y = splits_targets[name]
        preds = model.predict(X)
        mae = np.abs(preds - y).mean(axis=0)
        rows[name] = mae
    return pd.DataFrame(rows, index=target_cols).T


def get_or_build_features(df, cache_name):
    cache_path = os.path.join(OUTPUT_DIR, f"baseline_features_{cache_name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X"], data["y"]

    X = build_features(df)
    y = df[TARGET_COLS].values.astype(float)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS.keys()), default="rf",
                    help="rf = RandomForestRegressor (default), hgb = HistGradientBoostingRegressor")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading + merging splits...")
    train_full = load_merged("train")
    testA = load_merged("testA")
    testB = load_merged("testB")

    train_df, val_df = subject_split(train_full, test_size=0.15, seed=42)
    print(f"train subjects={train_df['subject_id'].nunique()} rows={len(train_df)}")
    print(f"val   subjects={val_df['subject_id'].nunique()} rows={len(val_df)}")

    splits = {"train": train_df, "val": val_df, "testA": testA, "testB": testB}

    print("\nExtracting width-profile features (front + side + height_cm) [cached after first run]...")
    t0 = time.time()
    features, targets = {}, {}
    for name, df in splits.items():
        features[name], targets[name] = get_or_build_features(df, name)
    print(f"done in {time.time() - t0:.1f}s. feature dim = {features['train'].shape[1]}")

    print(f"\nFitting {args.model} (MultiOutput)...")
    model = MODELS[args.model]()
    t0 = time.time()
    model.fit(features["train"], targets["train"])
    print(f"fit in {time.time() - t0:.1f}s")

    table = mae_table(model, features, targets)
    print(f"\n=== Baseline ({args.model}) MAE per measurement (cm) ===")
    print(table.round(2))
    print("\nMean MAE across measurements:")
    print(table.mean(axis=1).round(2))

    out_model = os.path.join(OUTPUT_DIR, f"baseline_{args.model}.joblib")
    joblib.dump({
        "model": model,
        "target_cols": TARGET_COLS,
        "scalar_cols": SCALAR_COLS,
        "n_slices": N_SLICES,
    }, out_model)
    print(f"\nSaved model to {out_model}")

    table.to_csv(os.path.join(OUTPUT_DIR, f"baseline_{args.model}_mae.csv"))


if __name__ == "__main__":
    main()
