"""
5-fold (subject-grouped) cross-validation + ensembling for the HGB baseline,
to test whether it beats the single-split HGB/RF numbers from bodym_baseline.py.

Two things are measured:
  1. Out-of-fold CV MAE: each row is predicted only by a model that never
     saw its subject during training -> an unbiased estimate of accuracy
     on BodyM's train distribution (comparable to the old "val" number).
  2. Ensembled testA/testB MAE: the 5 fold-models' predictions on testA/testB
     are averaged together -> directly comparable to the single-HGB numbers
     already measured, to see if ensembling actually helps.

Usage:
    python bodym_baseline_kfold.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor

from bodym_data import OUTPUT_DIR, TARGET_COLS, load_merged, subject_split
from bodym_baseline import get_or_build_features

N_FOLDS = 5


def make_model():
    return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, random_state=42))


def main():
    print("Reconstructing train_full features from existing caches...")
    train_full = load_merged("train")
    train_df, val_df = subject_split(train_full, test_size=0.15, seed=42)

    X_train, y_train = get_or_build_features(train_df, "train")
    X_val, y_val = get_or_build_features(val_df, "val")
    X_full = np.vstack([X_train, X_val])
    y_full = np.vstack([y_train, y_val])
    subjects_full = pd.concat(
        [train_df["subject_id"], val_df["subject_id"]], ignore_index=True
    ).values

    testA_df = load_merged("testA")
    testB_df = load_merged("testB")
    X_testA, y_testA = get_or_build_features(testA_df, "testA")
    X_testB, y_testB = get_or_build_features(testB_df, "testB")

    print(f"train_full: {X_full.shape[0]} rows, {len(set(subjects_full))} unique subjects")

    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_preds = np.zeros_like(y_full)
    testA_fold_preds = []
    testB_fold_preds = []

    for fold, (tr_idx, oof_idx) in enumerate(gkf.split(X_full, y_full, groups=subjects_full)):
        print(f"\nFold {fold + 1}/{N_FOLDS}: train={len(tr_idx)} oof={len(oof_idx)}")
        model = make_model()
        model.fit(X_full[tr_idx], y_full[tr_idx])

        oof_preds[oof_idx] = model.predict(X_full[oof_idx])
        testA_fold_preds.append(model.predict(X_testA))
        testB_fold_preds.append(model.predict(X_testB))

    oof_mae = np.abs(oof_preds - y_full).mean(axis=0)
    testA_ensemble = np.mean(testA_fold_preds, axis=0)
    testB_ensemble = np.mean(testB_fold_preds, axis=0)
    testA_mae = np.abs(testA_ensemble - y_testA).mean(axis=0)
    testB_mae = np.abs(testB_ensemble - y_testB).mean(axis=0)

    table = pd.DataFrame(
        {"5fold_oof(~val)": oof_mae, "5fold_ensemble_testA": testA_mae, "5fold_ensemble_testB": testB_mae},
        index=TARGET_COLS,
    ).T

    print("\n=== 5-fold HGB: MAE per measurement (cm) ===")
    print(table.round(2))
    print("\nMean MAE across measurements:")
    print(table.mean(axis=1).round(2))

    table.to_csv(os.path.join(OUTPUT_DIR, "baseline_hgb_5fold_mae.csv"))
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'baseline_hgb_5fold_mae.csv')}")


if __name__ == "__main__":
    main()
