"""
Steps 1-2 of the BodyM regression pipeline: merge metadata and sanity-check it.

Each BodyM split (train/testA/testB) ships three CSVs keyed by subject_id
(subject_to_photo_map, hwg_metadata, measurements) plus a mask/ and
mask_left/ folder of silhouette PNGs keyed by photo_id. A photo_id always
exists in *both* mask/ and mask_left/ (front + left-profile silhouettes of
the same pose), and a subject_id can have multiple photo_ids (different
poses of the same person, same target measurements).

Usage:
    python bodym_data.py
        Merges all three splits, prints sanity-check stats, saves
        merged_<split>.csv into bodym_output/, and saves a visual
        sanity-check grid of sample masks to bodym_output/sanity_check.png.
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
BODYM_DIR = os.path.join(PROJECT_ROOT, "data", "bodym")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "bodym_output")

SPLITS = ["train", "testA", "testB"]

# All of measurements.csv's columns except "height" -- that one is a known
# input (SCALAR_COLS below), not something to predict from the photos.
TARGET_COLS = [
    "chest", "waist", "hip", "shoulder-breadth", "arm-length",
    "ankle", "bicep", "calf", "forearm", "leg-length",
    "shoulder-to-crotch", "thigh", "wrist",
]

# Known-input scalar(s) fed alongside the images (matches deployment: the
# user tells the app their height).
SCALAR_COLS = ["height_cm"]

IMG_SIZE = 256

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])


def split_dir(split):
    return os.path.join(BODYM_DIR, split)


def load_merged(split):
    """Step 1: merge photo map + hwg metadata + measurements for one split."""

    d = split_dir(split)

    pmap = pd.read_csv(os.path.join(d, "subject_to_photo_map.csv"))
    hwg = pd.read_csv(os.path.join(d, "hwg_metadata.csv")).rename(
        columns={"height_cm": "height_cm"}
    )
    meas = pd.read_csv(os.path.join(d, "measurements.csv"))

    df = pmap.merge(hwg, on="subject_id").merge(
        meas, on="subject_id", suffixes=("", "_measured")
    )

    df["mask_path"] = df["photo_id"].apply(
        lambda pid: os.path.join(d, "mask", f"{pid}.png")
    )
    df["mask_left_path"] = df["photo_id"].apply(
        lambda pid: os.path.join(d, "mask_left", f"{pid}.png")
    )

    return df


def subject_split(df, test_size=0.15, seed=42):
    """
    Step 4, done correctly: split by subject_id first, then assign rows,
    so the same subject's repeated poses never leak across train/val.
    """

    from sklearn.model_selection import train_test_split

    subjects = df["subject_id"].unique()
    train_subjects, val_subjects = train_test_split(
        subjects, test_size=test_size, random_state=seed
    )

    train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)

    return train_df, val_df


class BodyMDataset(Dataset):
    """Step 5: front mask + side(left) mask + scalar(s) -> target measurements."""

    def __init__(self, dataframe, target_cols=TARGET_COLS, scalar_cols=SCALAR_COLS,
                 target_mean=None, target_std=None):
        self.df = dataframe.reset_index(drop=True)
        self.target_cols = target_cols
        self.scalar_cols = scalar_cols
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        front = transform(Image.open(row["mask_path"]).convert("L"))
        side = transform(Image.open(row["mask_left_path"]).convert("L"))

        scalars = torch.tensor(
            row[self.scalar_cols].values.astype(float), dtype=torch.float32
        )
        targets = row[self.target_cols].values.astype(float)

        if self.target_mean is not None:
            targets = (targets - self.target_mean) / self.target_std

        targets = torch.tensor(targets, dtype=torch.float32)

        return front, side, scalars, targets


def get_or_build_mask_cache(df, cache_name, img_size=IMG_SIZE):
    """
    Pre-decode + resize every front/side mask for `df` once and cache to
    disk as uint8 arrays. Re-decoding full-res PNGs from disk every epoch
    (as a naive PIL Dataset would) dominates CNN training time; reading
    from an in-memory array is orders of magnitude faster.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DIR, f"cache_{cache_name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["front"], data["side"]

    front = np.zeros((len(df), img_size, img_size), dtype=np.uint8)
    side = np.zeros((len(df), img_size, img_size), dtype=np.uint8)

    for i, row in enumerate(df.itertuples()):
        f = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        front[i] = cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA)

        s = cv2.imread(row.mask_left_path, cv2.IMREAD_GRAYSCALE)
        side[i] = cv2.resize(s, (img_size, img_size), interpolation=cv2.INTER_AREA)

    np.savez_compressed(cache_path, front=front, side=side)

    return front, side


class BodyMArrayDataset(Dataset):
    """Same as BodyMDataset, but reads from pre-cached in-memory uint8 arrays."""

    def __init__(self, dataframe, front_arr, side_arr, target_cols=TARGET_COLS,
                 scalar_cols=SCALAR_COLS, target_mean=None, target_std=None):
        self.df = dataframe.reset_index(drop=True)
        self.front_arr = front_arr
        self.side_arr = side_arr
        self.target_cols = target_cols
        self.scalar_cols = scalar_cols
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        front = torch.from_numpy(self.front_arr[idx]).float().unsqueeze(0) / 255.0
        side = torch.from_numpy(self.side_arr[idx]).float().unsqueeze(0) / 255.0

        scalars = torch.tensor(
            row[self.scalar_cols].values.astype(float), dtype=torch.float32
        )
        targets = row[self.target_cols].values.astype(float)

        if self.target_mean is not None:
            targets = (targets - self.target_mean) / self.target_std

        targets = torch.tensor(targets, dtype=torch.float32)

        return front, side, scalars, targets


def sanity_check():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged = {}

    for split in SPLITS:
        df = load_merged(split)
        merged[split] = df

        out_csv = os.path.join(OUTPUT_DIR, f"merged_{split}.csv")
        df.to_csv(out_csv, index=False)

        print(f"\n=== {split} ===")
        print("shape:", df.shape)
        print("unique subjects:", df["subject_id"].nunique(), "| rows (poses):", len(df))
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        print("columns with nulls:" if len(nulls) else "no nulls.")
        if len(nulls):
            print(nulls)
        missing_front = (~df["mask_path"].apply(os.path.exists)).sum()
        missing_side = (~df["mask_left_path"].apply(os.path.exists)).sum()
        print(f"missing mask files: front={missing_front} side={missing_side}")
        print("target stats:\n", df[TARGET_COLS].describe().loc[["mean", "std", "min", "max"]])

    # Visual sanity check: 2 samples, front + side mask side by side.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = merged["train"]
    fig, axes = plt.subplots(2, 2, figsize=(6, 8))
    for i in range(2):
        row = df.iloc[i]
        front_img = Image.open(row["mask_path"])
        side_img = Image.open(row["mask_left_path"])
        axes[i][0].imshow(front_img, cmap="gray")
        axes[i][0].set_title(f"{row['subject_id'][:8]} front")
        axes[i][0].axis("off")
        axes[i][1].imshow(side_img, cmap="gray")
        axes[i][1].set_title(f"{row['subject_id'][:8]} side")
        axes[i][1].axis("off")

    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "sanity_check.png")
    fig.savefig(out_png)
    print(f"\nSaved visual sanity check to {out_png}")


if __name__ == "__main__":
    sanity_check()
