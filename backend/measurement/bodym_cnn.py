"""
Steps 6, 8, 9: dual-branch ResNet18 CNN regressor (BMnet-style), trained on
cached front/side silhouette arrays from bodym_data.py.

Usage:
    python bodym_cnn.py --epochs 25 --batch-size 32
        Trains, prints per-epoch train/val loss, and saves the best
        (lowest val loss) checkpoint + normalization stats to
        bodym_output/cnn_best.pt.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from bodym_data import (
    OUTPUT_DIR, TARGET_COLS, SCALAR_COLS, load_merged, subject_split,
    get_or_build_mask_cache, BodyMArrayDataset,
)


class DualBranchRegressor(nn.Module):
    def __init__(self, n_scalars, n_targets):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.front_branch = nn.Sequential(*list(base.children())[:-1])

        base2 = models.resnet18(weights=None)
        base2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.side_branch = nn.Sequential(*list(base2.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512 + 512 + n_scalars, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_targets),
        )

    def forward(self, front, side, scalars):
        f = self.front_branch(front).flatten(1)
        s = self.side_branch(side).flatten(1)
        x = torch.cat([f, s, scalars], dim=1)
        return self.fc(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading + merging train split...")
    train_full = load_merged("train")
    train_df, val_df = subject_split(train_full, test_size=0.15, seed=42)
    print(f"train subjects={train_df['subject_id'].nunique()} rows={len(train_df)}")
    print(f"val   subjects={val_df['subject_id'].nunique()} rows={len(val_df)}")

    print("Building/loading mask caches (front+side, resized 256x256)...")
    t0 = time.time()
    train_front, train_side = get_or_build_mask_cache(train_df, "cnn_train")
    val_front, val_side = get_or_build_mask_cache(val_df, "cnn_val")
    print(f"cache ready in {time.time() - t0:.1f}s")

    # Step 6: normalize targets using train statistics only.
    target_mean = train_df[TARGET_COLS].mean().values
    target_std = train_df[TARGET_COLS].std().values
    np.savez(
        os.path.join(OUTPUT_DIR, "target_norm.npz"),
        mean=target_mean, std=target_std,
    )
    print("target_mean:", dict(zip(TARGET_COLS, target_mean.round(2))))
    print("target_std :", dict(zip(TARGET_COLS, target_std.round(2))))

    train_ds = BodyMArrayDataset(
        train_df, train_front, train_side,
        target_mean=target_mean, target_std=target_std,
    )
    val_ds = BodyMArrayDataset(
        val_df, val_front, val_side,
        target_mean=target_mean, target_std=target_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    model = DualBranchRegressor(
        n_scalars=len(SCALAR_COLS), n_targets=len(TARGET_COLS)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    ckpt_path = os.path.join(OUTPUT_DIR, "cnn_best.pt")
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for front, side, scalars, targets in train_loader:
            front, side = front.to(device), side.to(device)
            scalars, targets = scalars.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(front, side, scalars)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for front, side, scalars, targets in val_loader:
                front, side = front.to(device), side.to(device)
                scalars, targets = scalars.to(device), targets.to(device)
                preds = model(front, side, scalars)
                val_loss += criterion(preds, targets).item()

        train_loss = total_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        dt = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} ({dt:.1f}s)")

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_loss": val_loss, "seconds": dt})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "target_cols": TARGET_COLS,
                "scalar_cols": SCALAR_COLS,
                "target_mean": target_mean,
                "target_std": target_std,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

    with open(os.path.join(OUTPUT_DIR, "cnn_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
