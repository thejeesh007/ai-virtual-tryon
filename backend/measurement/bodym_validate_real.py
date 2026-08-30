"""
Step 11: validate the trained models on real photos in data/dataset/person_XXX.

These are real RGB photos, not BodyM's synthetic silhouettes, so this
script first segments each photo into a silhouette (MediaPipe Selfie
Segmentation) and re-frames it onto a 720x960 canvas using the same
bbox-height/canvas-height and framing ratios measured from real BodyM
training masks, so the pixel-scale the models were trained on is roughly
preserved, before feeding it to the baseline and CNN models.

There is no tape-measured ground truth for these people in the repo, so by
default this reports *predictions*, not MAE. Pass --ground-truth-csv to
also compute per-person MAE against known measurements.

Usage:
    python bodym_validate_real.py --heights person_001=170,person_002=165
    python bodym_validate_real.py --heights person_001=170,person_002=165 \
        --ground-truth-csv my_ground_truth.csv
"""

import argparse
import os

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import torch

from bodym_data import OUTPUT_DIR, TARGET_COLS, load_merged
from bodym_baseline import extract_width_profile
from bodym_cnn import DualBranchRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
REAL_MASK_DIR = os.path.join(OUTPUT_DIR, "real_masks")

CANVAS_W, CANVAS_H = 720, 960

mp_selfie = mp.solutions.selfie_segmentation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heights", type=str, required=True,
                    help="Comma-separated person=height_cm pairs, e.g. person_001=170,person_002=165")
    p.add_argument("--people", type=str, default="",
                    help="Comma-separated person folder names to run. Default: all people listed in --heights.")
    p.add_argument("--ground-truth-csv", type=str, default="",
                    help="Optional CSV with columns: person,chest,waist,hip,shoulder-breadth,arm-length")
    return p.parse_args()


def parse_heights(s):
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=")
        out[k.strip()] = float(v.strip())
    return out


# =========================================================
# Reference framing, measured from real BodyM training masks
# =========================================================

def compute_reference_framing(n_samples=150, seed=0):
    df = load_merged("train").sample(n=n_samples, random_state=seed)

    stats = {"front": {"h": [], "top": [], "cx": []},
             "side": {"h": [], "top": [], "cx": []}}

    for _, row in df.iterrows():
        for view, path in [("front", row["mask_path"]), ("side", row["mask_left_path"])]:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            ys, xs = np.where(img > 127)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            stats[view]["h"].append((y1 - y0) / H)
            stats[view]["top"].append(y0 / H)
            stats[view]["cx"].append(((x0 + x1) / 2) / W)

    ref = {}
    for view in ["front", "side"]:
        ref[view] = {
            "height_frac": float(np.mean(stats[view]["h"])),
            "top_frac": float(np.mean(stats[view]["top"])),
            "cx_frac": float(np.mean(stats[view]["cx"])),
        }

    return ref


# =========================================================
# Segmentation + reframing
# =========================================================

def segment_silhouette(image_bgr, segmenter):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)
    mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Clean up small speckle noise.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def reframe_to_bodym(mask, ref, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        raise RuntimeError("Segmentation found no person in this photo.")

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    ch, cw = crop.shape

    target_h = ref["height_frac"] * canvas_h
    scale = target_h / ch
    new_w, new_h = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    new_w, new_h = min(new_w, canvas_w), min(new_h, canvas_h)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    top = int(round(ref["top_frac"] * canvas_h))
    cx = int(round(ref["cx_frac"] * canvas_w))
    left = cx - new_w // 2

    top = max(0, min(top, canvas_h - new_h))
    left = max(0, min(left, canvas_w - new_w))

    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]


def find_image(person_dir, stem):
    for ext in IMAGE_EXTS:
        path = os.path.join(person_dir, f"{stem}{ext}")
        if os.path.exists(path):
            return path
    return None


def build_real_masks(person, ref, segmenter):
    person_dir = os.path.join(DATASET_DIR, person)
    front_file = find_image(person_dir, "front")
    side_file = find_image(person_dir, "side")

    if front_file is None or side_file is None:
        raise RuntimeError(f"Missing front/side image for {person} (looked for {IMAGE_EXTS})")

    front_bgr = cv2.imread(front_file)
    side_bgr = cv2.imread(side_file)

    if front_bgr is None or side_bgr is None:
        raise RuntimeError(f"Could not read front/side image for {person}")

    front_mask = reframe_to_bodym(segment_silhouette(front_bgr, segmenter), ref["front"])
    side_mask = reframe_to_bodym(segment_silhouette(side_bgr, segmenter), ref["side"])

    os.makedirs(REAL_MASK_DIR, exist_ok=True)
    front_path = os.path.join(REAL_MASK_DIR, f"{person}_front_mask.png")
    side_path = os.path.join(REAL_MASK_DIR, f"{person}_side_mask.png")
    cv2.imwrite(front_path, front_mask)
    cv2.imwrite(side_path, side_mask)

    return front_path, side_path


# =========================================================
# Inference
# =========================================================

def predict_baseline(front_path, side_path, height_cm):
    model_path = os.path.join(OUTPUT_DIR, "baseline_rf.joblib")
    if not os.path.exists(model_path):
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    target_cols = bundle["target_cols"]

    front_feat = extract_width_profile(front_path, n_slices=bundle["n_slices"])
    side_feat = extract_width_profile(side_path, n_slices=bundle["n_slices"])
    X = np.array([front_feat + side_feat + [height_cm]])

    preds = model.predict(X)[0]
    return dict(zip(target_cols, preds))


def predict_cnn(front_path, side_path, height_cm):
    ckpt_path = os.path.join(OUTPUT_DIR, "cnn_best.pt")
    if not os.path.exists(ckpt_path):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    target_cols = ckpt["target_cols"]
    target_mean, target_std = ckpt["target_mean"], ckpt["target_std"]

    model = DualBranchRegressor(n_scalars=1, n_targets=len(target_cols)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def load_tensor(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t.to(device)

    front_t = load_tensor(front_path)
    side_t = load_tensor(side_path)
    scalars_t = torch.tensor([[height_cm]], dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(front_t, side_t, scalars_t).cpu().numpy()[0]

    preds = preds * target_std + target_mean
    return dict(zip(target_cols, preds))


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    heights = parse_heights(args.heights)

    people = [p.strip() for p in args.people.split(",") if p.strip()] or list(heights.keys())

    gt = None
    if args.ground_truth_csv:
        gt = pd.read_csv(args.ground_truth_csv).set_index("person")

    print("Computing reference framing stats from BodyM training masks...")
    ref = compute_reference_framing()
    print("front:", {k: round(v, 3) for k, v in ref["front"].items()})
    print("side :", {k: round(v, 3) for k, v in ref["side"].items()})

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        results = {}

        for person in people:
            if person not in heights:
                print(f"Skipping {person}: no height supplied via --heights")
                continue

            print(f"\n=== {person} (height={heights[person]}cm) ===")
            front_path, side_path = build_real_masks(person, ref, segmenter)
            print(f"Saved silhouettes to {front_path}, {side_path}")

            baseline_pred = predict_baseline(front_path, side_path, heights[person])
            cnn_pred = predict_cnn(front_path, side_path, heights[person])

            if baseline_pred:
                print("Baseline prediction (cm):",
                      {k: round(v, 1) for k, v in baseline_pred.items()})
            else:
                print("Baseline model not found (run bodym_baseline.py first).")

            if cnn_pred:
                print("CNN prediction (cm):     ",
                      {k: round(v, 1) for k, v in cnn_pred.items()})
            else:
                print("CNN checkpoint not found (run bodym_cnn.py first, or wait for it to finish).")

            results[person] = {"baseline": baseline_pred, "cnn": cnn_pred}

            if gt is not None and person in gt.index:
                truth = gt.loc[person][TARGET_COLS].astype(float)
                print("Ground truth (cm):        ", truth.round(1).to_dict())
                if baseline_pred:
                    err = {k: round(abs(baseline_pred[k] - truth[k]), 2) for k in TARGET_COLS}
                    print("Baseline abs error (cm):  ", err)
                if cnn_pred:
                    err = {k: round(abs(cnn_pred[k] - truth[k]), 2) for k in TARGET_COLS}
                    print("CNN abs error (cm):       ", err)

    print("\nDone.")


if __name__ == "__main__":
    main()
