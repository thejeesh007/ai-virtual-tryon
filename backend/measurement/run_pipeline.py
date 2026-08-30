"""
Full body-measurement pipeline for every person in data/dataset/.

For each person_XXX folder containing front.png + side.png this:

  1. Runs MediaPipe (to get a person bounding box) + HMR2 (SMPL mesh
     reconstruction) on both photos.
  2. Scales each reconstructed mesh to the person's real height.
  3. Derives anatomical Y-levels (shoulder/chest/waist/hip) from the
     front mesh's SMPL joints.
  4. Removes arm vertices (SMPL skinning weights) and geometrically
     measures shoulder/chest/waist/hip circumference by combining the
     front mesh's width with the side mesh's depth.

All intermediate/final artifacts are saved into hmr2_output/ using the
same "<person>_<...>" naming already used for person_001.

Usage:
    python run_pipeline.py
        Prompts for each person's height interactively.

    python run_pipeline.py --heights person_001=172,person_002=170
        Supplies heights for one or more people non-interactively;
        anyone not listed is still prompted for.

    python run_pipeline.py --force
        Re-runs HMR2 reconstruction even if cached outputs already
        exist for a person (by default, a person already fully
        processed is skipped).
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import mediapipe as mp

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "4D-Humans"))

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset

from mesh_utils import (
    get_torso_faces,
    find_clean_level,
    find_waist_level,
    ellipse_circumference,
)

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "backend", "measurement", "hmr2_output")

mp_pose = mp.solutions.pose


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the body measurement pipeline for every person in data/dataset."
    )

    parser.add_argument(
        "--heights",
        type=str,
        default="",
        help="Comma-separated person=height_cm pairs, e.g. person_001=172,person_002=170. "
             "Anyone not listed here is prompted for interactively.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run reconstruction even if this person already has cached measurements.",
    )

    return parser.parse_args()


def parse_heights_arg(heights_arg):

    heights = {}

    for pair in heights_arg.split(","):

        pair = pair.strip()

        if not pair:
            continue

        person, value = pair.split("=")

        heights[person.strip()] = float(value.strip())

    return heights


# =========================================================
# DATASET DISCOVERY
# =========================================================

def discover_people(dataset_dir):

    people = []

    for name in sorted(os.listdir(dataset_dir)):

        person_dir = os.path.join(dataset_dir, name)

        if not os.path.isdir(person_dir):
            continue

        front_path = os.path.join(person_dir, "front.png")
        side_path = os.path.join(person_dir, "side.png")

        if os.path.exists(front_path) and os.path.exists(side_path):
            people.append(name)
        else:
            print(f"Skipping {name}: missing front.png/side.png")

    return people


def already_processed(person):

    measurements_path = os.path.join(
        OUTPUT_DIR, f"{person}_measurements.npy"
    )

    return os.path.exists(measurements_path)


def get_height_for(person, cli_heights):

    if person in cli_heights:
        return cli_heights[person]

    while True:

        raw = input(
            f"Enter actual height in cm for {person}: "
        ).strip()

        try:
            return float(raw)
        except ValueError:
            print("Please enter a number, e.g. 172")


# =========================================================
# MEDIAPIPE BOUNDING BOX
# =========================================================

def get_mediapipe_bbox(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise RuntimeError("MediaPipe could not detect a person.")

    h, w = image.shape[:2]

    points = []

    for landmark in result.pose_landmarks.landmark:

        if landmark.visibility < 0.3:
            continue

        points.append((landmark.x * w, landmark.y * h))

    if len(points) < 5:
        raise RuntimeError("Not enough visible MediaPipe landmarks.")

    points = np.array(points)

    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    width = x_max - x_min
    height = y_max - y_min

    x1 = max(0, x_min - width * 0.20)
    y1 = max(0, y_min - height * 0.15)
    x2 = min(w - 1, x_max + width * 0.20)
    y2 = min(h - 1, y_max + height * 0.15)

    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


# =========================================================
# HMR2 RECONSTRUCTION FOR ONE IMAGE
# =========================================================

def reconstruct(model, model_cfg, device, image_path):

    img_cv2 = cv2.imread(image_path)

    if img_cv2 is None:
        raise RuntimeError(f"Could not read {image_path}")

    bbox = get_mediapipe_bbox(img_cv2)

    dataset = ViTDetDataset(model_cfg, img_cv2, bbox)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    with torch.no_grad():

        for batch in dataloader:

            batch = recursive_to(batch, device)
            output = model(batch)

    vertices = (
        output["pred_vertices"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    keypoints_3d = (
        output["pred_keypoints_3d"][0]
        .detach()
        .cpu()
        .numpy()
    )

    return vertices, keypoints_3d


def save_obj(path, vertices, faces):

    with open(path, "w") as f:

        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def scale_to_height(vertices, actual_height_cm):

    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

    reconstructed_height = y_max - y_min

    scale = actual_height_cm / reconstructed_height

    return vertices * scale, scale


# =========================================================
# ANATOMICAL LEVELS (from the front mesh's SMPL joints)
# =========================================================

NECK = 12
RIGHT_SHOULDER = 17
LEFT_SHOULDER = 16
MID_HIP = 0
RIGHT_HIP = 2
LEFT_HIP = 1


def compute_anatomical_levels(front_vertices, J_regressor):

    joints = J_regressor @ front_vertices

    shoulder_y = (
        joints[RIGHT_SHOULDER][1] + joints[LEFT_SHOULDER][1]
    ) / 2.0

    mid_hip_y = joints[MID_HIP][1]

    hip_y = (
        joints[RIGHT_HIP][1] + joints[LEFT_HIP][1]
    ) / 2.0

    torso_length = shoulder_y - mid_hip_y

    chest_y = shoulder_y - 0.35 * torso_length
    waist_y = shoulder_y - 0.68 * torso_length

    return {
        "shoulder": float(shoulder_y),
        "chest": float(chest_y),
        "waist": float(waist_y),
        "hip": float(hip_y),
    }


# =========================================================
# GEOMETRIC MEASUREMENT (arm-filtered, front width + side depth)
# =========================================================

def measure_level(front_vertices, side_vertices, torso_faces, y_estimate, is_waist, chest_y=None, hip_y=None):

    if is_waist:
        y, front_result, side_result = find_waist_level(
            front_vertices, side_vertices, torso_faces, chest_y, hip_y,
        )
    else:
        y, front_result, side_result = find_clean_level(
            front_vertices, side_vertices, torso_faces, y_estimate,
        )

    if front_result is None or side_result is None:
        return None

    width = front_result["width"]
    depth = side_result["width"]

    return {
        "y": y,
        "y_offset": y - y_estimate,
        "circumference": ellipse_circumference(width, depth),
        "width": width,
        "depth": depth,
        "front_contours": front_result["num_contours"],
        "side_contours": side_result["num_contours"],
    }


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()
    cli_heights = parse_heights_arg(args.heights)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    people = discover_people(DATASET_DIR)

    if not people:
        print(f"No person_XXX folders with front.png + side.png found under {DATASET_DIR}")
        return

    print("People found:", ", ".join(people))

    print("\nLoading HMR2...")
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("HMR2 loaded on", device)

    faces = np.asarray(model.smpl.faces)
    lbs_weights = model.smpl.lbs_weights.detach().cpu().numpy()
    torso_faces = get_torso_faces(faces, lbs_weights)
    J_regressor = model.smpl.J_regressor.detach().cpu().numpy()

    all_results = {}

    for person in people:

        print("\n" + "=" * 60)
        print(person)
        print("=" * 60)

        if already_processed(person) and not args.force:
            print("Already processed (use --force to redo). Loading cached measurements.")
            all_results[person] = np.load(
                os.path.join(OUTPUT_DIR, f"{person}_measurements.npy"),
                allow_pickle=True,
            ).item()
            continue

        height_cm = get_height_for(person, cli_heights)

        person_dir = os.path.join(DATASET_DIR, person)

        view_vertices = {}
        view_keypoints = {}

        for view in ["front", "side"]:

            print(f"Reconstructing {person}/{view}.png ...")

            raw_vertices, keypoints_3d = reconstruct(
                model, model_cfg, device,
                os.path.join(person_dir, f"{view}.png"),
            )

            np.save(
                os.path.join(OUTPUT_DIR, f"{person}_{view}_vertices.npy"),
                raw_vertices,
            )

            np.save(
                os.path.join(OUTPUT_DIR, f"{person}_{view}_keypoints_3d.npy"),
                keypoints_3d,
            )

            save_obj(
                os.path.join(OUTPUT_DIR, f"{person}_{view}.obj"),
                raw_vertices,
                faces,
            )

            scaled_vertices, scale = scale_to_height(raw_vertices, height_cm)

            np.save(
                os.path.join(OUTPUT_DIR, f"{person}_{view}_scaled.npy"),
                scaled_vertices,
            )

            print(f"  scale={scale:.4f}  height_check={scaled_vertices[:,1].max()-scaled_vertices[:,1].min():.2f} cm")

            view_vertices[view] = scaled_vertices
            view_keypoints[view] = keypoints_3d

        front_vertices = view_vertices["front"]
        side_vertices = view_vertices["side"]

        levels = compute_anatomical_levels(front_vertices, J_regressor)

        np.save(
            os.path.join(OUTPUT_DIR, f"{person}_anatomical_levels.npy"),
            levels,
            allow_pickle=True,
        )

        print("\nAnatomical levels:")
        for name, y in levels.items():
            print(f"  {name:10s}: {y:.2f} cm")

        results = {}

        for name in ["shoulder", "chest", "waist", "hip"]:

            result = measure_level(
                front_vertices, side_vertices, torso_faces,
                levels[name],
                is_waist=(name == "waist"),
                chest_y=levels["chest"],
                hip_y=levels["hip"],
            )

            if result is None:
                print(f"\n{name.upper()}: could not find a clean torso ring.")
                continue

            results[name] = result

            print(f"\n{name.upper()}")
            print(f"  Y level       : {result['y']:.2f} cm (offset {result['y_offset']:+.2f} cm)")
            print(f"  Circumference : {result['circumference']:.2f} cm")
            print(f"  Width (front) : {result['width']:.2f} cm")
            print(f"  Depth (side)  : {result['depth']:.2f} cm")
            print(f"  Contours      : front={result['front_contours']} side={result['side_contours']}")

        np.save(
            os.path.join(OUTPUT_DIR, f"{person}_measurements.npy"),
            results,
            allow_pickle=True,
        )

        all_results[person] = results

        print(f"\nSaved measurements for {person}.")

    print("\n" + "=" * 60)
    print("SUMMARY - ALL PEOPLE")
    print("=" * 60)

    print(
        f"{'Person':<14}"
        f"{'Shoulder':>10}"
        f"{'Chest':>10}"
        f"{'Waist':>10}"
        f"{'Hip':>10}"
    )

    for person, results in all_results.items():

        def c(name):
            return f"{results[name]['circumference']:.1f}" if name in results else "n/a"

        print(
            f"{person:<14}"
            f"{c('shoulder'):>10}"
            f"{c('chest'):>10}"
            f"{c('waist'):>10}"
            f"{c('hip'):>10}"
        )


if __name__ == "__main__":
    main()
