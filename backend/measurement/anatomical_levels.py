import os
import sys
import numpy as np
import torch

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

HMR2_ROOT = os.path.join(
    PROJECT_ROOT,
    "third_party",
    "4D-Humans"
)

sys.path.insert(0, HMR2_ROOT)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "measurement",
    "hmr2_output"
)

# =========================================================
# HMR2
# =========================================================

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT

print("Loading HMR2...")

download_models(CACHE_DIR_4DHUMANS)

model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

print("HMR2 loaded.")

# =========================================================
# LOAD SCALED MESH
# =========================================================

mesh_path = os.path.join(
    OUTPUT_DIR,
    "person_001_front_scaled.npy"
)

vertices = np.load(mesh_path)

print("\nMesh:", vertices.shape)

# =========================================================
# LOAD HMR2 25-JOINT MAPPING
# =========================================================

# From the mapping you already inspected:
#
# HMR2 0  = Nose
# HMR2 1  = Neck
# HMR2 2  = RightShoulder
# HMR2 3  = RightElbow
# HMR2 4  = RightWrist
# HMR2 5  = LeftShoulder
# HMR2 6  = LeftElbow
# HMR2 7  = LeftWrist
# HMR2 8  = MidHip
# HMR2 9  = RightHip
# HMR2 10 = RightKnee
# HMR2 11 = RightAnkle
# HMR2 12 = LeftHip
# HMR2 13 = LeftKnee
# HMR2 14 = LeftAnkle

# =========================================================
# GET JOINTS DIRECTLY FROM SMPL
# =========================================================

J_regressor = model.smpl.J_regressor.detach().cpu().numpy()

# SMPL joints
joints = J_regressor @ vertices

# SMPL indices from your HMR2 mapping
NECK = 12
RIGHT_SHOULDER = 17
LEFT_SHOULDER = 16
MID_HIP = 0
RIGHT_HIP = 2
LEFT_HIP = 1

neck = joints[NECK]

right_shoulder = joints[RIGHT_SHOULDER]
left_shoulder = joints[LEFT_SHOULDER]

mid_hip = joints[MID_HIP]

right_hip = joints[RIGHT_HIP]
left_hip = joints[LEFT_HIP]

# =========================================================
# ANATOMICAL REFERENCE LEVELS
# =========================================================

shoulder_y = (
    right_shoulder[1]
    + left_shoulder[1]
) / 2.0

hip_y = (
    right_hip[1]
    + left_hip[1]
) / 2.0

mid_hip_y = mid_hip[1]

neck_y = neck[1]

# =========================================================
# DEFINE BODY HEIGHT
# =========================================================

body_height = (
    vertices[:, 1].max()
    - vertices[:, 1].min()
)

print("\n========================================")
print("ANATOMICAL JOINT LEVELS")
print("========================================")

print(
    f"Neck Y             : {neck_y:.2f} cm"
)

print(
    f"Shoulder Y         : {shoulder_y:.2f} cm"
)

print(
    f"Right shoulder Y   : {right_shoulder[1]:.2f} cm"
)

print(
    f"Left shoulder Y    : {left_shoulder[1]:.2f} cm"
)

print(
    f"MidHip Y           : {mid_hip_y:.2f} cm"
)

print(
    f"Right hip Y        : {right_hip[1]:.2f} cm"
)

print(
    f"Left hip Y         : {left_hip[1]:.2f} cm"
)

print(
    f"Body height        : {body_height:.2f} cm"
)

# =========================================================
# IMPORTANT:
# CHEST AND WAIST ARE NOT DIRECT SMPL JOINTS
# =========================================================
#
# Therefore we DO NOT use the real measurements to
# determine them.
#
# We define them anatomically as positions between
# shoulder and hip landmarks.
#
# These are INITIAL anatomical estimates.
#
# We will later validate whether these levels correspond
# correctly to the human anatomical regions.
# =========================================================

torso_length = shoulder_y - mid_hip_y

# Because Y increases upward in our reconstructed mesh:
#
# shoulder_y > chest_y > waist_y > hip_y
#
# Initial anatomical locations:

chest_y = (
    shoulder_y
    - 0.35 * torso_length
)

waist_y = (
    shoulder_y
    - 0.68 * torso_length
)

# =========================================================
# PRINT FINAL LEVELS
# =========================================================

print("\n========================================")
print("INITIAL MEASUREMENT LEVELS")
print("========================================")

print(
    f"Shoulder : {shoulder_y:.2f} cm"
)

print(
    f"Chest    : {chest_y:.2f} cm"
)

print(
    f"Waist    : {waist_y:.2f} cm"
)

print(
    f"Hip      : {hip_y:.2f} cm"
)

# =========================================================
# SAVE LEVELS
# =========================================================

levels = {
    "shoulder": float(shoulder_y),
    "chest": float(chest_y),
    "waist": float(waist_y),
    "hip": float(hip_y)
}

levels_path = os.path.join(
    OUTPUT_DIR,
    "person_001_anatomical_levels.npy"
)

np.save(
    levels_path,
    levels,
    allow_pickle=True
)

print("\nSaved:")
print(levels_path)

print("\n========================================")
print("ANATOMICAL LEVEL DETECTION COMPLETED")
print("========================================")