"""
Module 2 (Avatar Generation): fit SMPL shape parameters so the resulting
mesh's own measurements match a person's predicted measurements (from the
HGB baseline in bodym_baseline.py) as closely as possible, producing a
personalized 3D avatar -- per the project report's Sub-Problem 2 ("fit SMPL
parametric 3D body model to estimated measurements").

Targets chest/waist/hip/shoulder-breadth/arm-length -- the 5 measurements
mesh_utils.py already has extraction logic for. The remaining 8 BodyM
measurements (ankle/bicep/calf/forearm/leg-length/shoulder-to-crotch/
thigh/wrist) aren't wired up yet; extend measure_smpl_mesh() to add them
later if needed.

Requires an SMPL model file at smpl_models/smpl/SMPL_NEUTRAL.pkl (license-
gated, not in git -- see .gitignore). Copy your own SMPL_NEUTRAL.pkl there.

Usage:
    python smpl_fit.py --height 167 --chest 85 --waist 75 --hip 90 \
        --shoulder-breadth 38 --arm-length 52 --out person_007_avatar.obj
"""

import argparse
import os

import numpy as np
import smplx
import torch
from scipy.optimize import minimize

from mesh_utils import get_torso_faces, measure_level, ellipse_circumference

MODEL_DIR = os.path.join(os.path.dirname(__file__), "smpl_models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "bodym_output")

# Standard SMPL 24-joint skeleton (matches mesh_utils.py's constants).
PELVIS = 0
LEFT_HIP = 1
RIGHT_HIP = 2
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_WRIST = 20
RIGHT_WRIST = 21

N_FIT_BETAS = 6  # first 6 shape components capture most size/build variation;
                 # remaining betas fixed at 0 to keep a 5-measurement fit well-posed.


def load_smpl_model():
    model_path = os.path.join(MODEL_DIR, "smpl", "SMPL_NEUTRAL.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SMPL model not found at {model_path}. Copy your own "
            f"SMPL_NEUTRAL.pkl there (license-gated, see .gitignore)."
        )
    return smplx.create(MODEL_DIR, model_type="smpl", gender="neutral")


def forward(model, betas_np):
    betas = torch.zeros(1, model.num_betas, dtype=torch.float32)
    betas[0, :len(betas_np)] = torch.tensor(betas_np, dtype=torch.float32)

    with torch.no_grad():
        output = model(
            betas=betas,
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
        )

    vertices = output.vertices[0].numpy().astype(np.float64)
    joints = output.joints[0].numpy().astype(np.float64)
    return vertices, joints


def scale_to_height(vertices, joints, height_cm):
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    current_height = y_max - y_min
    scale = height_cm / current_height
    return vertices * scale, joints * scale


def compute_anatomical_levels(joints):
    shoulder_y = (joints[LEFT_SHOULDER][1] + joints[RIGHT_SHOULDER][1]) / 2.0
    hip_y = (joints[LEFT_HIP][1] + joints[RIGHT_HIP][1]) / 2.0
    mid_hip_y = joints[PELVIS][1]

    torso_length = shoulder_y - mid_hip_y

    chest_y = shoulder_y - 0.35 * torso_length
    waist_y = shoulder_y - 0.68 * torso_length

    return {"shoulder": shoulder_y, "chest": chest_y, "waist": waist_y, "hip": hip_y}


def find_level_single(vertices, faces, y_start, max_offset=5.0, step=0.5):
    """Like mesh_utils.find_clean_level, but for one complete mesh (no
    front/side cross-validation needed -- a clean synthetic SMPL mesh
    almost always gives one contour on the first try)."""

    offsets = [0.0]
    n_steps = int(max_offset / step)
    for i in range(1, n_steps + 1):
        offsets.append(i * step)
        offsets.append(-i * step)

    for offset in offsets:
        y = y_start + offset
        result = measure_level(vertices, faces, y)
        if result is not None and result["num_contours"] == 1:
            return result

    return None


def find_waist_single(vertices, faces, y_start, y_end, num_samples=15):
    best_result = None
    best_circumference = np.inf

    for y in np.linspace(y_start, y_end, num_samples):
        result = measure_level(vertices, faces, y)
        if result is None or result["num_contours"] != 1:
            continue
        if result["circumference"] < best_circumference:
            best_circumference = result["circumference"]
            best_result = result

    return best_result


def measure_smpl_mesh(model, torso_faces, betas_np, height_cm):
    vertices, joints = forward(model, betas_np)
    vertices, joints = scale_to_height(vertices, joints, height_cm)

    levels = compute_anatomical_levels(joints)

    chest = find_level_single(vertices, torso_faces, levels["chest"])
    hip = find_level_single(vertices, torso_faces, levels["hip"])
    waist = find_waist_single(vertices, torso_faces, levels["chest"], levels["hip"])

    shoulder_breadth = np.linalg.norm(joints[LEFT_SHOULDER] - joints[RIGHT_SHOULDER])
    arm_length = (
        np.linalg.norm(joints[LEFT_SHOULDER] - joints[LEFT_WRIST])
        + np.linalg.norm(joints[RIGHT_SHOULDER] - joints[RIGHT_WRIST])
    ) / 2.0

    return {
        "chest": chest["circumference"] if chest else None,
        "waist": waist["circumference"] if waist else None,
        "hip": hip["circumference"] if hip else None,
        "shoulder-breadth": float(shoulder_breadth),
        "arm-length": float(arm_length),
    }, vertices


def fit_betas(model, torso_faces, target, height_cm, n_betas=N_FIT_BETAS):
    target_cols = ["chest", "waist", "hip", "shoulder-breadth", "arm-length"]
    target_vec = np.array([target[c] for c in target_cols])

    eval_count = [0]

    def loss(betas_np):
        eval_count[0] += 1
        measured, _ = measure_smpl_mesh(model, torso_faces, betas_np, height_cm)
        pred_vec = np.array([
            measured[c] if measured[c] is not None else target[c] + 50.0  # penalize failed slices
            for c in target_cols
        ])
        err = pred_vec - target_vec
        print(f"  eval {eval_count[0]:3d}: betas={np.round(betas_np, 2)} "
              f"pred={np.round(pred_vec, 1)} loss={np.sum(err ** 2):.2f}")
        return np.sum(err ** 2)

    x0 = np.zeros(n_betas)
    result = minimize(loss, x0, method="Powell",
                       options={"maxiter": 60, "xtol": 0.05, "ftol": 0.5})

    return result.x, result


def save_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--chest", type=float, required=True)
    p.add_argument("--waist", type=float, required=True)
    p.add_argument("--hip", type=float, required=True)
    p.add_argument("--shoulder-breadth", type=float, required=True)
    p.add_argument("--arm-length", type=float, required=True)
    p.add_argument("--out", type=str, default="avatar.obj")
    return p.parse_args()


def main():
    args = parse_args()

    target = {
        "chest": args.chest,
        "waist": args.waist,
        "hip": args.hip,
        "shoulder-breadth": args.shoulder_breadth,
        "arm-length": args.arm_length,
    }

    print("Loading SMPL model...")
    model = load_smpl_model()

    faces = model.faces
    lbs_weights = model.lbs_weights.detach().numpy()
    torso_faces = get_torso_faces(faces, lbs_weights)
    print(f"faces total={len(faces)} torso(arm-filtered)={len(torso_faces)}")

    print(f"\nTarget measurements: {target} (height={args.height}cm)")
    print("\nFitting betas...")
    betas, result = fit_betas(model, torso_faces, target, args.height)

    print(f"\nConverged: {result.success}, final loss={result.fun:.2f}, "
          f"evals={result.nfev}")
    print(f"Fitted betas (first {N_FIT_BETAS}): {np.round(betas, 3)}")

    final_measured, vertices = measure_smpl_mesh(model, torso_faces, betas, args.height)
    print("\nFinal mesh measurements vs target:")
    for k in target:
        m = final_measured[k]
        print(f"  {k:20s}: measured={m:.1f}  target={target[k]:.1f}  "
              f"diff={abs(m - target[k]):.2f}" if m is not None else
              f"  {k:20s}: measurement failed")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.out)
    save_obj(out_path, vertices, faces)
    print(f"\nSaved avatar mesh to {out_path}")


if __name__ == "__main__":
    main()
