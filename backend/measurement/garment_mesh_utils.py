"""
Real-garment measurement extraction, for garment meshes from the MGN
(Multi-Garment Network) dataset -- data/garments/mgn/.

Each MGN subject folder has a garment mesh (e.g. TShirtNoCoat.obj,
Pants.obj) registered in the SAME coordinate space as that subject's own
body scan (smpl_registered.obj) -- confirmed by comparing cross-sectional
widths at matching heights, garment consistently sits just outside the
body's own silhouette. This lets us measure a garment's real chest/waist/
hip the same way mesh_utils.py measures a body: slice at a height,
find the closed contour, take its perimeter -- then compute this
garment's real ease as garment_measurement - that_subject's_own_body_measurement,
instead of a manually-typed ease value.
"""

import os
import pickle

import numpy as np
import torch

from mesh_utils import measure_level, get_torso_faces
from smpl_fit import load_smpl_model, compute_anatomical_levels

MGN_DIR = os.path.join(
    os.path.dirname(__file__), "../../data/garments/mgn/Multi-Garment_dataset"
)

_smpl_model = None
_smpl_torso_faces = None


def _get_smpl():
    """Lazily load once -- avoid reloading the ~87MB model file per call."""
    global _smpl_model, _smpl_torso_faces
    if _smpl_model is None:
        _smpl_model = load_smpl_model()
        lbs_weights = _smpl_model.lbs_weights.detach().numpy()
        _smpl_torso_faces = get_torso_faces(_smpl_model.faces, lbs_weights)
    return _smpl_model, _smpl_torso_faces


def posed_smpl_body(betas, pose, trans):
    """Regenerate the clean, standard-topology SMPL mesh for this exact
    subject from their registration.pkl parameters -- same body shape/pose
    MGN used to register the garment, but with working skinning weights so
    arm-filtering (get_torso_faces) actually works, unlike the raw
    27k-vertex scan mesh which has no per-vertex joint weights available
    and mixes arm+torso into one contour at chest height."""

    model, torso_faces = _get_smpl()

    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    global_orient = torch.tensor(pose[:3], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(pose[3:72], dtype=torch.float32).unsqueeze(0)
    transl = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(
            betas=betas_t, body_pose=body_pose,
            global_orient=global_orient, transl=transl,
        )

    vertices = output.vertices[0].numpy().astype(np.float64)
    joints = output.joints[0].numpy().astype(np.float64)
    return vertices, joints, torso_faces


def load_obj(path):
    """Minimal OBJ loader: vertices + triangulated faces (0-indexed).
    Handles v/vt/vn-style face references, ignores texture/normal indices."""

    vertices = []
    faces = []

    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(idx)

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def find_level_single(vertices, faces, y_start, max_offset=0.05, step=0.005):
    """Slice at y_start. measure_level() already picks the largest (by
    perimeter) contour at that height, which is reliably the main torso/
    limb ring even when a slice also clips something smaller (a sleeve
    edge, a stray fold) -- no need to demand a single contour. Only search
    nearby offsets as a fallback if the exact level misses the mesh
    entirely (returns None), e.g. right at a hem edge."""

    result = measure_level(vertices, faces, y_start)
    if result is not None:
        return result

    n_steps = int(max_offset / step)
    for i in range(1, n_steps + 1):
        for offset in (i * step, -i * step):
            result = measure_level(vertices, faces, y_start + offset)
            if result is not None:
                return result

    return None


def list_subjects():
    if not os.path.isdir(MGN_DIR):
        return []
    return sorted(
        d for d in os.listdir(MGN_DIR) if os.path.isdir(os.path.join(MGN_DIR, d))
    )


def subject_garments(subject_id):
    """Which garment files this subject has, e.g. {'TShirtNoCoat': path}."""
    subject_dir = os.path.join(MGN_DIR, subject_id)
    garments = {}
    for fname in os.listdir(subject_dir):
        if fname.endswith(".obj") and fname not in ("scan.obj", "smpl_registered.obj"):
            garments[fname[:-4]] = os.path.join(subject_dir, fname)
    return garments


def measure_garment_and_body(subject_id, garment_name, levels=("chest", "waist", "hip")):
    """
    levels: which of compute_anatomical_levels()'s Y-levels to measure at
    -- these come from this specific subject's own shoulder/hip *joint*
    positions (same method used everywhere else in this project), not a
    guessed fraction of total height, so they land in the right place
    regardless of this subject's individual proportions or pose.

    Returns {level_name: {"garment_cm": .., "body_cm": .., "ease_cm": ..}}
    for whichever levels the garment mesh actually covers.
    """

    subject_dir = os.path.join(MGN_DIR, subject_id)

    with open(os.path.join(subject_dir, "registration.pkl"), "rb") as f:
        reg = pickle.load(f, encoding="latin1")
    body_vertices, body_joints, body_faces = posed_smpl_body(reg["betas"], reg["pose"], reg["trans"])
    anatomical_levels = compute_anatomical_levels(body_joints)

    garment_path = subject_garments(subject_id)[garment_name]
    garment_vertices, garment_faces = load_obj(garment_path)

    results = {}
    for level_name in levels:
        y = anatomical_levels[level_name]

        body_result = find_level_single(body_vertices, body_faces, y)
        garment_result = find_level_single(garment_vertices, garment_faces, y)

        if body_result is None or garment_result is None:
            continue

        body_cm = body_result["circumference"] * 100.0
        garment_cm = garment_result["circumference"] * 100.0

        results[level_name] = {
            "garment_cm": garment_cm,
            "body_cm": body_cm,
            "ease_cm": garment_cm - body_cm,
        }

    return results
