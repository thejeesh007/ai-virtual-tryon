"""
Module 3/5 (simplified): garment "draping" and tightness heatmap, without
real cloth physics. This is the fast first version chosen to get a
complete end-to-end pipeline working; swap for TailorNet-based real
draping later if needed.

The trick: a garment's fit is defined by its own measurements (chest/
waist/hip/shoulder-breadth, as designed/cut) plus an ease value (extra
room the pattern builds in on top of those measurements -- a fitted cut
uses a small ease, a relaxed cut a larger one). Fit a *second* SMPL body
to (garment measurement + ease) using the exact same optimizer as
smpl_fit.py -- this produces a "garment-reference body": the body shape
the finished garment actually has room for. Since it's fit at the
wearer's own height with the same SMPL topology/pose, its vertices
correspond 1:1 to the wearer's avatar vertices by index.

Tightness per vertex = garment-reference radius - wearer radius, measured
from the shared vertical centerline (SMPL's pelvis sits at x=0,z=0 with no
global translation, so this holds for any betas):
    positive -> garment has room there (loose, blue)
    near zero -> comfortable (green)
    negative -> wearer's body exceeds the garment surface there (tight, red)

Output is a .ply mesh with per-vertex tightness baked in as color, using
the same red/green/blue convention as the project report's Layer 1 spec.

Usage:
    python garment_drape.py --height 167 --chest 85 --waist 75 --hip 90 \
        --shoulder-breadth 38 --arm-length 52 \
        --garment-chest 90 --garment-waist 78 --garment-hip 92 \
        --garment-shoulder-breadth 40 --ease 2 \
        --out person_007_tightness.ply
"""

import argparse
import os

import numpy as np

from smpl_fit import (
    MODEL_DIR, OUTPUT_DIR, load_smpl_model, forward, scale_to_height,
    fit_betas, measure_smpl_mesh, compute_anatomical_levels,
)
from mesh_utils import get_torso_faces


def fit_garment_body(model, torso_faces, garment_measurements, ease, height_cm):
    """Reuses smpl_fit.fit_betas, but targeting the garment's own
    (measurement + ease) as the shape it was cut for, instead of a
    wearer's real measurements. arm-length is filled in from the wearer's
    own height proportionally since the optimizer needs all 5 targets --
    it's not used for tightness anyway (torso-only garment)."""

    garment_target = {
        "chest": garment_measurements["chest"] + ease,
        "waist": garment_measurements["waist"] + ease,
        "hip": garment_measurements["hip"] + ease,
        "shoulder-breadth": garment_measurements["shoulder-breadth"] + ease,
        "arm-length": 0.30 * height_cm,  # rough proportion, unused for torso tightness
    }

    betas, result = fit_betas(model, torso_faces, garment_target, height_cm)
    return betas, garment_target, result


def compute_tightness(wearer_vertices, garment_vertices, torso_vertex_ids):
    """Per-vertex tightness = garment radius - wearer radius, from the
    shared x=0,z=0 vertical centerline. Only computed for torso_vertex_ids
    (the garment-covered region); everything else is left at NaN."""

    tightness = np.full(len(wearer_vertices), np.nan)

    wearer_radius = np.sqrt(wearer_vertices[:, 0] ** 2 + wearer_vertices[:, 2] ** 2)
    garment_radius = np.sqrt(garment_vertices[:, 0] ** 2 + garment_vertices[:, 2] ** 2)

    tightness[torso_vertex_ids] = (
        garment_radius[torso_vertex_ids] - wearer_radius[torso_vertex_ids]
    )

    return tightness


def tightness_to_color(tightness, tight_thresh=-1.0, loose_thresh=1.0):
    """Red (<=tight_thresh, pinching) -> green (0, comfortable) ->
    blue (>=loose_thresh, loose). NaN (no garment coverage) -> gray."""

    colors = np.full((len(tightness), 3), 180, dtype=np.uint8)  # gray default

    valid = ~np.isnan(tightness)
    t = np.clip(tightness[valid], tight_thresh, loose_thresh)

    # Normalize to [-1, 1] then split into red->green and green->blue halves.
    norm = t / max(abs(tight_thresh), abs(loose_thresh))

    r = np.where(norm < 0, 255, (255 * (1 - norm)).astype(np.uint8))
    g = np.where(norm < 0, (255 * (1 + norm)).astype(np.uint8), (255 * (1 - norm)).astype(np.uint8))
    b = np.where(norm < 0, 0, (255 * norm).astype(np.uint8))

    colors[valid] = np.stack([r, g, b], axis=1).astype(np.uint8)
    return colors


def save_ply(path, vertices, faces, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def region_summary(tightness, wearer_vertices, wearer_joints, band_halfwidth=2.5):
    """chest/waist/hip band summary: mean tightness for vertices within
    band_halfwidth (cm) of the actual anatomical Y-level used to fit that
    measurement -- not a crude fraction-of-height guess, so it lines up
    with what was actually optimized."""

    levels = compute_anatomical_levels(wearer_joints)
    y = wearer_vertices[:, 1]

    bands = {
        "chest": levels["chest"],
        "waist": levels["waist"],
        "hip": levels["hip"],
    }

    summary = {}
    for name, y_level in bands.items():
        mask = (np.abs(y - y_level) <= band_halfwidth) & ~np.isnan(tightness)
        summary[name] = float(np.nanmean(tightness[mask])) if mask.any() else None

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--chest", type=float, required=True)
    p.add_argument("--waist", type=float, required=True)
    p.add_argument("--hip", type=float, required=True)
    p.add_argument("--shoulder-breadth", type=float, required=True)
    p.add_argument("--arm-length", type=float, required=True)

    p.add_argument("--garment-chest", type=float, required=True,
                    help="Garment's own chest measurement, as cut (cm)")
    p.add_argument("--garment-waist", type=float, required=True)
    p.add_argument("--garment-hip", type=float, required=True)
    p.add_argument("--garment-shoulder-breadth", type=float, required=True)
    p.add_argument("--ease", type=float, required=True,
                    help="Design ease (cm) the pattern adds on top of the garment's own "
                         "measurements -- a fitted cut uses a small value, a relaxed cut a larger one")

    p.add_argument("--out", type=str, default="tightness.ply")
    return p.parse_args()


def main():
    args = parse_args()

    wearer_target = {
        "chest": args.chest, "waist": args.waist, "hip": args.hip,
        "shoulder-breadth": args.shoulder_breadth, "arm-length": args.arm_length,
    }

    print("Loading SMPL model...")
    model = load_smpl_model()
    faces = model.faces
    lbs_weights = model.lbs_weights.detach().numpy()
    torso_faces = get_torso_faces(faces, lbs_weights)
    torso_vertex_ids = np.unique(torso_faces)

    print(f"\nFitting wearer body to: {wearer_target}")
    wearer_betas, _ = fit_betas(model, torso_faces, wearer_target, args.height)
    wearer_vertices, wearer_joints = forward(model, wearer_betas)
    wearer_vertices, wearer_joints = scale_to_height(wearer_vertices, wearer_joints, args.height)

    garment_measurements = {
        "chest": args.garment_chest, "waist": args.garment_waist,
        "hip": args.garment_hip, "shoulder-breadth": args.garment_shoulder_breadth,
    }
    print(f"\nFitting garment reference body to: {garment_measurements} + {args.ease}cm ease")
    garment_betas, garment_target, _ = fit_garment_body(
        model, torso_faces, garment_measurements, args.ease, args.height
    )
    garment_vertices, garment_joints = forward(model, garment_betas)
    garment_vertices, garment_joints = scale_to_height(garment_vertices, garment_joints, args.height)

    tightness = compute_tightness(wearer_vertices, garment_vertices, torso_vertex_ids)
    colors = tightness_to_color(tightness)

    summary = region_summary(tightness, wearer_vertices, wearer_joints)
    print(f"\n=== Tightness summary: wearer in this garment ===")
    for region, val in summary.items():
        if val is None:
            print(f"  {region:22s}: no data")
            continue
        verdict = "TIGHT" if val < -1 else ("LOOSE" if val > 1 else "comfortable")
        print(f"  {region:22s}: {val:+.2f} cm ({verdict})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.out)
    save_ply(out_path, wearer_vertices, torso_faces, colors)
    print(f"\nSaved tightness-colored mesh to {out_path}")


if __name__ == "__main__":
    main()
