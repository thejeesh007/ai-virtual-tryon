import os
import sys
import numpy as np

from mesh_utils import (
    get_torso_faces,
    find_clean_level,
    find_waist_level,
    ellipse_circumference,
)


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
# LOAD SCALED MESHES
#
# Both the front-only and side-only SMPL reconstructions are
# used together: a single monocular photo cannot constrain
# body depth, so the front mesh's own Z-axis (depth) and the
# side mesh's own Z-axis (its unconstrained axis, i.e. body
# width) are both unreliable. Instead we take WIDTH from the
# front mesh's well-observed X-axis and DEPTH from the side
# mesh's well-observed X-axis (in a side photo, "depth" is
# what the camera actually sees as image-horizontal), and
# combine the two into a circumference via an ellipse model.
# =========================================================

front_vertices = np.load(
    os.path.join(OUTPUT_DIR, "person_001_front_scaled.npy")
).astype(np.float64)

side_vertices = np.load(
    os.path.join(OUTPUT_DIR, "person_001_side_scaled.npy")
).astype(np.float64)

faces = np.asarray(model.smpl.faces)

lbs_weights = (
    model.smpl.lbs_weights
    .detach()
    .cpu()
    .numpy()
)

torso_faces = get_torso_faces(faces, lbs_weights)

print("\nMesh:")
print("Front vertices:", front_vertices.shape)
print("Side vertices :", side_vertices.shape)
print("Faces (full)               :", faces.shape)
print("Faces (torso only, no arms):", torso_faces.shape)


# =========================================================
# LOAD AUTOMATIC ANATOMICAL LEVELS
# =========================================================

levels_path = os.path.join(
    OUTPUT_DIR,
    "person_001_anatomical_levels.npy"
)

levels = np.load(
    levels_path,
    allow_pickle=True
).item()

print("\nAnatomical levels loaded:")

for name, y in levels.items():
    print(f"{name:10s}: {y:.2f} cm")


# =========================================================
# MEASURE ONE ANATOMICAL LEVEL
#
# 1. Find the Y level closest to the anatomical estimate
#    where BOTH the front and side torso-only meshes form a
#    single closed ring (right at a joint line -- e.g. the
#    shoulder ball joint -- the ring can be genuinely broken
#    by the limb attachment, a few cm off it is not).
# 2. Width comes from the front ring, depth from the side
#    ring's own "width" (its well-observed axis).
# 3. Circumference is modeled as an ellipse from the two.
# =========================================================

def measure_level(y_estimate, is_waist=False):

    if is_waist:
        y, front_result, side_result = find_waist_level(
            front_vertices,
            side_vertices,
            torso_faces,
            levels["chest"],
            levels["hip"],
        )
    else:
        y, front_result, side_result = find_clean_level(
            front_vertices,
            side_vertices,
            torso_faces,
            y_estimate,
        )

    if front_result is None or side_result is None:
        return None

    width = front_result["width"]
    depth = side_result["width"]

    circumference = ellipse_circumference(width, depth)

    return {
        "y": y,
        "y_offset": y - y_estimate,
        "circumference": circumference,
        "width": width,
        "depth": depth,
        "front_contours": front_result["num_contours"],
        "side_contours": side_result["num_contours"],
    }


# =========================================================
# MEASURE ALL FOUR
# =========================================================

print("\n==============================================")
print("AUTOMATIC SMPL BODY MEASUREMENTS")
print("==============================================")

results = {}

for name in [
    "shoulder",
    "chest",
    "waist",
    "hip"
]:

    y = levels[name]

    result = measure_level(y, is_waist=(name == "waist"))

    if result is None:

        print(
            f"\n{name.upper()}: "
            "Could not find a clean torso ring nearby."
        )

        continue

    results[name] = result

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Y level       : "
        f"{result['y']:.2f} cm "
        f"(offset {result['y_offset']:+.2f} cm)"
    )

    print(
        f"Circumference : "
        f"{result['circumference']:.2f} cm"
    )

    print(
        f"Width (front) : "
        f"{result['width']:.2f} cm"
    )

    print(
        f"Depth (side)  : "
        f"{result['depth']:.2f} cm"
    )

    print(
        f"Contours      : "
        f"front={result['front_contours']} "
        f"side={result['side_contours']}"
    )


# =========================================================
# GROUND TRUTH — VALIDATION ONLY
# =========================================================

ground_truth = {
    "shoulder": 101.0,
    "chest": 98.0,
    "waist": 84.0,
    "hip": 105.0
}


# =========================================================
# ERROR ANALYSIS
# =========================================================

print("\n==============================================")
print("VALIDATION — PERSON 001")
print("==============================================")

print(
    f"{'Measurement':<12}"
    f"{'Predicted':>14}"
    f"{'Actual':>12}"
    f"{'Error':>12}"
    f"{'Error %':>12}"
)

for name in [
    "shoulder",
    "chest",
    "waist",
    "hip"
]:

    if name not in results:
        continue

    predicted = results[name]["circumference"]

    actual = ground_truth[name]

    error = predicted - actual

    error_percent = (
        abs(error)
        / actual
        * 100
    )

    print(
        f"{name:<12}"
        f"{predicted:>14.2f}"
        f"{actual:>12.2f}"
        f"{error:>12.2f}"
        f"{error_percent:>11.2f}%"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "person_001_measurements.npy"
)

np.save(
    results_path,
    results,
    allow_pickle=True
)

print("\nSaved:")
print(results_path)

print("\n==============================================")
print("MEASUREMENT COMPLETED")
print("==============================================")