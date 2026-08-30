import os
import sys
import numpy as np


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
# LOAD MESH
# =========================================================

mesh_path = os.path.join(
    OUTPUT_DIR,
    "person_001_front_scaled.npy"
)

vertices = np.load(mesh_path).astype(np.float64)

faces = np.asarray(model.smpl.faces)

print("\nMesh:")
print("Vertices:", vertices.shape)
print("Faces:", faces.shape)


# =========================================================
# SMPL JOINTS
# =========================================================

J_regressor = (
    model.smpl.J_regressor
    .detach()
    .cpu()
    .numpy()
)

joints = J_regressor @ vertices


# SMPL joint indices from your earlier inspection
NECK = 12
RIGHT_SHOULDER = 17
LEFT_SHOULDER = 16
MID_HIP = 0
RIGHT_HIP = 2
LEFT_HIP = 1


neck_y = joints[NECK][1]

shoulder_y = (
    joints[RIGHT_SHOULDER][1]
    + joints[LEFT_SHOULDER][1]
) / 2.0

mid_hip_y = joints[MID_HIP][1]

hip_joint_y = (
    joints[RIGHT_HIP][1]
    + joints[LEFT_HIP][1]
) / 2.0


# =========================================================
# CROSS SECTION
# =========================================================

def triangle_plane_intersection(
    v0,
    v1,
    v2,
    plane_y
):

    points = []

    edges = [
        (v0, v1),
        (v1, v2),
        (v2, v0)
    ]

    for a, b in edges:

        da = a[1] - plane_y
        db = b[1] - plane_y

        if abs(da) < 1e-8:
            points.append(a)

        if da * db < 0:

            t = (
                plane_y - a[1]
            ) / (
                b[1] - a[1]
            )

            p = a + t * (b - a)

            points.append(p)

    unique = []

    for p in points:

        if not any(
            np.linalg.norm(p - q) < 1e-7
            for q in unique
        ):
            unique.append(p)

    return unique


def extract_segments(
    plane_y
):

    segments = []

    for face in faces:

        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        points = triangle_plane_intersection(
            v0,
            v1,
            v2,
            plane_y
        )

        if len(points) == 2:

            segments.append(
                (points[0], points[1])
            )

    return segments


# =========================================================
# CONTOUR CONSTRUCTION
# =========================================================

def build_contours(
    segments,
    tolerance=1e-4
):

    unused = segments.copy()

    contours = []

    while unused:

        a, b = unused.pop(0)

        contour = [a, b]

        while True:

            end = contour[-1]

            found = False

            for i, (p, q) in enumerate(unused):

                if np.linalg.norm(
                    end - p
                ) < tolerance:

                    contour.append(q)
                    unused.pop(i)

                    found = True
                    break

                if np.linalg.norm(
                    end - q
                ) < tolerance:

                    contour.append(p)
                    unused.pop(i)

                    found = True
                    break

            if not found:
                break

        contours.append(
            np.array(contour)
        )

    return contours


# =========================================================
# CONTOUR MEASUREMENT
# =========================================================

def contour_info(contour):

    if len(contour) < 5:
        return None

    perimeter = (
        np.linalg.norm(
            np.diff(contour, axis=0),
            axis=1
        ).sum()
    )

    perimeter += np.linalg.norm(
        contour[0] - contour[-1]
    )

    x_min = contour[:, 0].min()
    x_max = contour[:, 0].max()

    z_min = contour[:, 2].min()
    z_max = contour[:, 2].max()

    width = x_max - x_min
    depth = z_max - z_min

    return {
        "perimeter": perimeter,
        "width": width,
        "depth": depth,
        "center_x": (x_min + x_max) / 2,
        "center_z": (z_min + z_max) / 2,
        "points": len(contour)
    }


# =========================================================
# MEASURE PLANE
# =========================================================

def measure_plane(y):

    segments = extract_segments(y)

    contours = build_contours(segments)

    infos = []

    for contour in contours:

        info = contour_info(contour)

        if info is not None:
            infos.append(info)

    if not infos:
        return None

    # Largest contour is normally the torso
    infos.sort(
        key=lambda x: x["perimeter"],
        reverse=True
    )

    return infos[0]


# =========================================================
# SCAN A REGION
# =========================================================

def scan_region(
    name,
    y_start,
    y_end,
    num_samples=40
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    ys = np.linspace(
        y_start,
        y_end,
        num_samples
    )

    results = []

    for y in ys:

        result = measure_plane(y)

        if result is None:
            continue

        results.append(
            {
                "y": y,
                **result
            }
        )

        print(
            f"Y={y:8.2f}  "
            f"C={result['perimeter']:8.2f}  "
            f"W={result['width']:7.2f}  "
            f"D={result['depth']:7.2f}"
        )

    return results


# =========================================================
# BODY REGIONS
# =========================================================

print("\n========================================")
print("SMPL ANATOMICAL REGION SCAN")
print("========================================")

print(
    f"Neck       : {neck_y:.2f}"
)

print(
    f"Shoulders  : {shoulder_y:.2f}"
)

print(
    f"MidHip     : {mid_hip_y:.2f}"
)

print(
    f"Hip joints : {hip_joint_y:.2f}"
)


# ---------------------------------------------------------
# 1. SHOULDER / UPPER TORSO
# ---------------------------------------------------------
# =========================================================
# CORRECTED BODY REGIONS
# =========================================================

# Shoulder region:
# slightly below the shoulder joint
shoulder_region = scan_region(
    "SHOULDER REGION",
    shoulder_y - 2,
    shoulder_y + 6
)


# Chest region:
# below the shoulders and above the waist
chest_region = scan_region(
    "CHEST REGION",
    shoulder_y + 5,
    shoulder_y + 25
)


# Waist region:
# lower torso, before the pelvis
waist_region = scan_region(
    "WAIST REGION",
    shoulder_y + 20,
    mid_hip_y - 2
)


# Hip region:
# around the pelvis
hip_region = scan_region(
    "HIP REGION",
    mid_hip_y - 2,
    mid_hip_y + 12
)

# =========================================================
# SAVE ALL SCANS
# =========================================================

scan_data = {
    "shoulder": shoulder_region,
    "chest": chest_region,
    "waist": waist_region,
    "hip": hip_region
}

save_path = os.path.join(
    OUTPUT_DIR,
    "person_001_anatomical_scan.npy"
)

np.save(
    save_path,
    scan_data,
    allow_pickle=True
)

print("\n========================================")
print("SCAN COMPLETED")
print("========================================")

print("\nSaved:")
print(save_path)