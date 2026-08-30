import os
import sys
import numpy as np
import torch


# =========================================================
# PROJECT PATHS
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
# LOAD HMR2
# =========================================================

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT


print("Loading HMR2...")

download_models(CACHE_DIR_4DHUMANS)

model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

print("HMR2 loaded.")


# =========================================================
# LOAD SCALED FRONT MESH
# =========================================================

mesh_path = os.path.join(
    OUTPUT_DIR,
    "person_001_front_scaled.npy"
)

vertices = np.load(mesh_path).astype(np.float64)

print("\nMesh:")
print("Vertices:", vertices.shape)


# =========================================================
# GET SMPL TRIANGLE FACES
# =========================================================

faces = np.asarray(model.smpl.faces)

print("Faces:", faces.shape)


# =========================================================
# BASIC MESH INFORMATION
# =========================================================

min_y = vertices[:, 1].min()
max_y = vertices[:, 1].max()

print("\nY range:")
print("min:", min_y)
print("max:", max_y)
print("height:", max_y - min_y)


# =========================================================
# PLANE / TRIANGLE INTERSECTION
# =========================================================

def triangle_plane_intersection(v0, v1, v2, plane_y):
    """
    Find intersection points between one triangle
    and a horizontal plane Y = plane_y.

    Returns:
        list containing 0, 1 or 2 points.
    """

    points = []

    edges = [
        (v0, v1),
        (v1, v2),
        (v2, v0)
    ]

    for a, b in edges:

        ya = a[1]
        yb = b[1]

        da = ya - plane_y
        db = yb - plane_y

        # Vertex exactly on plane
        if abs(da) < 1e-8:
            points.append(a)

        # Edge crosses plane
        if da * db < 0:

            t = (plane_y - ya) / (yb - ya)

            point = a + t * (b - a)

            points.append(point)

    # Remove duplicate points
    unique = []

    for p in points:

        if not any(
            np.linalg.norm(p - q) < 1e-7
            for q in unique
        ):
            unique.append(p)

    return unique


# =========================================================
# EXTRACT CROSS-SECTION SEGMENTS
# =========================================================

def extract_segments(vertices, faces, plane_y):

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
                (
                    points[0],
                    points[1]
                )
            )

    return segments


# =========================================================
# CONNECT SEGMENTS INTO CONTOURS
# =========================================================

def build_contours(segments, tolerance=1e-5):

    if not segments:
        return []

    unused = segments.copy()

    contours = []

    while unused:

        p1, p2 = unused.pop(0)

        contour = [p1, p2]

        changed = True

        while changed:

            changed = False

            end = contour[-1]

            for i, (a, b) in enumerate(unused):

                if np.linalg.norm(end - a) < tolerance:

                    contour.append(b)
                    unused.pop(i)
                    changed = True
                    break

                if np.linalg.norm(end - b) < tolerance:

                    contour.append(a)
                    unused.pop(i)
                    changed = True
                    break

        contours.append(
            np.array(contour)
        )

    return contours


# =========================================================
# PERIMETER
# =========================================================

def contour_perimeter(contour):

    if len(contour) < 3:
        return 0.0

    distances = np.linalg.norm(
        np.diff(contour, axis=0),
        axis=1
    )

    # Close contour
    closing_distance = np.linalg.norm(
        contour[0] - contour[-1]
    )

    return (
        distances.sum()
        + closing_distance
    )


# =========================================================
# MEASURE ONE LEVEL
# =========================================================

def measure_level(vertices, faces, y):

    segments = extract_segments(
        vertices,
        faces,
        y
    )

    contours = build_contours(
        segments
    )

    if not contours:
        return None

    measurements = []

    for contour in contours:

        perimeter = contour_perimeter(
            contour
        )

        measurements.append(
            (perimeter, contour)
        )

    # Largest contour = main body contour
    measurements.sort(
        key=lambda x: x[0],
        reverse=True
    )

    perimeter, contour = measurements[0]

    return {
        "circumference": perimeter,
        "contour": contour,
        "num_contours": len(contours)
    }


# =========================================================
# TEST A RANGE OF Y LEVELS
# =========================================================

print("\n========================================")
print("SCANNING SMPL SURFACE")
print("========================================")

# Don't scan directly at the extreme top/bottom.
levels = np.linspace(
    min_y + 10,
    max_y - 10,
    150
)

scan_results = []

for y in levels:

    result = measure_level(
        vertices,
        faces,
        y
    )

    if result is not None:

        scan_results.append(
            {
                "y": y,
                "circumference":
                    result["circumference"],
                "contours":
                    result["num_contours"]
            }
        )


# =========================================================
# PRINT RESULTS
# =========================================================

print(
    f"\n{'Y':>12}"
    f"{'Circumference':>20}"
    f"{'Contours':>12}"
)

for result in scan_results:

    print(
        f"{result['y']:12.2f}"
        f"{result['circumference']:20.2f}"
        f"{result['contours']:12d}"
    )


# =========================================================
# SAVE SCAN
# =========================================================

scan_path = os.path.join(
    OUTPUT_DIR,
    "person_001_cross_section_scan.npy"
)

np.save(
    scan_path,
    np.array(
        [
            [
                r["y"],
                r["circumference"],
                r["contours"]
            ]
            for r in scan_results
        ]
    )
)

print("\nSaved scan:")
print(scan_path)

print("\n========================================")
print("CROSS-SECTION EXTRACTION COMPLETED")
print("========================================")