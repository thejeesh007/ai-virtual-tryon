import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

HMR2_ROOT = os.path.join(
    PROJECT_ROOT,
    "third_party",
    "4D-Humans"
)

sys.path.insert(0, HMR2_ROOT)

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT


# =========================================================
# Load HMR2 only to obtain SMPL faces
# =========================================================

print("Loading HMR2...")

download_models(CACHE_DIR_4DHUMANS)

model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

print("HMR2 loaded.")


# =========================================================
# Load scaled mesh
# =========================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "measurement",
    "hmr2_output"
)

mesh_path = os.path.join(
    OUTPUT_DIR,
    "person_001_front_scaled.npy"
)

vertices = np.load(mesh_path).astype(np.float64)

faces = np.asarray(model.smpl.faces)

print("\nVertices:", vertices.shape)
print("Faces:", faces.shape)


# =========================================================
# Triangle-plane intersection
# =========================================================

def triangle_plane_intersection(v0, v1, v2, plane_y):

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

            t = (plane_y - a[1]) / (b[1] - a[1])

            point = a + t * (b - a)

            points.append(point)

    unique = []

    for p in points:

        if not any(
            np.linalg.norm(p - q) < 1e-7
            for q in unique
        ):
            unique.append(p)

    return unique


# =========================================================
# Extract segments
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
                (points[0], points[1])
            )

    return segments


# =========================================================
# Build contours
# =========================================================

def build_contours(segments, tolerance=1e-4):

    unused = segments.copy()
    contours = []

    while unused:

        p1, p2 = unused.pop(0)

        contour = [p1, p2]

        while True:

            end = contour[-1]

            found = False

            for i, (a, b) in enumerate(unused):

                if np.linalg.norm(end - a) < tolerance:

                    contour.append(b)
                    unused.pop(i)
                    found = True
                    break

                if np.linalg.norm(end - b) < tolerance:

                    contour.append(a)
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
# Contour statistics
# =========================================================

def contour_stats(contour):

    if len(contour) < 3:
        return None

    perimeter = np.linalg.norm(
        np.diff(contour, axis=0),
        axis=1
    ).sum()

    perimeter += np.linalg.norm(
        contour[0] - contour[-1]
    )

    x_min = contour[:, 0].min()
    x_max = contour[:, 0].max()

    z_min = contour[:, 2].min()
    z_max = contour[:, 2].max()

    width = x_max - x_min
    depth = z_max - z_min

    center_x = (
        x_min + x_max
    ) / 2

    center_z = (
        z_min + z_max
    ) / 2

    return {
        "points": len(contour),
        "perimeter": perimeter,
        "width": width,
        "depth": depth,
        "center_x": center_x,
        "center_z": center_z
    }


# =========================================================
# Diagnostic levels
#
# These are the torso levels where our previous scan
# showed interesting behavior.
# =========================================================

TEST_LEVELS = [
    -57.77,
    -53.69,
    -48.59,
    -45.53,
    -40.43,
    -35.33,
    -30.23,
    -25.13,
    -24.11
]


# =========================================================
# Analyze
# =========================================================

for y in TEST_LEVELS:

    print("\n")
    print("=" * 75)
    print(f"Y LEVEL = {y:.2f} cm")
    print("=" * 75)

    segments = extract_segments(
        vertices,
        faces,
        y
    )

    contours = build_contours(
        segments
    )

    print(
        "Segments:",
        len(segments)
    )

    print(
        "Contours:",
        len(contours)
    )

    contour_data = []

    for i, contour in enumerate(contours):

        stats = contour_stats(contour)

        if stats is None:
            continue

        stats["id"] = i

        contour_data.append(stats)

    # Largest perimeter first
    contour_data.sort(
        key=lambda x: x["perimeter"],
        reverse=True
    )

    print(
        f"\n{'ID':>4}"
        f"{'Points':>8}"
        f"{'Perimeter':>14}"
        f"{'Width':>12}"
        f"{'Depth':>12}"
        f"{'Center X':>12}"
        f"{'Center Z':>12}"
    )

    for c in contour_data:

        print(
            f"{c['id']:4d}"
            f"{c['points']:8d}"
            f"{c['perimeter']:14.2f}"
            f"{c['width']:12.2f}"
            f"{c['depth']:12.2f}"
            f"{c['center_x']:12.2f}"
            f"{c['center_z']:12.2f}"
        )