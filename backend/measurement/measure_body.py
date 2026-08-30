import os
import numpy as np


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "measurement",
    "hmr2_output"
)


# =========================================================
# Files
# =========================================================

FRONT_MESH = os.path.join(
    OUTPUT_DIR,
    "person_001_front_scaled.npy"
)

SIDE_MESH = os.path.join(
    OUTPUT_DIR,
    "person_001_side_scaled.npy"
)

FRONT_KEYPOINTS = os.path.join(
    OUTPUT_DIR,
    "person_001_front_keypoints_3d.npy"
)


# =========================================================
# Load data
# =========================================================

# =========================================================
# Load data
# =========================================================

front = np.load(FRONT_MESH)
side = np.load(SIDE_MESH)
keypoints = np.load(FRONT_KEYPOINTS)

print("Front vertices:", front.shape)
print("Side vertices :", side.shape)
print("Keypoints     :", keypoints.shape)


# =========================================================
# Scale keypoints to the same coordinate system as
# the scaled front mesh
# =========================================================

FRONT_HEIGHT = (
    front[:, 1].max()
    - front[:, 1].min()
)

# Load original, unscaled front mesh
original_front = np.load(
    os.path.join(
        OUTPUT_DIR,
        "person_001_front_vertices.npy"
    )
)

original_height = (
    original_front[:, 1].max()
    - original_front[:, 1].min()
)

front_scale = FRONT_HEIGHT / original_height

keypoints = keypoints * front_scale

print("\nFront mesh height:", FRONT_HEIGHT, "cm")
print("Original mesh height:", original_height)
print("Front scale:", front_scale)

# =========================================================
# Important HMR2 joints
# =========================================================

RIGHT_SHOULDER = 2
LEFT_SHOULDER = 5
MID_HIP = 8
RIGHT_HIP = 9
LEFT_HIP = 12


# =========================================================
# Determine anatomical Y levels
# =========================================================

shoulder_y = (
    keypoints[RIGHT_SHOULDER, 1]
    + keypoints[LEFT_SHOULDER, 1]
) / 2.0

hip_y = keypoints[MID_HIP, 1]


print("\nAnatomical reference levels")
print("--------------------------------")
print("Shoulder Y:", shoulder_y)
print("MidHip Y  :", hip_y)


# =========================================================
# Coordinate note
#
# HMR2 Y increases upward.
#
# Therefore:
#
# shoulder_y < hip_y
#
# because the person is upside-down in the
# coordinate values relative to normal image intuition.
# =========================================================


# =========================================================
# Convert Y to normalized torso position
#
# 0 = shoulder
# 1 = hip
# =========================================================

def torso_position(y):

    return (
        (y - shoulder_y)
        / (hip_y - shoulder_y)
    )


# =========================================================
# Get width at a Y level
# =========================================================

def get_width(vertices, y, tolerance=1.0):

    distances = np.abs(vertices[:, 1] - y)

    selected = vertices[distances <= tolerance]

    if len(selected) < 10:
        return None

    width = (
        selected[:, 0].max()
        - selected[:, 0].min()
    )

    return width


# =========================================================
# Get depth at a Y level
# =========================================================

def get_depth(vertices, y, tolerance=1.0):

    distances = np.abs(vertices[:, 1] - y)

    selected = vertices[distances <= tolerance]

    if len(selected) < 10:
        return None

    depth = (
        selected[:, 2].max()
        - selected[:, 2].min()
    )

    return depth


# =========================================================
# Ellipse circumference
# =========================================================

def ellipse_circumference(width, depth):

    a = width / 2.0
    b = depth / 2.0

    # Ramanujan approximation
    circumference = np.pi * (
        3 * (a + b)
        - np.sqrt(
            (3 * a + b)
            * (a + 3 * b)
        )
    )

    return circumference


# =========================================================
# Scan torso
# =========================================================

print("\nScanning torso...")
print("--------------------------------")

# We scan from shoulder to hip.
# Use 100 levels.

levels = np.linspace(
    shoulder_y,
    hip_y,
    100
)

results = []

for y in levels:

    width = get_width(
        front,
        y,
        tolerance=1.0
    )

    depth = get_depth(
        side,
        y,
        tolerance=1.0
    )

    if width is None or depth is None:
        continue

    circumference = ellipse_circumference(
        width,
        depth
    )

    position = torso_position(y)

    results.append(
        {
            "y": y,
            "position": position,
            "width": width,
            "depth": depth,
            "circumference": circumference
        }
    )


# =========================================================
# Print scan
# =========================================================

print(
    f"{'Position':>10}"
    f"{'Y':>12}"
    f"{'Width':>12}"
    f"{'Depth':>12}"
    f"{'Circumference':>18}"
)

for r in results:

    print(
        f"{r['position']:10.3f}"
        f"{r['y']:12.2f}"
        f"{r['width']:12.2f}"
        f"{r['depth']:12.2f}"
        f"{r['circumference']:18.2f}"
    )


# =========================================================
# Convert to arrays
# =========================================================

positions = np.array(
    [r["position"] for r in results]
)

circumferences = np.array(
    [r["circumference"] for r in results]
)


# =========================================================
# Find candidate regions
# =========================================================

# Upper torso:
# approximately shoulder -> middle torso

upper_mask = (
    (positions >= 0.10)
    &
    (positions <= 0.45)
)

# Middle torso:
# candidate waist region

middle_mask = (
    (positions >= 0.35)
    &
    (positions <= 0.70)
)

# Lower torso:
# candidate hip region

lower_mask = (
    (positions >= 0.60)
    &
    (positions <= 1.00)
)


def find_max(mask):

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    idx = indices[
        np.argmax(circumferences[indices])
    ]

    return results[idx]


def find_min(mask):

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    idx = indices[
        np.argmin(circumferences[indices])
    ]

    return results[idx]


chest_candidate = find_max(upper_mask)

waist_candidate = find_min(middle_mask)

hip_candidate = find_max(lower_mask)


# =========================================================
# Results
# =========================================================

print("\n==========================================")
print("CANDIDATE BODY MEASUREMENTS")
print("==========================================")


def print_candidate(name, result):

    if result is None:

        print(name, ": not found")
        return

    print(f"\n{name}")

    print("Y level:",
          round(result["y"], 2), "cm")

    print("Width:",
          round(result["width"], 2), "cm")

    print("Depth:",
          round(result["depth"], 2), "cm")

    print("Circumference:",
          round(result["circumference"], 2), "cm")

    print("Torso position:",
          round(result["position"], 3))


print_candidate(
    "Chest candidate",
    chest_candidate
)

print_candidate(
    "Waist candidate",
    waist_candidate
)

print_candidate(
    "Hip candidate",
    hip_candidate
)