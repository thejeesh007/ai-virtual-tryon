import numpy as np


# =========================================================
# SMPL JOINT INDICES (standard 24-joint SMPL skeleton)
# =========================================================

PELVIS = 0
LEFT_HIP = 1
RIGHT_HIP = 2
SPINE1 = 3
NECK = 12
LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21
LEFT_HAND = 22
RIGHT_HAND = 23

# Vertices dominated by these joints are the arms (deltoid
# through fingertips). They are excluded from torso cross
# sections because in a natural standing pose the arms hang
# against the sides of the body and can touch/overlap the
# torso surface, which corrupts plane/mesh slicing at
# shoulder, waist and hip height.
ARM_JOINTS = {
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HAND, RIGHT_HAND,
}


def get_torso_faces(faces, lbs_weights, arm_weight_threshold=0.5):
    """
    Filter out any face that touches an arm vertex, using
    per-vertex SMPL linear-blend-skinning weights.

    A vertex is treated as "arm" only when the arm joints
    (shoulder/elbow/wrist/hand) together hold the MAJORITY
    of its skinning weight. A plain argmax-per-vertex cut is
    too aggressive right at the shoulder joint: SMPL blends
    weights smoothly near the ball joint, so torso-side
    vertices next to the underarm can have the shoulder
    joint as their single largest weight while still being
    mostly torso/collar influenced overall. Thresholding on
    the summed arm weight keeps those torso vertices so the
    cross-section ring at shoulder height stays closed,
    while still dropping true arm/hand vertices (which carry
    a large majority of their weight on arm joints).

    Args:
        faces: (F, 3) int array of triangle vertex indices.
        lbs_weights: (V, 24) array of per-vertex joint weights.
        arm_weight_threshold: minimum summed arm-joint weight
            for a vertex to be excluded as "arm".

    Returns:
        (F', 3) int array containing only torso/leg/head faces.
    """

    arm_weight = lbs_weights[:, list(ARM_JOINTS)].sum(axis=1)

    is_arm_vertex = arm_weight > arm_weight_threshold

    face_touches_arm = is_arm_vertex[faces].any(axis=1)

    return faces[~face_touches_arm]


# =========================================================
# TRIANGLE / PLANE INTERSECTION
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

        # Vertex exactly on plane
        if abs(da) < 1e-8:
            points.append(a)

        # Edge crosses plane
        if da * db < 0:

            t = (plane_y - a[1]) / (b[1] - a[1])

            p = a + t * (b - a)

            points.append(p)

    # Remove duplicates
    unique = []

    for p in points:

        if not any(
            np.linalg.norm(p - q) < 1e-7
            for q in unique
        ):
            unique.append(p)

    return unique


def extract_segments(vertices, faces, plane_y):

    segments = []

    for face in faces:

        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        points = triangle_plane_intersection(
            v0, v1, v2, plane_y
        )

        if len(points) == 2:
            segments.append((points[0], points[1]))

    return segments


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

        contours.append(np.array(contour))

    return contours


def get_contour_info(contour):

    if len(contour) < 3:
        return None

    perimeter = np.linalg.norm(
        np.diff(contour, axis=0), axis=1
    ).sum()

    perimeter += np.linalg.norm(contour[0] - contour[-1])

    x_min = contour[:, 0].min()
    x_max = contour[:, 0].max()

    z_min = contour[:, 2].min()
    z_max = contour[:, 2].max()

    return {
        "perimeter": perimeter,
        "width": x_max - x_min,
        "depth": z_max - z_min,
        "center_x": (x_min + x_max) / 2,
        "center_z": (z_min + z_max) / 2,
        "points": len(contour),
    }


def select_largest_contour(contours):

    candidates = []

    for contour in contours:

        info = get_contour_info(contour)

        if info is None:
            continue

        candidates.append((info["perimeter"], contour, info))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0]


def ellipse_circumference(width, depth):
    """
    Ramanujan's approximation for the circumference of an
    ellipse with the given full width and depth.
    """

    a = width / 2.0
    b = depth / 2.0

    return np.pi * (
        3 * (a + b)
        - np.sqrt((3 * a + b) * (a + 3 * b))
    )


def find_clean_level(
    front_vertices,
    side_vertices,
    faces,
    y_start,
    max_offset=10.0,
    step=0.25,
):
    """
    Search outward (both directions) from y_start for the Y
    level nearest to it where slicing BOTH the front and the
    side mesh produces exactly one closed torso contour.

    Right at an SMPL joint line (e.g. the shoulder ball
    joint) the surface ring can be genuinely broken by the
    limb attachment, so the plane/mesh intersection returns
    several disconnected fragments instead of one ring. This
    finds the closest level, above or below, where the ring
    is intact in both independent single-view reconstructions
    -- a property of the mesh topology itself, not a fitted
    constant, so it generalizes across subjects/poses.

    Returns (y, front_result, side_result), or (y_start, None, None)
    if no clean level is found within max_offset.
    """

    offsets = [0.0]

    n_steps = int(max_offset / step)

    for i in range(1, n_steps + 1):
        offsets.append(i * step)
        offsets.append(-i * step)

    for offset in offsets:

        y = y_start + offset

        front_result = measure_level(front_vertices, faces, y)
        side_result = measure_level(side_vertices, faces, y)

        if (
            front_result is not None
            and side_result is not None
            and front_result["num_contours"] == 1
            and side_result["num_contours"] == 1
        ):
            return y, front_result, side_result

    return y_start, None, None


def find_waist_level(
    front_vertices,
    side_vertices,
    faces,
    y_start,
    y_end,
    num_samples=40,
):
    """
    The waist is anatomically the narrowest point of the
    torso, not a fixed fraction of the shoulder-to-hip
    distance. Scan between y_start (chest) and y_end (hip)
    and return the level with the smallest front+side
    ellipse circumference, restricted to levels where both
    meshes form a single clean torso ring.

    Returns (y, front_result, side_result), or
    (None, None, None) if no clean level was found.
    """

    best = (None, None, None)
    best_circumference = np.inf

    for y in np.linspace(y_start, y_end, num_samples):

        front_result = measure_level(front_vertices, faces, y)
        side_result = measure_level(side_vertices, faces, y)

        if (
            front_result is None
            or side_result is None
            or front_result["num_contours"] != 1
            or side_result["num_contours"] != 1
        ):
            continue

        circumference = ellipse_circumference(
            front_result["width"],
            side_result["width"],
        )

        if circumference < best_circumference:
            best_circumference = circumference
            best = (y, front_result, side_result)

    return best


def measure_level(vertices, faces, y):
    """
    Slice the (already arm-filtered) mesh at height y and
    return the circumference/width/depth of the outer torso
    contour.
    """

    segments = extract_segments(vertices, faces, y)

    contours = build_contours(segments)

    selected = select_largest_contour(contours)

    if selected is None:
        return None

    perimeter, contour, info = selected

    return {
        "y": y,
        "circumference": perimeter,
        "width": info["width"],
        "depth": info["depth"],
        "center_x": info["center_x"],
        "center_z": info["center_z"],
        "num_contours": len(contours),
        "points": info["points"],
    }
