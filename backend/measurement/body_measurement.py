"""
Body measurement engine (v2).

Why this exists
----------------
The original pipeline (measure_anatomical.py / anatomical_scan.py) sliced the
*full* SMPL mesh with a horizontal plane and took the largest closed contour
as "the torso". That works fine for chest/waist/hip, but badly overestimates
the shoulder measurement (118cm predicted vs 101cm actual).

The reason: SMPL is a single watertight mesh with no seam between the torso
and the arms. A horizontal plane at shoulder height cuts through the torso
ring AND both upper-arm tubes as one connected surface, so "largest
contour" ends up tracing out around the arms too.

The fix: use the SMPL linear-blend-skinning weights (which every SMPL model
already carries) to find, for each vertex, which skeleton joint it is most
strongly attached to. Vertices dominated by an arm joint (shoulder / elbow /
wrist / hand) or a lower-leg joint (knee / ankle / foot) are removed from
the mesh before slicing. This physically detaches the arms/shins from the
torso, so a shoulder-height slice only ever touches the torso surface.

Removing those faces does leave small open gaps in the mesh where the limb
used to attach (e.g. the armpit). A horizontal slice through that region
therefore produces an *open* arc instead of a closed ring. We close that by
bridging together the loose ends of these arcs with straight chords -- which
is exactly what a tailor's tape does at the armpit: it doesn't wrap around
the arm, it goes straight across.
"""

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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "measurement",
    "hmr2_output"
)


# =========================================================
# STANDARD SMPL 24-JOINT ORDER
# =========================================================
# Confirmed against this project's model.smpl.parents:
# 0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee,
# 5 right_knee, 6 spine2, 7 left_ankle, 8 right_ankle,
# 9 spine3, 10 left_foot, 11 right_foot, 12 neck,
# 13 left_collar, 14 right_collar, 15 head, 16 left_shoulder,
# 17 right_shoulder, 18 left_elbow, 19 right_elbow,
# 20 left_wrist, 21 right_wrist, 22 left_hand, 23 right_hand

PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, \
    L_ANKLE, R_ANKLE, SPINE3, L_FOOT, R_FOOT, NECK, \
    L_COLLAR, R_COLLAR, HEAD, L_SHOULDER, R_SHOULDER, \
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HAND, R_HAND = range(24)

ARM_JOINTS = {
    L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_HAND, R_HAND
}

LOWER_LEG_JOINTS = {
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
    L_FOOT, R_FOOT
}

# Faces dominated by these joints are removed before slicing.
EXCLUDE_JOINTS = ARM_JOINTS | LOWER_LEG_JOINTS


# =========================================================
# MODEL LOADING / MESH PREP
# =========================================================

def load_model():
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT

    print("Loading HMR2...")
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    print("HMR2 loaded.")

    return model


def get_faces(model):
    return np.asarray(model.smpl.faces)


def get_joints(model, vertices):
    J_regressor = model.smpl.J_regressor.detach().cpu().numpy()
    return J_regressor @ vertices


def get_torso_faces(model):
    """
    Faces with all 3 vertices dominated by a non-arm, non-lower-leg
    joint. This keeps the pelvis/spine/neck/collar region (and both
    hip joints, needed for the hip measurement) and drops everything
    from the shoulder joint outward and the knee joint downward.
    """

    weights = model.smpl.lbs_weights.detach().cpu().numpy()
    dominant = weights.argmax(axis=1)

    faces = get_faces(model)

    excluded_vertex = np.isin(dominant, list(EXCLUDE_JOINTS))
    face_touches_excluded = excluded_vertex[faces].any(axis=1)

    return faces[~face_touches_excluded]


# =========================================================
# VECTORIZED PLANE / MESH CROSS-SECTION
# =========================================================

def cross_section_segments(vertices, faces, plane_y):
    """
    Intersect every face with the horizontal plane Y = plane_y.
    Vectorized over all faces; only the handful of faces that
    actually straddle the plane produce a segment.
    """

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    pts_per_edge = []
    valid_per_edge = []

    for a, b in ((v0, v1), (v1, v2), (v2, v0)):

        da = a[:, 1] - plane_y
        db = b[:, 1] - plane_y

        valid = da * db < 0

        denom = b[:, 1] - a[:, 1]
        denom = np.where(denom == 0, 1e-12, denom)

        t = (plane_y - a[:, 1]) / denom
        p = a + t[:, None] * (b - a)

        pts_per_edge.append(p)
        valid_per_edge.append(valid)

    pts = np.stack(pts_per_edge, axis=1)      # (F, 3, 3)
    valid = np.stack(valid_per_edge, axis=1)  # (F, 3)

    counts = valid.sum(axis=1)
    idxs = np.where(counts == 2)[0]

    segments = []

    for i in idxs:
        p = pts[i][valid[i]]
        segments.append((p[0], p[1]))

    return segments


# =========================================================
# CONTOUR STITCHING (bidirectional) + GAP BRIDGING
# =========================================================

def _dist(p, q):
    return float(np.linalg.norm(p - q))


def stitch_segments(segments, tol=1e-3):
    """
    Chain raw segments into polyline fragments. Extends from BOTH
    ends (the original scripts only extended forward, which can
    needlessly fragment a contour into pieces that never get
    reassembled).
    """

    unused = list(segments)
    fragments = []

    while unused:

        a, b = unused.pop(0)
        frag = [a, b]

        changed = True

        while changed:
            changed = False

            end = frag[-1]

            for i, (p, q) in enumerate(unused):

                if _dist(end, p) < tol:
                    frag.append(q)
                    unused.pop(i)
                    changed = True
                    break

                if _dist(end, q) < tol:
                    frag.append(p)
                    unused.pop(i)
                    changed = True
                    break

            if changed:
                continue

            start = frag[0]

            for i, (p, q) in enumerate(unused):

                if _dist(start, p) < tol:
                    frag.insert(0, q)
                    unused.pop(i)
                    changed = True
                    break

                if _dist(start, q) < tol:
                    frag.insert(0, p)
                    unused.pop(i)
                    changed = True
                    break

        fragments.append([np.asarray(pt) for pt in frag])

    return fragments


def close_and_bridge(fragments, close_tol=1e-2, bridge_tol=20.0):
    """
    Fragments that are already closed loops (start ~= end) pass
    through untouched. Open fragments (created where arm/leg faces
    were removed) are greedily bridged end-to-end via their nearest
    loose endpoints, approximating the straight-across path a tape
    measure takes at the armpit.
    """

    closed = []
    open_frags = []

    for f in fragments:
        if _dist(f[0], f[-1]) < close_tol:
            closed.append(np.array(f))
        else:
            open_frags.append(list(f))

    while len(open_frags) > 1:

        best = None

        for i in range(len(open_frags)):
            for j in range(i + 1, len(open_frags)):

                fi, fj = open_frags[i], open_frags[j]

                for ei, pi in ((0, fi[0]), (-1, fi[-1])):
                    for ej, pj in ((0, fj[0]), (-1, fj[-1])):

                        d = _dist(pi, pj)

                        if best is None or d < best[0]:
                            best = (d, i, j, ei, ej)

        d, i, j, ei, ej = best

        if d > bridge_tol:
            break

        fi, fj = open_frags[i], open_frags[j]

        if ei == -1 and ej == 0:
            merged = fi + fj
        elif ei == -1 and ej == -1:
            merged = fi + fj[::-1]
        elif ei == 0 and ej == 0:
            merged = fi[::-1] + fj
        else:
            merged = fj + fi

        open_frags = [
            open_frags[k]
            for k in range(len(open_frags))
            if k not in (i, j)
        ]
        open_frags.append(merged)

    result = list(closed)

    for f in open_frags:
        if _dist(f[0], f[-1]) < bridge_tol:
            result.append(np.array(f))

    return result


def build_contours(vertices, faces, plane_y, tol=1e-3, bridge_tol=20.0):
    segments = cross_section_segments(vertices, faces, plane_y)
    fragments = stitch_segments(segments, tol=tol)
    return close_and_bridge(fragments, bridge_tol=bridge_tol)


# =========================================================
# CONTOUR METRICS
# =========================================================

def contour_info(contour):

    if len(contour) < 5:
        return None

    perimeter = np.linalg.norm(
        np.diff(contour, axis=0), axis=1
    ).sum()

    perimeter += np.linalg.norm(contour[0] - contour[-1])

    x_min, x_max = contour[:, 0].min(), contour[:, 0].max()
    z_min, z_max = contour[:, 2].min(), contour[:, 2].max()

    return {
        "circumference": float(perimeter),
        "width": float(x_max - x_min),
        "depth": float(z_max - z_min),
        "center_x": float((x_min + x_max) / 2),
        "center_z": float((z_min + z_max) / 2),
        "points": len(contour)
    }


def measure_level(vertices, faces, plane_y, tol=1e-3, bridge_tol=20.0):
    contours = build_contours(
        vertices, faces, plane_y, tol=tol, bridge_tol=bridge_tol
    )

    best = None

    for contour in contours:
        info = contour_info(contour)

        if info is None:
            continue

        if best is None or info["circumference"] > best["circumference"]:
            best = info

    if best is not None:
        best["y"] = float(plane_y)
        best["num_contours"] = len(contours)

    return best


# =========================================================
# REGION SEARCH
# =========================================================

def scan_region(vertices, faces, y_start, y_end, num_samples=60,
                 tol=1e-3, bridge_tol=20.0):

    ys = np.linspace(y_start, y_end, num_samples)
    results = []

    for y in ys:
        r = measure_level(vertices, faces, y, tol=tol, bridge_tol=bridge_tol)
        if r is not None:
            results.append(r)

    return results


def pick_extremum(results, mode):
    if not results:
        return None

    key = (lambda r: r["circumference"])

    if mode == "max":
        return max(results, key=key)

    if mode == "min":
        return min(results, key=key)

    raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# FULL MEASUREMENT PIPELINE FOR ONE PERSON
# =========================================================

def measure_person(model, vertices):
    """
    vertices: (6890, 3) scaled SMPL vertices (already in cm), for a
    single person/pose.

    Returns dict with shoulder/chest/waist/hip results plus the
    anatomical joint levels used to define the search regions.
    """

    full_faces = get_faces(model)
    torso_faces = get_torso_faces(model)

    joints = get_joints(model, vertices)

    neck_y = joints[NECK][1]
    shoulder_y = (joints[L_SHOULDER][1] + joints[R_SHOULDER][1]) / 2.0
    mid_hip_y = joints[PELVIS][1]
    hip_joint_y = (joints[L_HIP][1] + joints[R_HIP][1]) / 2.0

    levels = {
        "neck_y": float(neck_y),
        "shoulder_joint_y": float(shoulder_y),
        "mid_hip_y": float(mid_hip_y),
        "hip_joint_y": float(hip_joint_y),
    }

    # Y increases going DOWN the body in this mesh (neck < shoulder
    # < hip numerically), so "below the shoulder" means +offset.

    # ---- SHOULDER: torso-only mesh, max circumference near the
    # shoulder joint. Excluding the arms is what turns the old
    # "trace around the whole arm" measurement into a straight
    # across-the-armpit measurement.
    shoulder_scan = scan_region(
        vertices, torso_faces,
        shoulder_y - 2, shoulder_y + 6,
        num_samples=40
    )
    shoulder = pick_extremum(shoulder_scan, "max")

    # ---- CHEST: fullest point of the upper torso, below the
    # shoulders and above the waist. Torso-only mesh so a lingering
    # arm sliver can't distort it.
    chest_scan = scan_region(
        vertices, torso_faces,
        shoulder_y + 6, shoulder_y + 26,
        num_samples=60
    )
    chest = pick_extremum(chest_scan, "max")

    # ---- WAIST: narrowest point of the torso between the chest
    # and the hips (standard anthropometric definition).
    waist_scan = scan_region(
        vertices, torso_faces,
        shoulder_y + 22, mid_hip_y - 2,
        num_samples=80
    )
    waist = pick_extremum(waist_scan, "min")

    # ---- HIP: fullest point around the pelvis.
    hip_scan = scan_region(
        vertices, torso_faces,
        mid_hip_y - 2, mid_hip_y + 14,
        num_samples=60
    )
    hip = pick_extremum(hip_scan, "max")

    return {
        "levels": levels,
        "shoulder": shoulder,
        "chest": chest,
        "waist": waist,
        "hip": hip,
    }


def load_scaled_mesh(person_id, view="front"):
    path = os.path.join(OUTPUT_DIR, f"{person_id}_{view}_scaled.npy")
    return np.load(path).astype(np.float64)
