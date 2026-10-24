"""
Measurement-driven fit heatmap: colors the AVATAR's own body mesh directly,
based on comparing a real garment's actual measurements against what a
well-fitting garment of its fabric type SHOULD measure for this specific
user's own predicted body measurements. No garment mesh, no draping physics
-- entirely independent of third_party/GarmentCode's simple_drape.py
pipeline, which drapes a simulated 3D mesh and reads tightness off it
geometrically. This instead does the comparison as plain arithmetic per
body part (chest/waist/hip/shoulder/sleeve/neck) and paints the result onto
the SAME avatar mesh already shown for the bare-body view -- sidesteps
every failure mode that came from the wrap/relax/smoothing machinery
(boxy skirts, torn shoulders, exploding cloth sims, etc.), at the cost of
not showing what the garment itself looks like on the body.

Region assignment: every avatar vertex is classified into one of
neck/shoulder/chest/waist/hip (torso, via nearest anatomical Y-level,
reusing smpl_fit.compute_anatomical_levels), thigh/knee/ankle_hem (legs,
same nearest-Y-level idea -- legs stay vertical in this project's default
pose, so no tilt correction is needed there the way the arms need one), or
sleeve_length (the whole arm, via LBS-weight-based arm classification).
Vertices outside any relevant region for the current garment kind (legs
for a top, arms for pants) are left at the body's own skin tone.
"""

import os
import sys

import numpy as np
import torch

MEASUREMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../measurement"))
sys.path.insert(0, MEASUREMENT_DIR)

from smpl_fit import (  # noqa: E402
    load_smpl_model, forward, scale_to_height, compute_anatomical_levels,
    PELVIS, LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from mesh_utils import ARM_JOINTS, LEFT_FOOT, RIGHT_FOOT  # noqa: E402
from fabric_ease_reference import (  # noqa: E402
    FABRIC_TYPES, PRODUCT_FABRIC_TYPE, STANDARD_BODY_MEASUREMENTS_CM,
    BOTTOMS_FABRIC_TYPES, STANDARD_BOTTOMS_MEASUREMENTS_CM,
)

# Which measurement part governs which torso/leg Y-level, and which garment
# kinds actually have a torso/arm/leg region relevant to them at all (a "top"
# covers neck-to-hip + arms; "pants" covers waist-to-ankle_hem (legs), not
# arms; "dress" covers neck-to-hip + arms, same as a top, just longer -- the
# skirt portion below hip isn't covered by any measurement in the fabric-ease
# reference, so it stays uncolored, same honesty-over-false-precision call
# made elsewhere in this project for parts with no real calibration data).
# "knee" has no per-user measurement (no knee circumference is predicted by
# this project's BodyM model) and stays permanently uncolored, same
# treatment as "neck".
GARMENT_KIND_REGIONS = {
    "top": {"neck", "chest", "waist", "hip", "shoulder", "sleeve_length"},
    "dress": {"neck", "chest", "waist", "hip", "shoulder", "sleeve_length"},
    "pants": {"waist", "hip", "thigh", "knee", "ankle_hem"},
}


def _get_avatar_geometry(betas, height_cm):
    model = load_smpl_model()
    vertices, joints = forward(model, betas)
    vertices, joints = scale_to_height(vertices, joints, height_cm)
    lbs_weights = model.lbs_weights.detach().numpy()
    return vertices, joints, lbs_weights, model.faces


def _measure_shoulder_seam_width(vertices, joints, lbs_weights, arm_weight_threshold=0.5, band_cm=1.5):
    """A tailor's "shoulder" measurement is taken seam-to-seam across the
    back, at the point where a garment's shoulder seam actually sits -- the
    boundary between torso and arm at the top of the deltoid. This project's
    existing shoulder-breadth (used for avatar fitting in smpl_fit.py, left
    untouched here) is instead the straight-line distance between the
    LEFT_SHOULDER/RIGHT_SHOULDER SKELETON JOINTS, which sit inside the body,
    not on its surface -- validated against real tape-measured ground truth
    (person_007: real shoulder-breadth 38cm) to be several cm short of what
    a real garment's published "shoulder" spec means, a gap that doesn't
    shrink even with accurate prediction (see measurement_heatmap.py's
    PART_TO_MEASUREMENT_KEY comment). This is a SEPARATE measurement, used
    only for comparing against real garment specs here -- it must not
    replace the joint-based one used for BodyM-trained fitting, which is
    validated and correct for its own purpose.

    Finds the outermost TORSO-classified (not arm) vertex on each side
    within a narrow band around shoulder height -- i.e. the actual
    torso/arm boundary, which is where a garment's shoulder seam sits, not
    wherever the internal ball joint happens to be -- and returns the
    straight-line distance between them. Returns None if no clean boundary
    vertex is found on both sides (caller should fall back to the
    joint-based distance in that case).
    """
    levels = compute_anatomical_levels(joints)
    shoulder_y = levels["shoulder"]

    arm_weight = lbs_weights[:, list(ARM_JOINTS)].sum(axis=1)
    is_torso = arm_weight <= arm_weight_threshold

    ys = vertices[:, 1]
    xs = vertices[:, 0]
    in_band = np.abs(ys - shoulder_y) < band_cm

    left_indices = np.where(is_torso & in_band & (xs < 0))[0]
    right_indices = np.where(is_torso & in_band & (xs > 0))[0]
    if len(left_indices) == 0 or len(right_indices) == 0:
        return None

    left_point = vertices[left_indices[np.argmin(xs[left_indices])]]
    right_point = vertices[right_indices[np.argmax(xs[right_indices])]]
    return float(np.linalg.norm(left_point - right_point))


def _classify_regions(vertices, joints, lbs_weights, arm_weight_threshold=0.5):
    """Returns (region_labels, region_progress) -- region_labels is a list
    of str (one of "neck"/"shoulder"/"chest"/"waist"/"hip"/"thigh"/"knee"/
    "ankle_hem"/"sleeve_length"/None) per vertex; region_progress is only
    meaningful for arm vertices (0 at the shoulder joint, 1 at the wrist),
    kept for any future finer-grained sleeve work, not required by callers.

    "shoulder" is a TORSO-level band across the top of the torso (over the
    trapezius, between the neck and where the chest begins) -- deliberately
    NOT a cap on the arm. This has to match where
    _measure_shoulder_seam_width() actually samples the mesh (a band of
    torso-classified vertices around shoulder height, the same place a
    garment's shoulder seam sits and a real seam-to-seam tape measurement
    is taken): an earlier version of this function put the "shoulder"
    label on the arm instead (the deltoid cap nearest the joint, t<0.3
    along the arm), which meant the shoulder fit score was being painted
    onto a different piece of geometry than the one it was actually
    measured from, while the true top-of-torso area it WAS measured from
    fell into the much broader "neck" band above and stayed permanently
    uncolored. All arm vertices are "sleeve_length" now; there's no
    separate arm-side "shoulder" region.

    Legs (below leg_cutoff_y) are classified the same nearest-Y-level way
    as the torso instead of being left uncolored -- this project's default
    pose keeps the legs in SMPL's unrotated rest pose (vertical, unlike the
    deliberately-tilted arms), so a plain Y-level split is exact here, no
    tilt-correction needed the way _measure_shoulder_seam_width's arm work
    needed one. "knee" gets its own band purely so thigh/ankle_hem don't
    silently swallow it; it's never scored (no per-user knee circumference
    exists), same honest-gap treatment as "neck"."""

    n = len(vertices)
    arm_weight = lbs_weights[:, list(ARM_JOINTS)].sum(axis=1)
    is_arm = arm_weight > arm_weight_threshold

    levels = compute_anatomical_levels(joints)
    # neck: a narrow collar/throat band above the shoulder line.
    # shoulder: centered ON the shoulder line itself, matching
    # _measure_shoulder_seam_width()'s own sampling band -- so this label
    # covers the flat top of the torso, not just a knife-edge at shoulder_y.
    # chest/waist/hip unchanged. leg_cutoff_y is a bit below hip, past which
    # torso vertices are really legs.
    torso_length = levels["shoulder"] - joints[PELVIS][1]
    neck_y = levels["shoulder"] + 0.12 * torso_length
    shoulder_y = levels["shoulder"]
    leg_cutoff_y = levels["hip"] - 0.15 * torso_length

    torso_level_names = ["neck", "shoulder", "chest", "waist", "hip"]
    torso_level_ys = np.array([neck_y, shoulder_y, levels["chest"], levels["waist"], levels["hip"]])

    knee_y = (joints[LEFT_KNEE][1] + joints[RIGHT_KNEE][1]) / 2.0
    ankle_y = (joints[LEFT_ANKLE][1] + joints[RIGHT_ANKLE][1]) / 2.0
    thigh_y = (leg_cutoff_y + knee_y) / 2.0

    # Below the ankle joint is the foot, not a pant leg opening -- without an
    # explicit cutoff there, "ankle_hem" is the lowest reference band with
    # nothing below it to compete for nearest-level votes, so the whole foot
    # (heel to toes, all the way to the floor) gets pulled into "ankle_hem"
    # too. Confirmed directly: 492 of 682 ankle_hem-classified vertices sat
    # within 8cm of the floor. SMPL's own "foot" joint sits inside the foot
    # itself (toward the ball of the foot, y~1.9cm on a 175cm-tall neutral
    # body vs the ankle joint's y~7.2cm), so the midpoint between the ankle
    # and foot joints is a reasonable real ankle-cuff/foot boundary.
    foot_y = (joints[LEFT_FOOT][1] + joints[RIGHT_FOOT][1]) / 2.0
    foot_cutoff_y = (ankle_y + foot_y) / 2.0

    leg_level_names = ["thigh", "knee", "ankle_hem"]
    leg_level_ys = np.array([thigh_y, knee_y, ankle_y])

    left_shoulder_pos = joints[LEFT_SHOULDER]
    right_shoulder_pos = joints[RIGHT_SHOULDER]
    left_wrist_pos = joints[LEFT_WRIST]
    right_wrist_pos = joints[RIGHT_WRIST]

    regions = [None] * n
    progress = np.zeros(n)

    ys = vertices[:, 1]
    for i in range(n):
        if is_arm[i]:
            v = vertices[i]
            # Which arm (left/right) this vertex is closer to, then project
            # onto that arm's shoulder->wrist axis to get 0..1 progress.
            d_left = np.linalg.norm(v - left_shoulder_pos)
            d_right = np.linalg.norm(v - right_shoulder_pos)
            if d_left <= d_right:
                shoulder_pos, wrist_pos = left_shoulder_pos, left_wrist_pos
            else:
                shoulder_pos, wrist_pos = right_shoulder_pos, right_wrist_pos
            axis = wrist_pos - shoulder_pos
            axis_len2 = np.dot(axis, axis)
            t = 0.0 if axis_len2 < 1e-9 else np.dot(v - shoulder_pos, axis) / axis_len2
            t = float(np.clip(t, 0.0, 1.0))
            progress[i] = t
            regions[i] = "sleeve_length"
        elif ys[i] < foot_cutoff_y:
            regions[i] = None  # foot -- not covered by any pant/hem measurement
        elif ys[i] < leg_cutoff_y:
            nearest = np.argmin(np.abs(leg_level_ys - ys[i]))
            regions[i] = leg_level_names[nearest]
        else:
            nearest = np.argmin(np.abs(torso_level_ys - ys[i]))
            regions[i] = torso_level_names[nearest]

    return regions, progress


_RED = (255, 0, 0)     # tight
_GREEN = (0, 255, 0)   # good fit
_BLUE = (0, 0, 255)    # loose


def _ease_diff_to_color(diff_cm, tight_cm=-2.0, loose_cm=2.0):
    """Three flat colors only, no gradient/blending: red below tight_cm,
    green between tight_cm and loose_cm, blue above loose_cm. An earlier
    version faded continuously toward white past the tight/loose band (to
    keep two very-different-but-both-loose regions visually distinct), but
    per explicit request this reverts to plain discrete red/green/blue with
    nothing in between."""
    if diff_cm < tight_cm:
        return _RED
    if diff_cm > loose_cm:
        return _BLUE
    return _GREEN


_SKIN_COLOR = (216, 168, 120)  # matches the bare-avatar viewer's 0xd8a878


def _colors_for_regions(regions, relevant_parts, region_scores, tolerance_cm=2.0):
    colors = []
    for region in regions:
        if region is None or region not in relevant_parts or region not in region_scores:
            colors.append(_SKIN_COLOR)
            continue
        colors.append(_ease_diff_to_color(region_scores[region], tight_cm=-tolerance_cm, loose_cm=tolerance_cm))
    return colors


PART_TO_MEASUREMENT_KEY = {
    # region name -> (garment spec key, user measurement key)
    "chest": ("chest", "chest"),
    "waist": ("waist", "waist"),
    "hip": ("hip", "hip"),
    # "shoulder" is handled separately in compute_fit_heatmap(): the garment
    # key is still "shoulder", but the user-side value comes from
    # _measure_shoulder_seam_width() run on this session's own mesh, not
    # from user_measurements (see the detailed comment at that call site for
    # why the plain joint-based shoulder-breadth doesn't work here).
    "shoulder": ("shoulder", None),
    # Real per-garment sleeve-length and neck measurements don't exist in
    # any current PRODUCTS entry (tops only record chest/shoulder/body
    # length). A garment's overall body "length" was tried as a sleeve-length
    # stand-in, but it's a different anatomical measurement entirely (~70cm
    # body length vs ~47cm arm length) -- comparing them produced a
    # meaningless, always-maximally-loose signal, not a rough approximation.
    # Left unmapped (None) so these regions stay uncolored (honest gap)
    # rather than showing a confidently wrong number; a real fix needs
    # actual sleeve-length data added to PRODUCTS per garment.
    "sleeve_length": (None, None),
    "neck": (None, None),
}

# Used only by compute_fit_heatmap_by_fabric (the generic, no-specific-product
# mode below) -- there, unlike PART_TO_MEASUREMENT_KEY's real-product mode,
# "sleeve_length" DOES have a valid user-side counterpart: the standardized
# chart's sleeve_length is a true sleeve/arm length (61-68.5cm across sizes),
# which lines up with the BodyM predictor's own "arm-length" measurement --
# unlike a real product's overall body "length" field, which isn't the same
# quantity at all (see PART_TO_MEASUREMENT_KEY's sleeve_length comment).
# "neck" still has no counterpart: the BodyM-trained predictor never
# produces a neck measurement (its 5 targets are chest/waist/hip/
# shoulder-breadth/arm-length only), so it stays unmapped here too.
GENERIC_PART_TO_USER_KEY_TOPS = {
    "chest": "chest",
    "waist": "waist",
    "hip": "hip",
    "sleeve_length": "arm-length",
}

# Bottoms equivalent. "ankle_hem" maps to the predictor's "ankle"
# circumference -- same "garment opening ~= body circumference + ease"
# convention already used for chest/waist/hip/sleeve_length.
#
# "knee"/"inseam"/"outseam"/"rise" are all left unmapped: none of the 10
# bottoms fabrics currently in BOTTOMS_FABRIC_TYPES even specify an
# ease_by_part_cm for inseam/outseam (only waist/hip/thigh/knee/ankle_hem/
# rise), so mapping those here would be inert given the actual data. If a
# future fabric or product adds an inseam ease value, note that
# user_measurements' "leg-length" (hip-JOINT-to-ankle, per smpl_fit.py's
# measure_smpl_mesh) is NOT the same quantity as a true inseam
# (crotch-to-ankle) -- the hip joint sits a few cm above the actual crotch
# fork (confirmed directly this session while fixing the shoulder-to-crotch
# measurement: ~86cm hip-joint level vs ~80cm true crotch level on the same
# body) -- so it would need to be flagged as an approximation, not treated
# as exact. "knee" has no per-user knee circumference at all (same as
# "neck" for tops); "rise" (crotch-to-waist) and "outseam" (waist-to-ankle)
# have no matching BodyM measurement either.
GENERIC_PART_TO_USER_KEY_BOTTOMS = {
    "waist": "waist",
    "hip": "hip",
    "thigh": "thigh",
    "ankle_hem": "ankle",
}


def compute_fit_heatmap(product_key, product, size, user_measurements, betas, height_cm):
    """
    product_key: the PRODUCTS dict key (e.g. "nobero_polo") -- used to look
        up which fabric type represents this product, since that mapping
        is keyed by product identity, not anything stored in the product
        dict itself.
    product: one entry from real_garment.py's PRODUCTS dict (needs "kind"
        and "size_guide").
    size: e.g. "M" -- must be a key in product["size_guide"].
    user_measurements: this session's predicted measurements dict (chest,
        waist, hip, shoulder-breadth, arm-length, ...).
    betas, height_cm: this session's fitted SMPL shape + real height, used
        to regenerate the exact same avatar mesh already shown in the
        viewer (not a garment mesh).

    Returns {"vertices": [[x,y,z],...], "faces": [[i,j,k],...],
             "colors": [[r,g,b],...], "region_scores": {part: diff_cm}}
    -- vertices/faces are the plain AVATAR mesh (same one showMesh() already
    renders), colors are per-vertex tightness, region_scores is the raw
    per-part ease-difference numbers for display in a stats panel.

    Note on "sleeve_length": the fabric-ease reference's sleeve_length ease is
    defined relative to a garment's own sleeve-length measurement, which
    real_garment.py's PRODUCTS doesn't record for any current product (only
    chest/shoulder/length for tops). Using the garment's overall body
    "length" as the closest available stand-in is an approximation, flagged
    here rather than silently treated as exact -- a real fix would mean
    adding actual sleeve-length data to PRODUCTS per garment.
    """
    kind = product["kind"]
    relevant_parts = GARMENT_KIND_REGIONS.get(kind, set())
    spec = product["size_guide"][size]

    fabric_key = PRODUCT_FABRIC_TYPE.get(product_key)
    fabric = FABRIC_TYPES.get(fabric_key) if fabric_key else None
    if fabric is None:
        # Fall back to a neutral middle-of-the-road knit if this product
        # has no mapped fabric type yet, rather than failing outright.
        fabric = FABRIC_TYPES["cotton_jersey_knit"]

    vertices, joints, lbs_weights, faces = _get_avatar_geometry(betas, height_cm)

    # "shoulder" needs a value computed fresh off this avatar's own mesh
    # (see _measure_shoulder_seam_width's docstring) rather than the BodyM
    # joint-based user_measurements["shoulder-breadth"] -- that number and
    # a real garment's "shoulder" spec are on different measurement
    # conventions (anatomical joint-to-joint vs. tailoring seam-to-seam
    # across the back), so diffing them was always off by ~6-9cm regardless
    # of how accurate the joint prediction was. Verified against person_007
    # (real height 175cm): the joint-based predictor gives 35.98cm against
    # a real ground truth of 38cm (a normal ~2cm error) yet still lands 8.5cm
    # short of the size-M polo's 44.45cm spec; the mesh-surface measurement
    # here gives 42.56cm for the same body -- overshooting the anatomical
    # ground truth by ~4.5cm on purpose (it follows the shoulder's curve
    # further out, the way a tailor's tape does over a seam), which brings
    # the gap against the garment spec down to a plausible ~1.9cm instead.
    user_shoulder = _measure_shoulder_seam_width(vertices, joints, lbs_weights)

    region_scores = {}
    for part in relevant_parts:
        garment_key, user_key = PART_TO_MEASUREMENT_KEY.get(part, (None, None))
        if part == "shoulder":
            if "shoulder" not in spec or user_shoulder is None:
                continue
            actual_ease = spec["shoulder"] - user_shoulder
        else:
            if garment_key is None or garment_key not in spec or user_key not in user_measurements:
                continue
            actual_ease = spec[garment_key] - user_measurements[user_key]
        expected_ease = fabric["ease_by_part_cm"].get(part)
        if expected_ease is None:
            continue
        region_scores[part] = actual_ease - expected_ease

    tolerance_cm = fabric.get("tolerance_cm", 2.0)
    regions, _ = _classify_regions(vertices, joints, lbs_weights)
    colors = _colors_for_regions(regions, relevant_parts, region_scores, tolerance_cm)

    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "colors": [list(c) for c in colors],
        "region_scores": region_scores,
        "fabric_label": fabric["label"],
        "tolerance_cm": tolerance_cm,
    }


def compute_fit_heatmap_by_fabric(fabric_key, size, user_measurements, betas, height_cm, garment_kind="top"):
    """
    Generic mode: instead of a specific real product's hand-measured spec,
    builds a hypothetical garment measurement directly from the standardized
    size chart the user supplied (STANDARD_BODY_MEASUREMENTS_CM for tops,
    STANDARD_BOTTOMS_MEASUREMENTS_CM for pants/shorts) plus the chosen
    fabric's own ease_by_part_cm, exactly per that chart's own stated
    formula: garment_measurement[part] = standard_body[size][part] + ease[part].
    Lets someone check "if I bought a size M in this fabric, would it fit
    me" from a garment-category + fabric-type + size selection, with no
    specific real product/listing involved.

    fabric_key: a key into fabric_ease_reference.FABRIC_TYPES (tops) or
        BOTTOMS_FABRIC_TYPES (bottoms) -- whichever garment_kind selects.
    size: a key into the matching standard chart -- S/M/L/XL/XXL/XXXL for
        tops, but "28"/"30"/.../"40" (waist size in inches, as a string)
        for bottoms, matching how pants are actually sized in stores.
    garment_kind: "top" (default) or "bottom" -- picks the chart, fabric
        catalog, and user-measurement mapping to use.
    user_measurements, betas, height_cm: same as compute_fit_heatmap.

    Note on the diff computed below: it reduces algebraically to
    (standard_body[size][part] - user_measurements[part]). Since the
    hypothetical garment measurement was built by adding this fabric's own
    expected ease to the standard body measurement, that same ease cancels
    out of (actual_ease - expected_ease) -- comparing the same body against
    the same size in two different fabrics gives an identical raw diff for
    both. That's correct, not a bug: once a garment has been cut with the
    right ease for its fabric, whether it actually fits comes down to how
    the wearer's own body compares to the standard body that size/fabric
    was drafted for. What differs per fabric is how forgiving that
    deviation is to actually wear -- a stretch fabric can absorb a shortfall
    a rigid weave can't -- so the fabric's own stretchiness instead shows up
    in tolerance_cm (see fabric_ease_reference.CATEGORY_TOLERANCE_CM),
    returned alongside region_scores and used to widen or narrow the
    tight/good/loose color band this same diff is rendered with.
    """
    if garment_kind == "bottom":
        fabric_catalog, standard_chart, user_key_map = (
            BOTTOMS_FABRIC_TYPES, STANDARD_BOTTOMS_MEASUREMENTS_CM, GENERIC_PART_TO_USER_KEY_BOTTOMS,
        )
    elif garment_kind == "top":
        fabric_catalog, standard_chart, user_key_map = (
            FABRIC_TYPES, STANDARD_BODY_MEASUREMENTS_CM, GENERIC_PART_TO_USER_KEY_TOPS,
        )
    else:
        raise ValueError(f"Unknown garment_kind: {garment_kind}")

    if fabric_key not in fabric_catalog:
        raise ValueError(f"Unknown fabric type: {fabric_key}")
    if size not in standard_chart:
        raise ValueError(f"Unknown size: {size}")

    fabric = fabric_catalog[fabric_key]
    standard_body = standard_chart[size]

    vertices, joints, lbs_weights, faces = _get_avatar_geometry(betas, height_cm)
    # Only tops/dresses need the mesh-measured shoulder seam width;
    # skip it for bottoms (needlessly running a mesh search that's never
    # used and whose narrow-band assumptions were only validated for the
    # torso's shoulder line).
    user_shoulder = _measure_shoulder_seam_width(vertices, joints, lbs_weights) if garment_kind == "top" else None

    region_scores = {}
    for part, standard_value in standard_body.items():
        expected_ease = fabric["ease_by_part_cm"].get(part)
        if expected_ease is None:
            continue
        if part == "shoulder":
            user_value = user_shoulder
        else:
            user_key = user_key_map.get(part)
            user_value = user_measurements.get(user_key) if user_key else None
        if user_value is None:
            continue
        garment_measurement = standard_value + expected_ease
        actual_ease = garment_measurement - user_value
        region_scores[part] = actual_ease - expected_ease

    tolerance_cm = fabric.get("tolerance_cm", 2.0)
    relevant_parts = set(standard_body.keys())
    regions, _ = _classify_regions(vertices, joints, lbs_weights)
    colors = _colors_for_regions(regions, relevant_parts, region_scores, tolerance_cm)

    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "colors": [list(c) for c in colors],
        "region_scores": region_scores,
        "fabric_label": fabric["label"],
        "tolerance_cm": tolerance_cm,
    }
