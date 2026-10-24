"""
Module 2 (Avatar Generation): fit SMPL shape parameters so the resulting
mesh's own measurements match a person's predicted measurements (from the
HGB baseline in bodym_baseline.py) as closely as possible, producing a
personalized 3D avatar -- per the project report's Sub-Problem 2 ("fit SMPL
parametric 3D body model to estimated measurements").

Targets all 13 BodyM measurements (chest/waist/hip/shoulder-breadth/
arm-length/ankle/bicep/calf/forearm/leg-length/shoulder-to-crotch/thigh/
wrist) -- see measure_smpl_mesh() and fit_betas() for how each is measured
directly off the SMPL mesh and combined into one fitting loss.

Requires an SMPL model file at smpl_models/smpl/SMPL_NEUTRAL.pkl (license-
gated, not in git -- see .gitignore). Copy your own SMPL_NEUTRAL.pkl there.

Usage:
    python smpl_fit.py --height 167 --chest 85 --waist 75 --hip 90 \
        --shoulder-breadth 38 --arm-length 52 --out person_007_avatar.obj
"""

import argparse
import os

import numpy as np
import smplx
import torch
from scipy.optimize import minimize

from mesh_utils import (
    get_torso_faces, measure_level, ellipse_circumference,
    get_side_faces, measure_plane, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "smpl_models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "bodym_output")

# Standard SMPL 24-joint skeleton (matches mesh_utils.py's constants).
PELVIS = 0
LEFT_HIP = 1
RIGHT_HIP = 2
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_WRIST = 20
RIGHT_WRIST = 21

N_FIT_BETAS = 6  # first 6 shape components capture most size/build variation;
                 # remaining betas fixed at 0 to keep a 5-measurement fit well-posed.

# SMPL's own zero/rest pose holds the arms fully horizontal (a T-pose, ~90
# degrees from hanging down) -- measured directly, not assumed. GarmentCode's
# own reference body instead poses its arms much closer to the torso (~40
# degrees from hanging down), which is what its sewing patterns' sleeves are
# actually shaped to wrap around. Using the full T-pose for our avatar meant
# every sleeved garment had to reach across far more empty space than the
# pattern was designed for, with nothing there to taper against -- confirmed
# empirically (reducing sleeve puff fabric didn't fix the resulting
# fan/wing shape; the arm being too far away did). -50 degrees of shoulder
# rotation (measured) lands the arm at ~40 degrees from vertical, matching
# GarmentCode's convention. Applied to the avatar's own default pose (not
# just the garment-drape export) so the two stay in the same pose -- a
# mismatch there previously caused a different bug (a gaping neckline).
ARM_LOWER_ANGLE_DEG = -50
LEFT_SHOULDER_Z = (LEFT_SHOULDER - 1) * 3 + 2
RIGHT_SHOULDER_Z = (RIGHT_SHOULDER - 1) * 3 + 2
# Flexion axis for the knee joint (bending the shin back under a forward-
# tilted thigh) -- confirmed empirically, not assumed: a seated-pose test
# (hip -90deg + knee +90deg, both axis 0) produced the correct "L-shaped"
# leg silhouette on the first try (see garment_export.py's sitting-pose
# investigation). Axis 0 = X, matching mesh_utils.py's joint-index
# convention (LEFT_KNEE=4, RIGHT_KNEE=5 in SMPL's 24-joint skeleton).
LEFT_KNEE_X = (LEFT_KNEE - 1) * 3 + 0
RIGHT_KNEE_X = (RIGHT_KNEE - 1) * 3 + 0
# Ab/adduction axis for the hip joint (spreading both legs apart sideways
# into a wider stance) -- confirmed empirically, not assumed: axis 0 (the
# same axis used for knee/hip FLEXION) turned out to be the wrong one for
# this; a first attempt at axis 2 with equal-and-opposite signs on each leg
# produced a crossed-leg X-shape (each ankle ending up on the OPPOSITE side
# from its own hip), confirmed by plotting the actual joint positions, not
# just by the separation numbers (which looked plausible even though the
# legs had crossed). Flipping which leg gets the positive vs negative sign
# (still axis 2) gave a clean, monotonically widening "V" stance instead
# (hip separation < knee separation < ankle separation, each leg staying on
# its own side throughout) -- see garment_export.py's leg-spread
# investigation for the sign convention this landed on.
LEFT_HIP_Z = (LEFT_HIP - 1) * 3 + 2
RIGHT_HIP_Z = (RIGHT_HIP - 1) * 3 + 2


def default_body_pose(arm_angle_deg=ARM_LOWER_ANGLE_DEG, knee_angle_deg=0.0, hip_spread_deg=0.0):
    """arm_angle_deg: defaults to the empirically-tuned angle GarmentCode's
    own sleeve patterns assume (see the comment above) -- every caller that
    doesn't pass this explicitly gets the exact same pose as before. An
    override exists for the multi-pose fit check (garment_export.py):
    testing a MODEST variation around this angle, not a dramatically
    different pose like arms raised or fully at the sides -- a sleeve's
    shape is cut for THIS specific arm angle, so a large deviation would
    reintroduce the same "sleeve reaching across empty space" bug this
    angle was originally tuned to fix, not test anything meaningful about
    real-world fit robustness.

    knee_angle_deg: 0 (default) keeps the same straight-leg standing pose
    every existing caller already used -- no behavior change unless a
    caller passes this explicitly. A nonzero value bends both knees forward
    by that many degrees, for the pants-kind leg-variation pose check
    (garment_export.py's check_fit_pose_robustness) -- NOT the same thing
    as a seated pose (no hip flexion here, just a slight standing-stance
    knee give). Calibrated against nobero_cargo_pants on the real Warp
    engine: 15deg is clean (0 body-cloth intersections), 30deg already
    shows real strain (77 intersections) well short of a full 90deg seated
    bend's outright simulation failure -- the pose check stays inside the
    clean zone.

    hip_spread_deg: 0 (default) keeps the same straight-leg stance. A
    nonzero value spreads both legs apart sideways into a wider standing
    stance -- a SECOND, independent pants-kind leg-variation pose check,
    alongside knee_angle_deg (see check_fit_pose_robustness). Calibrated
    against nobero_cargo_pants on the real Warp engine: 10deg is clean (0
    body-cloth intersections, confirmed visually), 20deg already shows
    real strain (44 intersections, though still visually fine), and 35deg
    is a confirmed, VISIBLE failure -- the inseam sags into a droopy pouch
    between the legs and the shins come out exposed, the fabric unable to
    reach both widely-spread legs from a pattern cut for a normal stance.
    This has a narrower safe margin than knee_angle_deg's own 30deg strain
    point, which makes sense: spreading stresses the inseam/crotch seam's
    own fixed length directly, which has much less slack to give than a
    knee bend does. The pose check stays well inside the confirmed-clean
    zone (10deg), not the merely-still-visually-fine 20deg."""
    pose = torch.zeros(1, 69)
    angle = np.radians(arm_angle_deg)
    pose[0, LEFT_SHOULDER_Z] = angle
    pose[0, RIGHT_SHOULDER_Z] = -angle
    if knee_angle_deg:
        knee_angle = np.radians(knee_angle_deg)
        pose[0, LEFT_KNEE_X] = knee_angle
        pose[0, RIGHT_KNEE_X] = knee_angle
    if hip_spread_deg:
        spread_angle = np.radians(hip_spread_deg)
        pose[0, LEFT_HIP_Z] = spread_angle
        pose[0, RIGHT_HIP_Z] = -spread_angle
    return pose


def load_smpl_model():
    model_path = os.path.join(MODEL_DIR, "smpl", "SMPL_NEUTRAL.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SMPL model not found at {model_path}. Copy your own "
            f"SMPL_NEUTRAL.pkl there (license-gated, see .gitignore)."
        )
    return smplx.create(MODEL_DIR, model_type="smpl", gender="neutral")


def forward(model, betas_np):
    betas = torch.zeros(1, model.num_betas, dtype=torch.float32)
    betas[0, :len(betas_np)] = torch.tensor(betas_np, dtype=torch.float32)

    with torch.no_grad():
        output = model(
            betas=betas,
            body_pose=default_body_pose(),
            global_orient=torch.zeros(1, 3),
        )

    vertices = output.vertices[0].numpy().astype(np.float64)
    joints = output.joints[0].numpy().astype(np.float64)
    return vertices, joints


def scale_to_height(vertices, joints, height_cm):
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    current_height = y_max - y_min
    scale = height_cm / current_height
    vertices = vertices * scale
    joints = joints * scale

    # Anchor feet to y=0 (floor), not just scaled -- otherwise the mesh keeps
    # SMPL's raw pelvis-relative origin (feet end up a few dozen cm below/above
    # y=0 depending on shape). Garment draping (garment_export.py) assumes the
    # same floor-anchored convention for its own body export, so without this
    # shift here, a draped garment ends up offset from where this avatar's
    # body actually is when both are shown together in the viewer.
    floor_shift = vertices[:, 1].min()
    vertices[:, 1] -= floor_shift
    joints[:, 1] -= floor_shift
    return vertices, joints


def compute_anatomical_levels(joints):
    shoulder_y = (joints[LEFT_SHOULDER][1] + joints[RIGHT_SHOULDER][1]) / 2.0
    hip_y = (joints[LEFT_HIP][1] + joints[RIGHT_HIP][1]) / 2.0
    mid_hip_y = joints[PELVIS][1]

    torso_length = shoulder_y - mid_hip_y

    chest_y = shoulder_y - 0.35 * torso_length
    waist_y = shoulder_y - 0.68 * torso_length

    return {"shoulder": shoulder_y, "chest": chest_y, "waist": waist_y, "hip": hip_y}


def find_level_single(vertices, faces, y_start, max_offset=5.0, step=0.5):
    """Like mesh_utils.find_clean_level, but for one complete mesh (no
    front/side cross-validation needed -- a clean synthetic SMPL mesh
    almost always gives one contour on the first try)."""

    offsets = [0.0]
    n_steps = int(max_offset / step)
    for i in range(1, n_steps + 1):
        offsets.append(i * step)
        offsets.append(-i * step)

    for offset in offsets:
        y = y_start + offset
        result = measure_level(vertices, faces, y)
        if result is not None and result["num_contours"] == 1:
            return result

    return None


def find_waist_single(vertices, faces, y_start, y_end, num_samples=8):
    best_result = None
    best_circumference = np.inf

    for y in np.linspace(y_start, y_end, num_samples):
        result = measure_level(vertices, faces, y)
        if result is None or result["num_contours"] != 1:
            continue
        if result["circumference"] < best_circumference:
            best_circumference = result["circumference"]
            best_result = result

    return best_result


def find_extremum_single(vertices, faces, y_start, y_end, num_samples=5, mode="max", max_contours=2):
    """Like find_waist_single (which always searches for the MINIMUM
    circumference -- the waist is the narrowest point of the torso), but
    generalized to also search for a MAXIMUM (the calf/thigh/bicep/forearm
    are each the WIDEST point of their limb segment, not the narrowest).
    Also tolerates up to `max_contours` closed rings per level instead of
    requiring exactly one: below the crotch, a torso-faces Y-slice
    naturally produces two separate rings (left leg + right leg), and
    measure_level's own contour selection already returns just the larger
    one's circumference, so accepting num_contours==2 here is correct, not
    a fallback -- unlike the torso (chest/waist/hip), which should only
    ever have a single ring at a clean level."""
    best_result = None
    best_c = -np.inf if mode == "max" else np.inf

    for y in np.linspace(y_start, y_end, num_samples):
        result = measure_level(vertices, faces, y)
        if result is None or result["num_contours"] > max_contours:
            continue
        c = result["circumference"]
        better = (c > best_c) if mode == "max" else (c < best_c)
        if better:
            best_c = c
            best_result = result

    return best_result


def find_crotch_level(vertices, faces, y_start, y_end, num_samples=6):
    """Find the highest Y between y_start (hip JOINT level) and y_end (knee
    level) where torso_faces has visibly split into two separate leg
    rings. Needed because the hip joint itself sits above where the legs
    actually fork -- confirmed directly: on a neutral-shape mesh the hip
    joint level sits at y=86.4 while the true fork isn't until ~y=80.3,
    and the band in between is a saddle-shaped transition zone where the
    single "ring" build_contours finds is a degenerate, wrongly-stitched
    loop with an inflated perimeter (110.4cm measured there -- bigger than
    the hip circumference itself, 105.8cm). Both the thigh search and
    shoulder-to-crotch need to stay clear of this band rather than
    treating the hip joint (or SMPL's "pelvis" root joint, which sits even
    higher still) as the crotch."""
    for y in np.linspace(y_start, y_end, num_samples):
        result = measure_level(vertices, faces, y)
        if result is not None and result["num_contours"] == 2:
            return y
    return (y_start + y_end) / 2.0  # fallback: no clean transition found


def find_extremum_along_axis(vertices, faces, p0, p1, t_start, t_end, num_samples=5, mode="max"):
    """Same idea as find_extremum_single, but for a limb segment that isn't
    vertical (the arms, posed at a tilt -- see measure_plane's docstring):
    slices perpendicular to the p0->p1 axis at each sampled fraction t
    instead of at a fixed world height, so the cross-section is the limb's
    true (non-elongated) circumference at that point along its own length."""
    axis = p1 - p0
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-9:
        return None
    normal = axis / axis_len

    best_result = None
    best_c = -np.inf if mode == "max" else np.inf

    for t in np.linspace(t_start, t_end, num_samples):
        point = p0 + t * axis
        result = measure_plane(vertices, faces, point, normal)
        if result is None:
            continue
        c = result["circumference"]
        better = (c > best_c) if mode == "max" else (c < best_c)
        if better:
            best_c = c
            best_result = result

    return best_result


def measure_smpl_mesh(model, torso_faces, betas_np, height_cm, lbs_weights=None):
    """Measures all 13 BodyM measurements directly off the fitted SMPL
    mesh. The first 5 (chest/waist/hip/shoulder-breadth/arm-length) use the
    original torso-slice/joint-distance approach; the other 8 follow the
    same two techniques, picked per limb:

    - Legs stay in SMPL's unrotated rest pose (vertical), so a plain
      Y-height slice through torso_faces (which still contains both legs,
      only arms were excluded) is exact: ankle and thigh/calf are each
      found by searching for the narrowest/widest ring in the appropriate
      band (mirroring how the waist itself is found as the narrowest point
      between chest and hip).
    - Arms are deliberately posed at a tilt (see default_body_pose), so
      the same horizontal-slice approach would cut across them at an angle
      and overstate their circumference. Bicep/forearm/wrist instead slice
      perpendicular to each arm's own shoulder/elbow/wrist axis
      (find_extremum_along_axis), on a single-limb face subset
      (get_side_faces) so the (infinite) slicing plane can't accidentally
      intersect the torso or the other arm. Both sides are measured and
      averaged, same convention as the existing shoulder-breadth/arm-length.
    - leg-length and shoulder-to-crotch have no natural closed ring to
      measure (they're lengths, not circumferences) and no dedicated
      "crotch" joint exists in SMPL; both are computed as straight
      joint-to-joint distances (hip-to-ankle, and mid-shoulder-to-pelvis),
      the same kind of approximation already used for arm-length/
      shoulder-breadth.

    lbs_weights is only required for the 8 new measurements (to isolate
    single-limb faces) -- omit it to fall back to just the original 5, e.g.
    for any caller that hasn't been updated yet.
    """
    vertices, joints = forward(model, betas_np)
    vertices, joints = scale_to_height(vertices, joints, height_cm)

    levels = compute_anatomical_levels(joints)

    chest = find_level_single(vertices, torso_faces, levels["chest"])
    hip = find_level_single(vertices, torso_faces, levels["hip"])
    waist = find_waist_single(vertices, torso_faces, levels["chest"], levels["hip"])

    shoulder_breadth = np.linalg.norm(joints[LEFT_SHOULDER] - joints[RIGHT_SHOULDER])
    arm_length = (
        np.linalg.norm(joints[LEFT_SHOULDER] - joints[LEFT_WRIST])
        + np.linalg.norm(joints[RIGHT_SHOULDER] - joints[RIGHT_WRIST])
    ) / 2.0

    measured = {
        "chest": chest["circumference"] if chest else None,
        "waist": waist["circumference"] if waist else None,
        "hip": hip["circumference"] if hip else None,
        "shoulder-breadth": float(shoulder_breadth),
        "arm-length": float(arm_length),
    }

    if lbs_weights is None:
        return measured, vertices

    knee_y = (joints[LEFT_KNEE][1] + joints[RIGHT_KNEE][1]) / 2.0
    ankle_y = (joints[LEFT_ANKLE][1] + joints[RIGHT_ANKLE][1]) / 2.0
    hip_y = levels["hip"]

    crotch_y = find_crotch_level(vertices, torso_faces, hip_y, knee_y)

    ankle = find_extremum_single(vertices, torso_faces, ankle_y - 3.0, ankle_y + 3.0, mode="min")
    calf = find_extremum_single(vertices, torso_faces, ankle_y, knee_y, mode="max")
    # Stays strictly below crotch_y (not hip_y) -- see find_crotch_level's
    # docstring for why the hip-to-crotch band itself must be avoided.
    thigh = find_extremum_single(vertices, torso_faces, knee_y, crotch_y - 1.0, mode="max")

    leg_length = (
        (joints[LEFT_HIP][1] - joints[LEFT_ANKLE][1])
        + (joints[RIGHT_HIP][1] - joints[RIGHT_ANKLE][1])
    ) / 2.0
    mid_shoulder = (joints[LEFT_SHOULDER] + joints[RIGHT_SHOULDER]) / 2.0
    # Uses the actual mesh fork point, not the hip or pelvis JOINT (both
    # sit well above where the legs visually separate -- see
    # find_crotch_level's docstring).
    shoulder_to_crotch = mid_shoulder[1] - crotch_y

    # torso_faces already had every arm-touching face EXCLUDED (get_torso_faces),
    # so isolating a single arm's own faces has to start from the model's
    # full, unfiltered face list -- not torso_faces, which by construction
    # no longer contains any arm geometry at all.
    left_arm_faces = get_side_faces(model.faces, lbs_weights, LEFT_ARM_JOINTS)
    right_arm_faces = get_side_faces(model.faces, lbs_weights, RIGHT_ARM_JOINTS)

    def _bilateral_arm_measure(joint_a_l, joint_b_l, joint_a_r, joint_b_r, t_start, t_end, mode):
        left = find_extremum_along_axis(
            vertices, left_arm_faces, joints[joint_a_l], joints[joint_b_l], t_start, t_end, mode=mode,
        )
        right = find_extremum_along_axis(
            vertices, right_arm_faces, joints[joint_a_r], joints[joint_b_r], t_start, t_end, mode=mode,
        )
        vals = [r["circumference"] for r in (left, right) if r is not None]
        return float(np.mean(vals)) if vals else None

    bicep = _bilateral_arm_measure(LEFT_SHOULDER, LEFT_ELBOW, RIGHT_SHOULDER, RIGHT_ELBOW, 0.15, 0.85, "max")
    forearm = _bilateral_arm_measure(LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST, 0.15, 0.85, "max")
    wrist = _bilateral_arm_measure(LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST, 0.85, 0.98, "min")

    measured.update({
        "ankle": ankle["circumference"] if ankle else None,
        "bicep": bicep,
        "calf": calf["circumference"] if calf else None,
        "forearm": forearm,
        "leg-length": float(leg_length),
        "shoulder-to-crotch": float(shoulder_to_crotch),
        "thigh": thigh["circumference"] if thigh else None,
        "wrist": wrist,
    })

    return measured, vertices


ALL_TARGET_COLS = [
    "chest", "waist", "hip", "shoulder-breadth", "arm-length",
    "ankle", "bicep", "calf", "forearm", "leg-length",
    "shoulder-to-crotch", "thigh", "wrist",
]


def fit_betas(model, torso_faces, target, height_cm, n_betas=N_FIT_BETAS, lbs_weights=None):
    """Fits SMPL betas against however many of ALL_TARGET_COLS are present
    in `target` (13 if lbs_weights is given and target has all of them; 5
    if not, for backward compatibility). Uses RELATIVE error (measured -
    target) / target instead of raw cm differences before squaring and
    summing -- these 13 measurements span very different natural scales
    (wrist ~16cm vs hip ~110cm), so an unweighted sum of squared cm
    differences would let the largest-scale measurements dominate the fit
    and effectively ignore small ones like wrist/ankle even when they're
    proportionally far off. Relative error puts all 13 on comparable
    footing without needing hand-tuned per-measurement weights.

    Note: SMPL's shape space here is only n_betas (6) free parameters, so
    fitting 13 targets is a genuine least-squares COMPROMISE, not an exact
    solve -- expect small nonzero residuals on most measurements even at
    convergence, same as any other 6-parameters-to-13-targets fit.
    """
    target_cols = [c for c in ALL_TARGET_COLS if c in target]
    target_vec = np.array([target[c] for c in target_cols])

    eval_count = [0]

    def loss(betas_np):
        eval_count[0] += 1
        measured, _ = measure_smpl_mesh(model, torso_faces, betas_np, height_cm, lbs_weights=lbs_weights)
        pred_vec = np.array([
            measured[c] if measured[c] is not None else target[c] + target[c] * 0.5  # penalize failed slices
            for c in target_cols
        ])
        rel_err = (pred_vec - target_vec) / target_vec
        loss_val = np.sum(rel_err ** 2)
        print(f"  eval {eval_count[0]:3d}: betas={np.round(betas_np, 2)} "
              f"pred={np.round(pred_vec, 1)} loss={loss_val:.4f}")
        return loss_val

    # Measured directly on a real photo: maxiter=120/ftol=0.001 took 507
    # evals / 822s to reach loss=0.0141; dropping to maxiter=20/ftol=0.01
    # still ran 167 evals / 276s for loss=0.0149 -- barely any better,
    # because Powell's "maxiter" counts outer iterations, and each one runs
    # an internal per-direction line search that can call the loss function
    # many times regardless of that setting. maxfev instead caps the actual
    # number of loss evaluations directly, which is what wall-clock time
    # here actually scales with (~1.6s/eval). Capping at 45 lands close to
    # where the very first run's own eval log had already plateaued (loss
    # 0.0183 by eval 44), for a fit that finishes in well under a minute
    # instead of several.
    x0 = np.zeros(n_betas)
    result = minimize(loss, x0, method="Powell",
                       options={"maxiter": 20, "maxfev": 45, "xtol": 0.1, "ftol": 0.01})

    return result.x, result


def save_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--chest", type=float, required=True)
    p.add_argument("--waist", type=float, required=True)
    p.add_argument("--hip", type=float, required=True)
    p.add_argument("--shoulder-breadth", type=float, required=True)
    p.add_argument("--arm-length", type=float, required=True)
    # The other 8 BodyM measurements are optional on the CLI (kept required=False
    # so ad-hoc single-measurement testing still works); any omitted are simply
    # left out of the fit, same as fit_betas()'s general fallback behavior.
    p.add_argument("--ankle", type=float, default=None)
    p.add_argument("--bicep", type=float, default=None)
    p.add_argument("--calf", type=float, default=None)
    p.add_argument("--forearm", type=float, default=None)
    p.add_argument("--leg-length", type=float, default=None)
    p.add_argument("--shoulder-to-crotch", type=float, default=None)
    p.add_argument("--thigh", type=float, default=None)
    p.add_argument("--wrist", type=float, default=None)
    p.add_argument("--out", type=str, default="avatar.obj")
    return p.parse_args()


def main():
    args = parse_args()

    target = {
        "chest": args.chest,
        "waist": args.waist,
        "hip": args.hip,
        "shoulder-breadth": args.shoulder_breadth,
        "arm-length": args.arm_length,
    }
    optional = {
        "ankle": args.ankle, "bicep": args.bicep, "calf": args.calf,
        "forearm": args.forearm, "leg-length": args.leg_length,
        "shoulder-to-crotch": args.shoulder_to_crotch,
        "thigh": args.thigh, "wrist": args.wrist,
    }
    target.update({k: v for k, v in optional.items() if v is not None})

    print("Loading SMPL model...")
    model = load_smpl_model()

    faces = model.faces
    lbs_weights = model.lbs_weights.detach().numpy()
    torso_faces = get_torso_faces(faces, lbs_weights)
    print(f"faces total={len(faces)} torso(arm-filtered)={len(torso_faces)}")

    print(f"\nTarget measurements: {target} (height={args.height}cm)")
    print("\nFitting betas...")
    betas, result = fit_betas(model, torso_faces, target, args.height, lbs_weights=lbs_weights)

    print(f"\nConverged: {result.success}, final loss={result.fun:.2f}, "
          f"evals={result.nfev}")
    print(f"Fitted betas (first {N_FIT_BETAS}): {np.round(betas, 3)}")

    final_measured, vertices = measure_smpl_mesh(model, torso_faces, betas, args.height, lbs_weights=lbs_weights)
    print("\nFinal mesh measurements vs target:")
    for k in target:
        m = final_measured[k]
        print(f"  {k:20s}: measured={m:.1f}  target={target[k]:.1f}  "
              f"diff={abs(m - target[k]):.2f}" if m is not None else
              f"  {k:20s}: measurement failed")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.out)
    save_obj(out_path, vertices, faces)
    print(f"\nSaved avatar mesh to {out_path}")


if __name__ == "__main__":
    main()
