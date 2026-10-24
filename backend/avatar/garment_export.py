"""
Export our fitted SMPL avatar in the format GarmentCode's drape_dress()
expects, and call it: real garment pattern (precomputed once per product+
size by the teammate's real_garment.py) + our user's own personalized
body, instead of GarmentCode's generic bundled reference body.

Two conventions their drape engine assumes, confirmed by inspecting
third_party/GarmentCode/assets/bodies/m_smpl_average_A40.obj and running
their own example end to end:
  - Units: meters (their load_body() multiplies by 100 internally). Our
    own avatar viewer/SMPL fitting works in centimeters, so this needs a
    separate meters-scale export, not the same file the web UI uses.
  - Feet at y=0 exactly (their reference body's y_min is 0.0), since the
    garment pattern mesh's own vertical placement assumes that. Our
    zero-pose SMPL avatar's y_min is a few dozen cm negative by default,
    so this must be corrected or the wrap step's 30cm close-threshold can
    miss the body's silhouette to a badly wrapped result.
  - Pose: uses smpl_fit.default_body_pose() -- the exact same pose the main
    avatar (shown in the web viewer) is generated with, not a
    garment-drape-only variant. An earlier version of this function used a
    DIFFERENT pose than the displayed avatar (first a plain zero/rest pose
    vs. a rotated one, later both unrotated but full-T-pose while
    GarmentCode's own patterns assume a much less extreme arm angle) --
    any mismatch between what's draped and what's shown produces visible
    seams-in-the-wrong-place bugs, so this must always stay identical to
    forward()'s own default, not redefine its own rotation here.
"""

import os
import sys

import numpy as np
import torch
import trimesh

# measurement_heatmap.py is a sibling module in this same backend/avatar/
# directory, but bare (non-package-qualified) imports here only resolve
# when something has already put the module's OWN directory on sys.path --
# smpl_fit's import below relies on the same trick, via avatar.pipeline's
# side effect of adding backend/measurement/ (confirmed by testing
# garment_export.py's import in isolation: it fails without pipeline
# imported first). backend/avatar/ itself is never added by anything else,
# so this file has to do it for its own sibling import to resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smpl_fit import load_smpl_model, default_body_pose, ARM_LOWER_ANGLE_DEG
from measurement_heatmap import _get_avatar_geometry, _classify_regions, _measure_shoulder_seam_width

GARMENTCODE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../third_party/GarmentCode")
)


def export_for_draping(betas, height_cm, out_path, arm_angle_deg=None, knee_angle_deg=None,
                        hip_spread_deg=None):
    """betas: the same fitted shape parameters smpl_fit.fit_betas() finds
    for this person (from their predicted chest/waist/hip/shoulder-
    breadth/arm-length). Writes a meters-scale, feet-at-floor .obj at
    out_path, in the exact same pose as the avatar shown in the web viewer,
    ready to pass as drape_dress(body_obj=out_path).

    arm_angle_deg: leave as None for the normal single-pose flow -- that
    keeps this identical to the avatar's own default pose, per the module
    docstring's warning above about the two needing to always match. It
    exists only for the multi-pose fit check (check_fit_pose_robustness's
    top-kind pose check), to export a SECOND body at a modest pose variation
    alongside the primary one; the avatar shown in the viewer is never
    re-rendered at this angle.

    knee_angle_deg: same idea, for the pants-kind leg-bend pose check
    instead -- a top's own pose check varies the arm, a pants' varies the
    knee, never both on the same body (see check_fit_pose_robustness).

    hip_spread_deg: same idea again, for the pants-kind leg-SPREAD pose
    check (a second, independent pants-kind variant alongside
    knee_angle_deg -- never combined with it on the same body either)."""

    model = load_smpl_model()

    betas_t = torch.zeros(1, model.num_betas, dtype=torch.float32)
    betas_t[0, :len(betas)] = torch.tensor(betas, dtype=torch.float32)

    pose = default_body_pose(
        arm_angle_deg=arm_angle_deg if arm_angle_deg is not None else ARM_LOWER_ANGLE_DEG,
        knee_angle_deg=knee_angle_deg if knee_angle_deg is not None else 0.0,
        hip_spread_deg=hip_spread_deg if hip_spread_deg is not None else 0.0,
    )
    with torch.no_grad():
        output = model(betas=betas_t, body_pose=pose, global_orient=torch.zeros(1, 3))

    vertices = output.vertices[0].numpy().astype(np.float64)

    # Scale to the real height (in meters, matching GarmentCode's convention),
    # then shift so feet sit exactly at y=0.
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    scale = (height_cm / 100.0) / (y_max - y_min)
    vertices = vertices * scale
    vertices[:, 1] -= vertices[:, 1].min()

    faces = model.faces
    with open(out_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    return out_path


def _compute_achieved_ease(kind, spec, user_chest, user_waist, user_hip):
    """Pure arithmetic: this garment's real measurement minus this user's
    real measurement -- the ACHIEVED ease. No mesh, no physics simulation
    needed, so this is usable instantly for every size of a product (see
    recommend_best_size below), not just the one currently being physically
    draped. Extracted out of drape_real_garment_on_user's own inline logic
    so the two can never drift apart on what "achieved ease" means for the
    same kind -- the exact bug class that caused the original "S and XXL
    both show Good fit" issue (two copies of fit logic disagreeing).

    Returns (avg_ease_cm, region_ease) where region_ease is a {part: ease_cm}
    dict matching drape_real_garment_on_user's own convention (chest for a
    top -- "shoulder" is added separately by the caller, since that needs
    a mesh measurement this pure function doesn't have; hip+waist for
    pants); None for "dress" (no per-region recolor support there)."""
    if kind == "top":
        body_chest_ref = spec.get("to_fit_chest", user_chest)
        avg_ease_cm = spec["chest"] - body_chest_ref
        return avg_ease_cm, {"chest": avg_ease_cm}
    elif kind == "pants":
        hip_ease_cm = spec["hip"] - user_hip
        wb_mult = spec["hip"] / user_hip
        achieved_waist_cm = wb_mult * user_waist
        waist_ease_achieved = achieved_waist_cm - user_waist
        avg_ease_cm = (hip_ease_cm + waist_ease_achieved) / 2
        return avg_ease_cm, {"hip": hip_ease_cm, "waist": waist_ease_achieved}
    elif kind == "dress":
        avg_ease_cm = spec["bust"] - user_chest
        return avg_ease_cm, None
    else:
        raise ValueError(kind)


def _compute_fit_label(kind, product_key, is_generic, product, avg_ease_cm,
                        PRODUCT_FABRIC_TYPE, FABRIC_TYPES, BOTTOMS_FABRIC_TYPES):
    """Pure arithmetic: compares the ACHIEVED ease (from _compute_achieved_ease
    above) against this fabric's EXPECTED ease to produce the actual
    cross-size Snug/Good fit/Loose verdict -- see the original fit_label
    comment (preserved in git history) for the full reasoning. Extracted
    for the same reason as _compute_achieved_ease: recommend_best_size
    needs this exact logic, byte-for-byte, for every size of a product at
    once, with zero risk of it silently diverging from the one real drape
    already uses.

    Returns (tolerance_cm, expected_ease_cm, diff_cm, fit_label, fit_confidence).
    fit_confidence (High/Medium/Low) is a SEPARATE signal from fit_label,
    not blended into it -- same reasoning this project already applies to
    keeping the heatmap color and fit_label apart (two different questions
    shouldn't share one number).

    A fabric's ease_by_part_cm entry (e.g. ease_by_part["chest"]) can be
    EITHER a plain number (legacy: a symmetric Good-fit band of ideal +-
    tolerance_cm, exactly as before) OR a {"min","ideal","max"} dict (an
    explicit, possibly ASYMMETRIC band straight from a richer source chart
    -- e.g. jeans_stretch_denim's waist is min=-1/ideal=0/max=2, not
    symmetric; maxi_rayon_challis's hip is min=12/ideal=25/max=60, since a
    flared skirt tolerates vastly more ease than a fitted bust does, which
    a single +-tolerance number can't express at all). Both forms resolve
    to the same (lo, hi, ideal) triple below via _resolve_band, so
    fit_label/fit_confidence handle either uniformly -- verified
    algebraically to reproduce the old symmetric-only formula exactly for
    every existing product (same fit_label, same fit_confidence, not just
    "close enough")."""
    if is_generic:
        tolerance_cm = product.get("tolerance_cm", 3.0)
        # Generic product keys are "generic_top_<fabric_key>" /
        # "generic_bottom_<fabric_key>" -- strip the prefix to get straight
        # back to the fabric_ease_reference.py entry.
        fabric_key = product_key.replace("generic_top_", "").replace("generic_bottom_", "")
    else:
        fabric_key = PRODUCT_FABRIC_TYPE.get(product_key)
        fabric_table_for_tolerance = BOTTOMS_FABRIC_TYPES if kind == "pants" else FABRIC_TYPES
        tolerance_cm = fabric_table_for_tolerance.get(fabric_key, {}).get("tolerance_cm", 3.0)

    fabric_table = BOTTOMS_FABRIC_TYPES if kind == "pants" else FABRIC_TYPES
    ease_by_part = fabric_table.get(fabric_key, {}).get("ease_by_part_cm", {})

    def _resolve_band(part_value, fallback_ideal):
        """-> (lo, hi, ideal) for one body part's ease_by_part_cm entry,
        whichever of the two shapes it's stored as."""
        if isinstance(part_value, dict):
            return part_value["min"], part_value["max"], part_value["ideal"]
        ideal = part_value if part_value is not None else fallback_ideal
        return ideal - tolerance_cm, ideal + tolerance_cm, ideal

    if kind == "pants":
        hip_val = ease_by_part.get("hip")
        waist_val = ease_by_part.get("waist", hip_val)
        hip_lo, hip_hi, hip_ideal = _resolve_band(hip_val, 4.0)
        waist_lo, waist_hi, waist_ideal = _resolve_band(waist_val, 4.0)
        lo = (hip_lo + waist_lo) / 2
        hi = (hip_hi + waist_hi) / 2
        expected_ease_cm = (hip_ideal + waist_ideal) / 2
    else:
        lo, hi, expected_ease_cm = _resolve_band(ease_by_part.get("chest"), 4.0)

    half_width = (hi - lo) / 2 or 1e-9  # guard a degenerate zero-width band
    mid = (hi + lo) / 2
    diff_cm = avg_ease_cm - expected_ease_cm

    # Distance from avg_ease_cm to the NEAREST band boundary (lo or hi), as a
    # fraction of the band's own half-width. Positive and growing = deep
    # inside the Good-fit band (confidently Good fit); negative/positive and
    # growing past an edge = clearly Snug/Loose, not a marginal call. Near
    # zero (either sign) = right on top of a boundary, where a few mm of
    # measurement error could flip the verdict.
    if avg_ease_cm < lo:
        fit_label = "Snug"
        margin = (avg_ease_cm - lo) / half_width
    elif avg_ease_cm > hi:
        fit_label = "Loose"
        margin = (avg_ease_cm - hi) / half_width
    else:
        fit_label = "Good fit"
        margin = (half_width - abs(avg_ease_cm - mid)) / half_width

    if abs(margin) >= 0.5:
        fit_confidence = "High"
    elif abs(margin) >= 0.15:
        fit_confidence = "Medium"
    else:
        fit_confidence = "Low"

    return tolerance_cm, expected_ease_cm, diff_cm, fit_label, fit_confidence


# Which real dress products are confirmed stable on the real Warp simulator,
# and across what user HEIGHT range -- a dress's skirt (and, as it turns
# out, its bodice) is cut to one fixed absolute length, calibrated against
# the reference body's own ~175cm proportions, not rescaled per the actual
# user's height. On a notably shorter body the torso itself is
# proportionally shorter too, so the bodice panel (cut for a ~175cm torso
# length) gets vertically crushed rather than lying flat. Calibrated
# against shasmi_maxi_dress M via real Warp sims, visually checked (not
# just by the numbers):
#   175cm: clean, 0 body-cloth intersections.
#   165cm: stats alone look worse (88 intersections) but the actual render
#          (front AND back) is still clean -- the extra "intersections" are
#          hidden inside the skirt's own folds where it pools slightly more
#          at the floor (expected: a dress cut for 175cm naturally puddles
#          more on someone 10cm shorter, same as in real life).
#   150cm: a REAL, VISIBLE failure -- the bodice is visibly bunched/crushed
#          at the bust seam, 402 body-cloth intersections, and the
#          simulator's own classifier flags is_fail=True (the only test in
#          this whole dress investigation where that happened).
# No data between 150-165cm or above 175cm yet -- the range below is
# deliberately conservative (set above the confirmed-clean 165cm, well
# above the confirmed-broken 150cm), not a precisely pinned boundary.
DRESS_WARP_HEIGHT_RANGE_CM = {
    "shasmi_maxi_dress": (160, 200),
}


def drape_real_garment_on_user(product_key, size, betas, height_cm, user_tag, out_dir,
                                user_measurements=None, engine="simple", arm_angle_deg=None,
                                knee_angle_deg=None, hip_spread_deg=None):
    """Reuses the teammate's already-precomputed pattern/ease/material for
    (product_key, size) -- that part is garment-specific, not user-specific
    -- and drapes it onto THIS user's own body instead of the generic
    reference body.

    user_measurements: this user's own predicted chest/waist/hip (from
    pipeline.generate_avatar()'s measurements dict). Ease used to always be
    computed against GarmentCode's fixed reference body (bust=100cm,
    hips=100cm, waist=80cm) regardless of who was actually wearing the
    garment -- fine for someone close to those proportions, silently wrong
    for anyone far from them (a person needing XXL got the same "ease"
    number as someone needing S, even though a fixed-size pattern fits each
    of them completely differently). Falls back to the reference body's
    own measurements if not given, e.g. for the CLI/reference-body-only
    path in real_garment.py.

    engine: "simple" (default) uses simple_drape.py's wrap+relax geometric
        approximation, as before. "pbd" uses this project's own hand-rolled
        real-physics solver (pbd_drape.py) instead. "warp" uses the actual
        NVIDIA Warp-based GPU simulator GarmentCodeData itself was built
        with (warp_drape.py), shelling out to a WSL2 conda env since that's
        the only place on this machine the GPU-enabled Warp fork is built --
        by far the best fit quality of the three, but only works on a
        machine with that WSL2 environment set up. "warp" is validated on
        "top" (polo, sweatshirt), "pants" (cargo pants), and "dress"
        (shasmi_maxi_dress -- a flared A-line maxi: 0 body-cloth
        intersections, clean drape front and back, though the long flowing
        skirt's own slow-settling fabric means the simulator's internal
        equilibrium check doesn't always report "converged" even when the
        visual result is fine -- confirmed by actually looking at the
        render, not just the stats, same discipline as every other engine
        validation here). A dress with a dramatically different silhouette
        (bodycon, heavily pleated, etc.) hasn't specifically been tried --
        worth a quick visual check the same way, not assumed safe just
        because "dress" as a kind now passes this gate. "pbd" is only validated on "top"
        (the polo) -- pants would need its own anchor/collision tuning
        (its anchor_height_frac=0.80 default assumes a shirt hung from the
        shoulders, not pants anchored at the waist) that hasn't been done,
        since "warp" is the better engine anyway wherever it's available.
        Raises ValueError for an unvalidated engine/kind combination rather
        than silently running an untested one.

    arm_angle_deg: leave as None for the normal single-pose flow (the
        avatar's own default pose, per export_for_draping's docstring). Set
        only by check_fit_pose_robustness() below, to re-drape the SAME
        garment+size on the SAME body shape at a slightly different arm
        angle -- a robustness check, not an alternate recommendation, so
        this is never exposed directly to the main single-drape API path.

    knee_angle_deg: same idea as arm_angle_deg, for pants-kind garments'
        own leg-BEND variation pose check instead of the arm -- also only
        ever set by check_fit_pose_robustness(), never together with
        arm_angle_deg at once on the same call (a top's check varies the
        arm; a pants' varies the leg).

    hip_spread_deg: a second, independent pants-kind leg variation --
        spreading both legs apart sideways instead of bending the knees.
        Also only ever set by check_fit_pose_robustness(), and never
        together with knee_angle_deg on the same call (each is its own
        separate pose-check variant, not combined into one pose).
    """

    sys.path.insert(0, os.path.join(GARMENTCODE_DIR, "dress_dataset"))
    from simple_drape import drape_dress, load_body, color_from_gap
    from pbd_drape import drape_dress_pbd
    from warp_drape import drape_dress_warp
    from fabrics import FABRICS
    from real_garment import PRODUCTS
    from generic_garment import GENERIC_PRODUCTS, ensure_pattern_generated, WARP_UNSTABLE_PRODUCTS

    measurement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "measurement")
    if measurement_dir not in sys.path:
        sys.path.insert(0, measurement_dir)
    from fabric_ease_reference import PRODUCT_FABRIC_TYPE, FABRIC_TYPES, BOTTOMS_FABRIC_TYPES

    is_generic = product_key in GENERIC_PRODUCTS
    product = GENERIC_PRODUCTS[product_key] if is_generic else PRODUCTS[product_key]

    if engine == "pbd" and product["kind"] != "top":
        raise ValueError(
            f"The PBD physics engine is only validated for \"top\"-kind garments so far "
            f"(got kind={product['kind']!r} for {product_key}); use engine=\"simple\" or \"warp\" for this one."
        )
    if engine == "warp" and product["kind"] not in ("top", "pants", "dress"):
        raise ValueError(
            f"The real Warp GPU simulator engine is only validated for \"top\"-, \"pants\"-, and "
            f"\"dress\"-kind garments so far (got kind={product['kind']!r} for {product_key}); "
            f"use engine=\"simple\" for this one."
        )
    if engine == "warp" and product_key in WARP_UNSTABLE_PRODUCTS:
        raise ValueError(
            f"{product_key} is confirmed unstable under the real Warp simulator -- see "
            f"generic_garment.py's WARP_UNSTABLE_PRODUCTS comment for which specific failure this "
            f"product had; use engine=\"simple\" for this one."
        )
    if engine == "warp" and product_key in DRESS_WARP_HEIGHT_RANGE_CM:
        min_h, max_h = DRESS_WARP_HEIGHT_RANGE_CM[product_key]
        if not (min_h <= height_cm <= max_h):
            raise ValueError(
                f"{product_key} on the real Warp simulator is only confirmed stable for heights "
                f"{min_h}-{max_h}cm (got {height_cm}cm) -- see DRESS_WARP_HEIGHT_RANGE_CM's comment "
                f"for the real tests behind this range (at 150cm the bodice visibly bunched/crushed and "
                f"the simulator itself flagged is_fail=True); use engine=\"simple\" for this height instead."
            )

    dress_id = f"{product_key}_{size}" if is_generic else f"real__{product_key}_{size}"
    precomputed_dir = os.path.join(GARMENTCODE_DIR, "dress_dataset", "output", dress_id)

    if is_generic and not os.path.isdir(precomputed_dir):
        # Lazy pattern generation: this (fabric, size) combination may never
        # have been requested before -- see generic_garment.py's module
        # docstring for why these aren't all precomputed upfront.
        ensure_pattern_generated(product_key, size)

    if not os.path.isdir(precomputed_dir):
        raise FileNotFoundError(
            f"No precomputed pattern for {dress_id} -- run "
            f"`python real_garment.py {product_key} {size}` in "
            f"third_party/GarmentCode/dress_dataset/ first."
        )

    # Read the pattern spec from the shared precomputed dir, but write this
    # user's own drape result to a separate per-user dir -- otherwise every
    # user's run would overwrite the one shared/reusable pattern output.
    spec_path = next(
        os.path.join(precomputed_dir, p) for p in os.listdir(precomputed_dir)
        if p.endswith("_specification.json")
    )

    # Nested under user_tag, not just dress_id: the BODY file's own name
    # already included user_tag (so that part was always safe), but the
    # drape engine's OWN output .ply is named only after dress_id -- with
    # everyone sharing one dress_id/ folder, two calls for the same
    # garment+size (two different users, or check_fit_pose_robustness's own
    # 3 back-to-back poses for the SAME user) would overwrite each other's
    # .ply on disk. Confirmed live: the pose-check feature's "Variant A" and
    # "Variant B" buttons were rendering the identical garment mesh, because
    # by the time the API layer loaded each pose's .ply (after all 3 drapes
    # had already run), only the LAST one written still existed on disk.
    user_sample_dir = os.path.join(out_dir, dress_id, user_tag)
    os.makedirs(user_sample_dir, exist_ok=True)
    body_obj_path = os.path.join(user_sample_dir, f"{user_tag}_drape_body.obj")
    export_for_draping(betas, height_cm, body_obj_path, arm_angle_deg=arm_angle_deg,
                        knee_angle_deg=knee_angle_deg, hip_spread_deg=hip_spread_deg)

    from assets.bodies.body_params import BodyParameters
    body = BodyParameters(str(os.path.join(GARMENTCODE_DIR, "assets", "bodies", "m_smpl_average_A40.yaml")))
    spec = product["size_guide"][size]

    skip_wrap_below_y = None

    # Prefer this specific user's own predicted measurements over the fixed
    # reference body's -- see the user_measurements docstring above. Our own
    # pipeline's measurement dict uses "chest" for what a dress's real size
    # guide calls "bust"; same anatomical measurement, different naming
    # convention between the two data sources.
    um = user_measurements or {}
    user_chest = um.get("chest", body["bust"])
    user_waist = um.get("waist", body["waist"])
    user_hip = um.get("hip", body["hips"])

    # Per-region ease: the SAME real chest/shoulder (or hip/waist, for
    # pants) ease numbers already computed for the measurement heatmap (see
    # measurement_heatmap.py), used for TWO separate things below: (1)
    # engine="pbd" feeds this straight into its own wrap-to-body step so a
    # top's drape stops applying one blended (chest-only) ease everywhere,
    # including the shoulder/collar, where it was silently wrong; (2) EVERY
    # engine's OUTPUT heatmap gets recolored against this per-region ease
    # after draping (see the post-processing block below) -- a single
    # avg_ease_cm-centered band was too narrow to capture how gap-to-body
    # naturally varies across a garment's own surface (tight at the torso,
    # looser at a sleeve hem even on a well-fitting shirt), so the drape
    # heatmap looked like one flat color instead of the same kind of
    # region-by-region red/green/blue picture the measurement heatmap
    # already shows. Built for every engine now, not just "pbd".
    region_ease = None
    body_regions = None

    avg_ease_cm, region_ease = _compute_achieved_ease(product["kind"], spec, user_chest, user_waist, user_hip)

    if product["kind"] in ("top", "pants"):
        body_vertices, body_joints, body_lbs_weights, _ = _get_avatar_geometry(betas, height_cm)
        body_regions, _ = _classify_regions(body_vertices, body_joints, body_lbs_weights)
        if product["kind"] == "top" and "shoulder" in spec:
            # Mesh-measured seam-to-seam width, not the raw joint-based
            # shoulder-breadth -- see _measure_shoulder_seam_width's own
            # docstring for why those are different measurement
            # conventions and only this one is comparable to a real
            # garment's published "shoulder" spec.
            user_shoulder = _measure_shoulder_seam_width(body_vertices, body_joints, body_lbs_weights)
            if user_shoulder is not None:
                region_ease["shoulder"] = spec["shoulder"] - user_shoulder
    elif product["kind"] == "dress":
        # A single ease/wrap-standoff can't fit both the tight bodice and
        # the flared skirt below it (see simple_drape.wrap_to_body) -- skip
        # wrapping entirely below the waist so the skirt drapes from
        # gravity + stretch/bend instead of snapping onto the legs. The
        # reference body's own _waist_level is in ITS coordinate frame
        # (fixed 178.1cm height); export_for_draping() uses the exact same
        # floor-anchored, zero-pose convention just scaled to THIS user's
        # real height, so the same waist-height fraction carries over.
        waist_frac = body["_waist_level"] / body["height"]
        skip_wrap_below_y = waist_frac * height_cm

    material = FABRICS[product["fabric_standin"]]["material"]

    # The heatmap's "good fit" (green) band needs to be centered on THIS
    # fabric's own intended ease, not a fixed absolute gap-to-body distance
    # -- otherwise a negative-ease stretch fabric (meant to sit SMALLER than
    # the body) shows as tight/red almost everywhere even when draped
    # exactly as intended, since gap-to-body is never actually negative.
    # tolerance_cm is the fabric-category-specific "how far from the target
    # ease still counts as a good fit" width already established for the
    # measurement heatmap (fabric_ease_reference.py's CATEGORY_TOLERANCE_CM)
    # -- reused here so both heatmaps agree on what "good fit" means for the
    # same fabric, not just what's already the same window this garment's
    # OWN fabric was designed to sit in. Generic products carry their own
    # tolerance_cm directly (see generic_garment.py); real Nobero products
    # look theirs up via PRODUCT_FABRIC_TYPE's mapping to the same table.
    tolerance_cm, expected_ease_cm, diff_cm, fit_label, fit_confidence = _compute_fit_label(
        product["kind"], product_key, is_generic, product, avg_ease_cm,
        PRODUCT_FABRIC_TYPE, FABRIC_TYPES, BOTTOMS_FABRIC_TYPES,
    )

    drape_fn = {"pbd": drape_dress_pbd, "warp": drape_dress_warp}.get(engine, drape_dress)
    # region_ease/body_regions are only passed INTO the drape call for
    # "pbd" -- that's the one engine whose own wrap-to-body step can use a
    # per-vertex ease array (see pbd_drape.drape_dress_pbd's docstring).
    # Every engine still gets the region-aware RECOLOR below regardless.
    extra_kwargs = {"region_ease": region_ease, "body_regions": body_regions} if engine == "pbd" else {}
    stats = drape_fn(
        dress_id,
        sample_dir=user_sample_dir,
        spec_path=spec_path,
        body_obj=body_obj_path,
        avg_ease_cm=avg_ease_cm,
        material=material,
        skip_wrap_below_y=skip_wrap_below_y,
        heatmap_tight_cm=avg_ease_cm - tolerance_cm,
        heatmap_loose_cm=avg_ease_cm + tolerance_cm,
        verbose=True,
        **extra_kwargs,
    )

    # Recolor the OUTPUT heatmap against the PERCENTILE spread of gap-to-
    # body THIS drape actually produced (15th percentile -> red, 85th ->
    # blue). This does NOT try to be comparable across sizes -- three
    # earlier attempts at an absolute, ease-informed band all failed that
    # goal for a real physical reason: real cloth folds, so most of a
    # garment's surface rests close to the body via folds regardless of
    # overall size, with the extra room from a bigger size concentrated in
    # a smaller "billow" tail (sleeve ends, hem) rather than spread evenly
    # -- confirmed directly (nobero_polo XXXL, ~29cm of ease, still came out
    # 82.6% "tight"-colored under an ease-centered band, because the BULK
    # of its own gap distribution sat well below where that band's math put
    # the tight cutoff). A percentile band sidesteps that: it's purely
    # about showing which PART of this one garment is relatively tighter or
    # looser than the rest of the SAME garment, which it does reliably.
    # Cross-size fit comparison is handled separately, by fit_label below,
    # using gap_cm_mean -- confirmed monotonic with size in every test this
    # session (nobero_polo: S=1.13cm -> XL=1.79cm -> XXXL=2.40cm).
    draped_ply_path = os.path.join(GARMENTCODE_DIR, stats["draped_mesh"])
    draped_mesh = trimesh.load(draped_ply_path, process=False)
    draped_verts = np.asarray(draped_mesh.vertices)

    body_mesh, proximity, _ = load_body(body_obj_path)
    _, gap_cm, _ = proximity.on_surface(draped_verts)

    tight_cm = float(np.percentile(gap_cm, 15))
    loose_cm = float(np.percentile(gap_cm, 85))
    if loose_cm - tight_cm < 1.0:
        # A near-uniform gap (an unusually even drape) would otherwise
        # collapse color_from_gap's band to near-zero width and blow every
        # tiny noise fluctuation up into a full red-to-blue swing.
        loose_cm = tight_cm + 1.0
    colors = color_from_gap(gap_cm, tight_cm=tight_cm, loose_cm=loose_cm)

    draped_mesh.visual.vertex_colors = colors
    draped_mesh.export(draped_ply_path)

    # fit_label: the actual cross-size "does this fit" signal, kept
    # separate from the heatmap color for the reason on the recolor block
    # above (gap_cm_mean is a simulated, fold-sensitive geometric quantity
    # -- real cloth folding defeats using it as a direct fit verdict, the
    # same reason it defeated the heatmap color). Computed once, up front,
    # via _compute_fit_label (shared with recommend_best_size below) --
    # just applying the already-computed verdict here.
    stats["fit_label"] = fit_label
    stats["fit_confidence"] = fit_confidence
    stats["achieved_ease_cm"] = round(float(avg_ease_cm), 2)
    stats["expected_ease_cm"] = round(float(expected_ease_cm), 2)
    # This user's own posed body for THIS drape -- already an absolute path
    # (out_dir is), so the API layer can load it directly, same as it
    # already does for stats["draped_mesh"]. Needed so a caller can show the
    # actual posed avatar next to the draped garment, not just the garment
    # alone -- see check_fit_pose_robustness below, whose whole point is
    # comparing the body in several different poses.
    stats["body_obj_path"] = body_obj_path

    return stats


# How far from ARM_LOWER_ANGLE_DEG a "modest, realistic capture variability"
# pose check should go. Calibrated empirically against nobero_polo M on the
# Warp engine, visually checked (not just by the numbers -- see this
# project's own heatmap-coloring history for why that matters):
#   +-8deg  (-42deg/-58deg): gap-to-body stats close to the default pose's
#            own, sleeve still tracks the arm, visually near-identical.
#   +-15deg (-35deg/-65deg): already visibly starting to separate from the
#            arm (the early "fan" shape the module docstring warns about) --
#            pct_loose_gt_8cm jumped ~24x over the default pose even though
#            nothing actually failed. That's the pattern's own arm-angle
#            assumption being strained, not a meaningful fit-robustness
#            signal, so it's deliberately NOT used here.
POSE_CHECK_OFFSET_DEG = 8

# The pants-kind analogue of POSE_CHECK_OFFSET_DEG above -- a slight
# standing-stance knee bend (NOT the 90deg hip+knee seated pose separately
# confirmed to break the pattern entirely: "Failed to achieve static
# equilibrium", 129 body-cloth intersections, even after widening the
# pattern's own design.pants.width to its max 1.5 -- see the sitting-pose
# investigation). Calibrated against nobero_cargo_pants size 32 on the real
# Warp engine, visually checked:
#   0deg  : the existing straight-leg standing pose (the proven baseline).
#   15deg : clean, 0 body-cloth intersections, normal equilibrium, drapes
#           all the way to the ankle with no bunching.
#   30deg : real strain begins -- 77 intersections -- though it still
#           reaches equilibrium (not a hard failure like the seated pose).
# The two check angles below stay inside the clean zone, same reasoning as
# the arm case's 15-degree rejection.
KNEE_CHECK_ANGLES_DEG = (8.0, 15.0)

# A SECOND, independent pants-kind leg variation -- spreading both legs
# apart sideways into a wider stance, instead of bending the knees forward.
# Calibrated against nobero_cargo_pants size 32 on the real Warp engine,
# visually checked (see smpl_fit.py's default_body_pose docstring for the
# hip_spread_deg parameter, and this project's own scratchpad calibration
# scripts for the raw numbers):
#   10deg/leg : clean, 0 body-cloth intersections, visually confirmed fine.
#   20deg/leg : real strain begins -- 44 intersections -- still visually
#               fine though, not a hard failure.
#   35deg/leg : confirmed, VISIBLE failure -- the inseam sags into a droopy
#               pouch and the shins come out exposed; the pattern's fixed
#               inseam length can't reach both widely-spread legs.
# Narrower margin than the knee-bend case (strain begins at 20deg here vs
# 30deg there), so unlike KNEE_CHECK_ANGLES_DEG (which goes up to the clean
# boundary itself, 15deg), both check angles here stay comfortably inside
# the confirmed-clean 10deg boundary rather than reaching right up to it.
SPREAD_CHECK_ANGLES_DEG = (5.0, 10.0)


def check_fit_pose_robustness(product_key, size, betas, height_cm, user_tag, out_dir,
                               user_measurements=None, engine="warp", primary_stats=None):
    """Re-drapes the SAME garment+size on the SAME body SHAPE at two modest
    pose variants to check whether the fit verdict holds up under the kind
    of small pose variation a real capture naturally has, instead of
    trusting one single static pose. Opt-in and separate from the main
    drape call (each variant is a full ~20-25s real GPU sim via WSL2, so
    this isn't run automatically).

    WHICH body part varies depends on the garment's own kind -- varying the
    arm for pants, or the leg for a top, would be testing something that
    doesn't actually affect that garment's fit at all:
      - "top": the arm, +-POSE_CHECK_OFFSET_DEG around ARM_LOWER_ANGLE_DEG
        (unchanged from the original single-axis version of this feature).
      - "pants": the leg, along TWO independent axes -- bent forward by
        each of KNEE_CHECK_ANGLES_DEG, AND spread apart sideways by each of
        SPREAD_CHECK_ANGLES_DEG (0/0 for "default", since that's the
        already-proven straight-leg stance) -- pants are what actually
        responds to either leg variation; a top does not. The two axes are
        never combined into one pose (see export_for_draping/drape_real_
        garment_on_user's own docstrings) -- each is its own separate
        variant, so pants get 5 total poses (default + 2 knee-bend + 2
        leg-spread) where a top gets 3 (default + 2 arm-angle).

    primary_stats: pass the already-computed default-pose result (from a
    prior drape_real_garment_on_user call) to reuse it as one of the poses
    instead of re-draping it again. If not given, this drapes the default
    pose itself first.

    Returns {"poses": [...one dict per pose (3 for a top, 5 for pants),
    each the full drape_real_garment_on_user stats (so a caller can load
    that pose's own draped garment mesh AND its own posed body, not just
    read its fit_label as text) plus "pose"/"pose_description"],
    "agreement": "<n>/<total>", "confidence": "High" | "Medium" | "Low"}.
    Deliberately doesn't try to collapse the confidence into a single
    blended number -- for a capstone demo, showing the actual verdicts side
    by side is more honest than a single opaque confidence percentage
    would be.
    """
    sys.path.insert(0, os.path.join(GARMENTCODE_DIR, "dress_dataset"))
    from generic_garment import GENERIC_PRODUCTS
    from real_garment import PRODUCTS
    is_generic = product_key in GENERIC_PRODUCTS
    product = GENERIC_PRODUCTS[product_key] if is_generic else PRODUCTS[product_key]

    if product["kind"] == "pants":
        angles = [
            ("default", {"knee_angle_deg": 0.0}, "Straight legs (default stance)"),
            ("variant_a", {"knee_angle_deg": KNEE_CHECK_ANGLES_DEG[0]},
             f"Knees bent {KNEE_CHECK_ANGLES_DEG[0]:.0f}°"),
            ("variant_b", {"knee_angle_deg": KNEE_CHECK_ANGLES_DEG[1]},
             f"Knees bent {KNEE_CHECK_ANGLES_DEG[1]:.0f}°"),
            ("variant_c", {"hip_spread_deg": SPREAD_CHECK_ANGLES_DEG[0]},
             f"Legs spread {SPREAD_CHECK_ANGLES_DEG[0]:.0f}°"),
            ("variant_d", {"hip_spread_deg": SPREAD_CHECK_ANGLES_DEG[1]},
             f"Legs spread {SPREAD_CHECK_ANGLES_DEG[1]:.0f}°"),
        ]
    else:
        angles = [
            ("default", {"arm_angle_deg": ARM_LOWER_ANGLE_DEG}, f"Arms at {ARM_LOWER_ANGLE_DEG:.0f}° (default)"),
            ("variant_a", {"arm_angle_deg": ARM_LOWER_ANGLE_DEG - POSE_CHECK_OFFSET_DEG},
             f"Arms at {ARM_LOWER_ANGLE_DEG - POSE_CHECK_OFFSET_DEG:.0f}°"),
            ("variant_b", {"arm_angle_deg": ARM_LOWER_ANGLE_DEG + POSE_CHECK_OFFSET_DEG},
             f"Arms at {ARM_LOWER_ANGLE_DEG + POSE_CHECK_OFFSET_DEG:.0f}°"),
        ]

    poses = []
    for pose_name, pose_kwargs, description in angles:
        if pose_name == "default" and primary_stats is not None:
            stats = primary_stats
        else:
            stats = drape_real_garment_on_user(
                product_key, size, betas, height_cm,
                user_tag=f"{user_tag}_{pose_name}",
                out_dir=out_dir,
                user_measurements=user_measurements,
                engine=engine,
                **pose_kwargs,
            )
        # Keep the FULL stats (not a trimmed summary) -- "draped_mesh" and
        # "body_obj_path" are what let a caller actually render this pose's
        # avatar + garment, not just report its fit_label as text. Tag on
        # which pose this is since that's not otherwise in drape_real_
        # garment_on_user's own return value.
        stats["pose"] = pose_name
        stats["pose_description"] = description
        poses.append(stats)

    labels = [p["fit_label"] for p in poses]
    total = len(labels)
    agree_count = max(labels.count(l) for l in set(labels))
    agreement = f"{agree_count}/{total}"
    # Same High/Medium/Low bands as the original 3-pose top/pants check,
    # generalized so they still apply now that pants runs 5 poses instead
    # of 3: full agreement is High, a majority (but not all) is Medium,
    # anything else (no label with more than half the votes) is Low.
    if agree_count == total:
        confidence = "High"
    elif agree_count / total > 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Good-fit confidence estimates whether this size is actually a good
    # fit, not how certain we are that any verdict (including Snug/Loose)
    # is stable. It is a heuristic, not a calibrated probability.
    fit_confidence_points = {"High": 100, "Medium": 70, "Low": 40}
    default_is_good = poses[0].get("fit_label") == "Good fit"
    fit_score = fit_confidence_points.get(poses[0].get("fit_confidence"), 40) if default_is_good else 0
    good_fit_pose_score = round(sum(p.get("fit_label") == "Good fit" for p in poses) / total * 100)
    good_fit_confidence = round(fit_score * 0.6 + good_fit_pose_score * 0.4)

    return {
        "poses": poses,
        "agreement": agreement,
        "confidence": confidence,
        "good_fit_confidence": good_fit_confidence,
    }


def _tuck_in_top_mesh(mesh, waist_height_cm):
    """Approximates a "tucked in" look by discarding the top's own hem
    geometry below the waist line, instead of simulating real cloth-on-
    cloth contact between the shirt and the pants' waistband -- neither
    this project's PBD solver nor warp_drape.py's Warp integration drapes
    two garments against EACH OTHER at once (each only ever sees the body),
    so an actual tucked fold/bulge isn't attempted here. This only removes
    the visual clash of a shirt hem poking out past/overlapping the pants'
    own waist in an "untucked" render -- not a simulation of tucked fabric.

    waist_height_cm must be in the SAME coordinate frame as mesh's own
    vertices -- the draped garment .ply comes back in GarmentCode's
    internal centimeter, feet-at-y=0 convention (see drape_real_garment_
    on_user's recolor block above), so the caller computes the waist
    height the same way the dress "skip_wrap_below_y" logic already does,
    in that same cm frame.

    Drops whole TRIANGLES that lie entirely below the line (vs. slicing
    the mesh exactly at the plane) -- cheap, and avoids trimesh's plane-
    capping needing to invent new boundary vertices that would have no
    vertex-color data of their own."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    below = vertices[:, 1] < waist_height_cm
    face_fully_below = np.all(below[faces], axis=1)
    keep_face_idx = np.where(~face_fully_below)[0]
    if len(keep_face_idx) == 0 or len(keep_face_idx) == len(faces):
        # Nothing to trim (e.g. a cropped top with no hem below the waist
        # line at all) -- return the mesh unchanged rather than risk an
        # empty/degenerate submesh.
        return mesh
    return mesh.submesh([keep_face_idx], append=True)


def drape_outfit_on_user(top_key, top_size, bottom_key, bottom_size, betas, height_cm, user_tag,
                          out_dir, user_measurements=None, engine="simple", tuck_in=False):
    """Drapes a top AND a pants-kind bottom together on the SAME body, for
    the "whole outfit in one view" feature -- NOT a real cloth-on-cloth
    interaction simulation between the two garments. Each piece is draped
    completely independently (two full calls to drape_real_garment_on_user,
    same betas/height/pose so they land on the identical body) and simply
    composited into one result; the top has no awareness the bottom exists
    and vice versa. This is a real, deliberate simplification -- a shirt
    resting ON TOP of pants' own waistband (as opposed to each independently
    draping onto the bare body underneath) isn't modeled, which is exactly
    why tuck_in below is a cheap post-hoc mesh trim rather than a physical
    effect.

    top_key/bottom_key: top_key's product must be kind="top" and
    bottom_key's must be kind="pants" -- raises ValueError otherwise (e.g.
    trying to pair two tops, or a dress in either slot: a dress already
    covers both halves of the body, so it doesn't have an "other half" to
    pair with).

    tuck_in: False (default) renders both garments exactly as each engine
    draped them independently -- for a top whose hem naturally falls below
    the waist, this can visibly overlap/clash with the pants' own
    waistband in the render, since neither drape knows the other garment
    exists. True trims the top's own mesh at the body's waist height (see
    _tuck_in_top_mesh) to approximate a tucked-in look instead.

    Returns {"top": <drape_real_garment_on_user stats, mesh file already
    trimmed if tuck_in>, "bottom": <same, untouched>, "body_obj_path":
    <shared body, from the top's own drape -- byte-identical to the
    bottom's since both used the same betas/height/pose>, "tucked_in": bool}.
    """
    sys.path.insert(0, os.path.join(GARMENTCODE_DIR, "dress_dataset"))
    from real_garment import PRODUCTS
    from generic_garment import GENERIC_PRODUCTS

    top_product = GENERIC_PRODUCTS.get(top_key) or PRODUCTS.get(top_key)
    bottom_product = GENERIC_PRODUCTS.get(bottom_key) or PRODUCTS.get(bottom_key)
    if top_product is None:
        raise ValueError(f"Unknown product: {top_key}")
    if bottom_product is None:
        raise ValueError(f"Unknown product: {bottom_key}")
    if top_product["kind"] != "top":
        raise ValueError(
            f"{top_key} is kind={top_product['kind']!r}, not \"top\" -- the outfit's top slot needs a "
            f"top-kind garment (a dress already covers both halves, so it can't be paired with a bottom)."
        )
    if bottom_product["kind"] != "pants":
        raise ValueError(
            f"{bottom_key} is kind={bottom_product['kind']!r}, not \"pants\" -- the outfit's bottom slot "
            f"needs a pants-kind garment (a dress already covers both halves, so it can't be paired with a top)."
        )

    # Two independent drapes, same body (same betas/height/pose -- neither
    # call is given an arm/knee/hip-spread override, so both get the exact
    # same default_body_pose() result): the whole point of two separate
    # user_tag subdirectories is the same reason check_fit_pose_robustness's
    # poses each get their own (see drape_real_garment_on_user's own
    # comment on this) -- two calls sharing one dress_id/user_tag folder
    # would overwrite each other's output .ply.
    top_stats = drape_real_garment_on_user(
        top_key, top_size, betas, height_cm,
        user_tag=f"{user_tag}_top", out_dir=out_dir,
        user_measurements=user_measurements, engine=engine,
    )
    bottom_stats = drape_real_garment_on_user(
        bottom_key, bottom_size, betas, height_cm,
        user_tag=f"{user_tag}_bottom", out_dir=out_dir,
        user_measurements=user_measurements, engine=engine,
    )

    if tuck_in:
        from assets.bodies.body_params import BodyParameters
        body = BodyParameters(str(os.path.join(GARMENTCODE_DIR, "assets", "bodies", "m_smpl_average_A40.yaml")))
        # Same waist-height-fraction-of-height computation dress kind
        # already uses for skip_wrap_below_y above -- carries over from the
        # reference body's own coordinate frame to THIS user's real height
        # the same way.
        waist_frac = body["_waist_level"] / body["height"]
        waist_height_cm = waist_frac * height_cm

        top_ply_path = os.path.join(GARMENTCODE_DIR, top_stats["draped_mesh"])
        top_mesh = trimesh.load(top_ply_path, process=False)
        tucked_mesh = _tuck_in_top_mesh(top_mesh, waist_height_cm)
        tucked_mesh.export(top_ply_path)

    return {
        "top": top_stats,
        "bottom": bottom_stats,
        "body_obj_path": top_stats["body_obj_path"],
        "tucked_in": tuck_in,
    }


def recommend_best_size(product_key, user_measurements):
    """Which size of this product would actually fit this user best --
    instantly, across EVERY available size at once, not just whichever one
    they happen to have selected. This is pure arithmetic (_compute_achieved_
    ease + _compute_fit_label, the exact same functions drape_real_garment_
    on_user itself uses for its own fit_label) -- no body mesh, no physics
    simulation, so it costs milliseconds rather than the ~20-60s a real
    drape takes, and can safely be run against all 6-ish sizes of a product
    on every request without the user waiting at all.

    user_measurements: this user's own predicted chest/waist/hip, same
    convention as drape_real_garment_on_user's own parameter.

    Returns {"sizes": [...one dict per size, each {"size", "fit_label",
    "achieved_ease_cm", "expected_ease_cm", "diff_cm"}, in the product's own
    size_guide order], "recommended_size": the single best one}.

    "Best" picks, in order: (1) the first "Good fit" size in the product's
    own size order (smallest-to-largest, matching how a real size guide
    lists them) -- ties broken toward the smaller size, same reasoning a
    human reading a size chart would use, rather than picking whichever
    "Good fit" has the smallest numeric diff_cm (several sizes can equally
    round-trip into "Good fit" once they're within tolerance_cm, and the
    smallest one is the more useful, more specific recommendation); (2) if
    NONE are "Good fit" (true for a very unusual body relative to this
    product's whole size range), the size whose diff_cm has the smallest
    absolute value -- closest to fitting even though technically outside
    tolerance, which is still the most useful single answer to give."""
    sys.path.insert(0, os.path.join(GARMENTCODE_DIR, "dress_dataset"))
    from generic_garment import GENERIC_PRODUCTS
    from real_garment import PRODUCTS

    measurement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "measurement")
    if measurement_dir not in sys.path:
        sys.path.insert(0, measurement_dir)
    from fabric_ease_reference import PRODUCT_FABRIC_TYPE, FABRIC_TYPES, BOTTOMS_FABRIC_TYPES

    is_generic = product_key in GENERIC_PRODUCTS
    product = GENERIC_PRODUCTS[product_key] if is_generic else PRODUCTS[product_key]

    um = user_measurements or {}
    # No reference-body fallback here (unlike drape_real_garment_on_user) --
    # a size recommendation with no real measurements to go on isn't a
    # meaningful answer, so this requires them explicitly rather than
    # silently recommending against a generic body that isn't this user.
    if not all(k in um for k in ("chest", "waist", "hip")):
        raise ValueError("recommend_best_size needs chest, waist, and hip in user_measurements")
    user_chest, user_waist, user_hip = um["chest"], um["waist"], um["hip"]

    sizes_out = []
    for size, spec in product["size_guide"].items():
        avg_ease_cm, _ = _compute_achieved_ease(product["kind"], spec, user_chest, user_waist, user_hip)
        tolerance_cm, expected_ease_cm, diff_cm, fit_label, fit_confidence = _compute_fit_label(
            product["kind"], product_key, is_generic, product, avg_ease_cm,
            PRODUCT_FABRIC_TYPE, FABRIC_TYPES, BOTTOMS_FABRIC_TYPES,
        )
        sizes_out.append({
            "size": size,
            "fit_label": fit_label,
            "fit_confidence": fit_confidence,
            "achieved_ease_cm": round(float(avg_ease_cm), 2),
            "expected_ease_cm": round(float(expected_ease_cm), 2),
            "diff_cm": round(float(diff_cm), 2),
        })

    good_fits = [s for s in sizes_out if s["fit_label"] == "Good fit"]
    if good_fits:
        # Prefer the most CONFIDENTLY good fit among them (deepest inside
        # the tolerance band), not just the first one in size order --
        # several sizes can all clear the "Good fit" cutoff while differing
        # a lot in how comfortably they clear it, and fit_confidence is
        # exactly the signal that tells them apart. max() keeps the first
        # occurrence on a tie, so equal-confidence sizes still break toward
        # the smaller/first-listed one, same as before this change.
        confidence_rank = {"High": 2, "Medium": 1, "Low": 0}
        recommended = max(good_fits, key=lambda s: confidence_rank[s["fit_confidence"]])["size"]
    else:
        recommended = min(sizes_out, key=lambda s: abs(s["diff_cm"]))["size"]

    return {"sizes": sizes_out, "recommended_size": recommended}
