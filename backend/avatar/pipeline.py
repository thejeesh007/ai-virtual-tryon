"""
Module 2 orchestration: front photo + side photo + height -> predicted
measurements + a personalized SMPL avatar mesh. Wires together the
existing scripts in backend/measurement/ (segmentation, HGB prediction,
SMPL fitting) behind one function the API layer calls per request.

Everything expensive that doesn't depend on the request (SMPL model,
HGB model, MediaPipe segmenter, BodyM reference framing stats) is loaded
once at import time, not per-call.
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np

MEASUREMENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../measurement")
)
sys.path.insert(0, MEASUREMENT_DIR)

from bodym_validate_real import (  # noqa: E402
    compute_reference_framing, segment_silhouette, reframe_to_bodym,
    predict_baseline, CANVAS_W, CANVAS_H,
)
from smpl_fit import (  # noqa: E402
    load_smpl_model, forward, scale_to_height, fit_betas, N_FIT_BETAS,
)
from mesh_utils import get_torso_faces  # noqa: E402

AVATAR_TARGET_COLS = ["chest", "waist", "hip", "shoulder-breadth", "arm-length"]

_state = {}


def startup():
    """Call once when the API server starts."""
    print("[avatar.pipeline] loading reference framing stats...")
    _state["ref"] = compute_reference_framing()

    print("[avatar.pipeline] loading MediaPipe selfie segmentation...")
    _state["segmenter"] = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=1
    )

    print("[avatar.pipeline] loading SMPL model...")
    model = load_smpl_model()
    _state["smpl_model"] = model
    faces = model.faces
    lbs_weights = model.lbs_weights.detach().numpy()
    _state["torso_faces"] = get_torso_faces(faces, lbs_weights)
    _state["faces"] = faces

    print("[avatar.pipeline] ready.")


def _decode_image(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image -- is it a valid image file?")
    return img


def _mask_to_array(mask):
    """mask (uint8 720x960 image) -> float array in [0,1] for width-profile
    extraction, matching what extract_width_profile expects from a file."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def generate_avatar(front_bytes, side_bytes, height_cm):
    """
    Returns:
        {
          "measurements": {col: value, ...},   # all 13 BodyM measurements
          "vertices": [[x,y,z], ...],           # fitted avatar mesh, cm
          "faces": [[i,j,k], ...],
        }
    """
    if "ref" not in _state:
        raise RuntimeError("pipeline.startup() was not called")

    front_img = _decode_image(front_bytes)
    side_img = _decode_image(side_bytes)

    segmenter = _state["segmenter"]
    ref = _state["ref"]

    front_mask = reframe_to_bodym(segment_silhouette(front_img, segmenter), ref["front"])
    side_mask = reframe_to_bodym(segment_silhouette(side_img, segmenter), ref["side"])

    # predict_baseline reads mask files from disk (extract_width_profile
    # takes a path) -- write the reframed masks to a temp location once
    # per request rather than changing that function's signature.
    tmp_dir = os.path.join(MEASUREMENT_DIR, "bodym_output", "api_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    front_path = os.path.join(tmp_dir, "front_mask.png")
    side_path = os.path.join(tmp_dir, "side_mask.png")
    cv2.imwrite(front_path, front_mask)
    cv2.imwrite(side_path, side_mask)

    measurements = predict_baseline(front_path, side_path, height_cm, model_name="hgb")
    if measurements is None:
        raise RuntimeError("HGB baseline model not found -- run bodym_baseline.py first")

    avatar_target = {k: measurements[k] for k in AVATAR_TARGET_COLS}

    model = _state["smpl_model"]
    torso_faces = _state["torso_faces"]
    faces = _state["faces"]

    betas, _ = fit_betas(model, torso_faces, avatar_target, height_cm, n_betas=N_FIT_BETAS)
    vertices, _ = forward(model, betas)
    vertices, _ = scale_to_height(vertices, vertices, height_cm)

    return {
        "measurements": {k: float(v) for k, v in measurements.items()},
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
    }
