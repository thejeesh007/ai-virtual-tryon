"""
API layer: front photo + side photo + height -> predicted measurements +
3D avatar mesh (JSON), served to the frontend viewer in frontend/public/.

Run with:
    uvicorn backend.api.main:app --reload --port 8000
(from the project root, with the venv activated)

Then open http://localhost:8000/ in a browser.
"""

import os
import sys
import time
import traceback
import uuid

import numpy as np
import requests
import trimesh
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from avatar.pipeline import generate_avatar, startup as pipeline_startup  # noqa: E402
from avatar.garment_export import (  # noqa: E402
    drape_real_garment_on_user, drape_outfit_on_user, check_fit_pose_robustness, recommend_best_size,
    GARMENTCODE_DIR,
)
from avatar.measurement_heatmap import compute_fit_heatmap, compute_fit_heatmap_by_fabric  # noqa: E402

sys.path.insert(0, os.path.join(GARMENTCODE_DIR, "dress_dataset"))
from real_garment import PRODUCTS  # noqa: E402
from generic_garment import GENERIC_PRODUCTS  # noqa: E402

MEASUREMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../measurement"))
sys.path.insert(0, MEASUREMENT_DIR)
from fabric_ease_reference import (  # noqa: E402
    FABRIC_TYPES, STANDARD_BODY_MEASUREMENTS_CM,
    BOTTOMS_FABRIC_TYPES, STANDARD_BOTTOMS_MEASUREMENTS_CM,
)
from smpl_fit import load_smpl_model, forward as smpl_forward, scale_to_height  # noqa: E402

AUTH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../auth"))
sys.path.insert(0, AUTH_DIR)
from auth_db import init_schema  # noqa: E402
import auth_service  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend", "public")
DRAPE_OUT_DIR = os.path.join(GARMENTCODE_DIR, "dress_dataset", "output", "per_user")

# In-memory session store for the phone-captures / laptop-displays handoff.
# {session_id: {"status": "waiting"|"done"|"error", "result": {...} | None,
#               "error": str | None, "created": float}}
SESSIONS = {}
SESSION_TTL_SECONDS = 30 * 60

app = FastAPI(title="AI Virtual Try-On -- Avatar Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    pipeline_startup()
    init_schema()


def _run_pipeline(front_bytes, side_bytes, height_cm):
    return generate_avatar(front_bytes, side_bytes, height_cm)


def _require_user(authorization: str = Header(default=None)):
    """FastAPI dependency: reads "Authorization: Bearer <token>", resolves
    it to a user_id via auth_service's in-memory token map, and 401s if
    missing/invalid/expired. Doesn't touch the DB itself -- resolve_token
    is a plain in-memory lookup, same cost as checking the phone-handoff
    SESSIONS dict elsewhere in this file."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization[len("Bearer "):]
    user_id = auth_service.resolve_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session -- please log in again")
    return user_id


@app.post("/api/generate")
def api_generate(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    height_cm: float = Form(...),
):
    try:
        result = _run_pipeline(front.file.read(), side.file.read(), height_cm)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# Accounts: signup/login + per-user saved measurements
# =========================================================

@app.post("/api/auth/signup")
def api_signup(username: str = Form(...), password: str = Form(...)):
    try:
        token, user_id = auth_service.signup(username, password)
    except auth_service.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"token": token, "user_id": user_id, "has_saved_scan": False}


@app.post("/api/auth/login")
def api_login(username: str = Form(...), password: str = Form(...)):
    try:
        token, user_id = auth_service.login(username, password)
    except auth_service.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    saved = auth_service.get_measurements(user_id)
    return {"token": token, "user_id": user_id, "has_saved_scan": saved is not None}


@app.get("/api/auth/scan")
def api_get_saved_scan(user_id: int = Depends(_require_user)):
    """Returns this user's most recently saved measurements + a freshly
    regenerated avatar mesh from their stored SMPL betas -- lets the
    frontend show "use your existing measurements" without re-uploading a
    photo. 404s if this user has never saved a scan (new account, or an
    existing one that always chose "take a new photo")."""
    saved = auth_service.get_measurements(user_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="No saved measurements for this account yet")

    model = load_smpl_model()
    vertices, joints = smpl_forward(model, saved["betas"])
    vertices, joints = scale_to_height(vertices, joints, saved["height_cm"])

    return {
        "measurements": saved["measurements"],
        "vertices": vertices.tolist(),
        "faces": model.faces.tolist(),
        "betas": saved["betas"],
        "height_cm": saved["height_cm"],
        "updated_at": saved["updated_at"],
    }


@app.post("/api/auth/scan")
def api_new_scan(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    height_cm: float = Form(...),
    user_id: int = Depends(_require_user),
):
    """Runs a fresh photo -> measurements + avatar pass (same pipeline as
    /api/generate) and saves the result against this logged-in account,
    replacing any previously saved scan. The uploaded photo bytes only
    ever live in memory for this one request -- generate_avatar() never
    writes the original photo to disk (only a reframed silhouette mask to
    a shared temp path, overwritten on the next request) -- per this
    project's privacy requirement of never persisting a user's photo
    anywhere, only what's predicted from it."""
    try:
        result = _run_pipeline(front.file.read(), side.file.read(), height_cm)
        auth_service.save_measurements(
            user_id, result["measurements"], result["betas"], result["height_cm"],
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class SaveScanBody(BaseModel):
    measurements: dict
    betas: list
    height_cm: float


@app.post("/api/auth/save_scan")
def api_save_scan(body: SaveScanBody, user_id: int = Depends(_require_user)):
    """Persists an ALREADY-computed scan result against this logged-in
    account, without re-running photo processing. The frontend's existing
    QR-phone and direct-upload capture flows both funnel through the same
    /api/session/{id}/submit + poll mechanism (see main.py's phone-handoff
    section below) and already end up with the full result (measurements/
    betas/height_cm) client-side once capture finishes -- this lets either
    path save that same result to the logged-in account as a follow-up
    step, instead of needing its own separate account-aware upload
    endpoint duplicating the capture UI."""
    auth_service.save_measurements(user_id, body.measurements, body.betas, body.height_cm)
    return {"status": "saved"}


# =========================================================
# Phone-captures / laptop-displays session handoff
# =========================================================

@app.post("/api/session/new")
def session_new():
    session_id = uuid.uuid4().hex[:10]
    SESSIONS[session_id] = {"status": "waiting", "result": None, "error": None, "created": time.time()}
    return {"session_id": session_id}


@app.post("/api/session/{session_id}/submit")
def session_submit(
    session_id: str,
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    height_cm: float = Form(...),
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session (it may have expired -- reload the laptop page for a new QR code)")

    try:
        result = _run_pipeline(front.file.read(), side.file.read(), height_cm)
        SESSIONS[session_id] = {"status": "done", "result": result, "error": None, "created": time.time()}
        return {"status": "done"}
    except Exception as e:
        traceback.print_exc()
        SESSIONS[session_id] = {"status": "error", "result": None, "error": str(e), "created": time.time()}
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/result")
def session_result(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session")

    # Lazy cleanup of old sessions.
    if time.time() - session["created"] > SESSION_TTL_SECONDS:
        del SESSIONS[session_id]
        raise HTTPException(status_code=404, detail="Session expired")

    if session["status"] == "error":
        return JSONResponse(status_code=500, content={"status": "error", "detail": session["error"]})

    if session["status"] == "done":
        return {"status": "done", **session["result"]}

    return {"status": "waiting"}


def _available_sizes(product_key, product, is_generic):
    """A generic product's pattern is generated lazily on first request for
    any size (see generic_garment.ensure_pattern_generated) -- every size in
    its size_guide is always available. A REAL product's pattern is NOT
    lazily generated; only whichever sizes someone has actually run through
    real_garment.py exist on disk. Without this filter, a real product with
    partial precomputation (e.g. shasmi_maxi_dress: S/M/L/XL/XXL exist, XS/
    XXXL don't) would list every size_guide entry as pickable, and picking
    an unprecomputed one would only fail once the user actually tried to
    drape it ("No precomputed pattern...") -- caught here instead, up front."""
    if is_generic:
        return list(product["size_guide"].keys())
    return [
        size for size in product["size_guide"].keys()
        if os.path.isdir(os.path.join(GARMENTCODE_DIR, "dress_dataset", "output", f"real__{product_key}_{size}"))
    ]


@app.get("/api/garments")
def list_garments():
    """Catalog for the garment picker: which real products/sizes are
    available to try on, plus the generic (non-Nobero) fabric/size
    combinations generic_garment.py can generate on demand -- these have no
    real product page, hence the "url": "" fallback."""
    return {
        key: {
            "label": p["label"],
            "url": p.get("url", ""),
            "kind": p["kind"],
            "sizes": _available_sizes(key, p, key in GENERIC_PRODUCTS),
        }
        for key, p in {**PRODUCTS, **GENERIC_PRODUCTS}.items()
    }


def _drape(product_key, size, engine, betas, height_cm, measurements, user_tag):
    """Shared by both drape endpoints below -- see _fit_heatmap_by_fabric
    for why this app ended up needing both a session-based and a stateless
    version of the same computation (the "use my saved measurements" login
    path never creates a session at all, same reason the fit-heatmap
    endpoint needed a stateless twin)."""
    if product_key not in PRODUCTS and product_key not in GENERIC_PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Unknown product: {product_key}")

    try:
        stats = drape_real_garment_on_user(
            product_key, size, betas, height_cm,
            user_tag=user_tag, out_dir=DRAPE_OUT_DIR,
            user_measurements=measurements,
            engine=engine,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    ply_path = os.path.join(PROJECT_ROOT, "third_party", "GarmentCode", stats["draped_mesh"])
    mesh = trimesh.load(ply_path, process=False)

    return {
        "vertices": np.asarray(mesh.vertices).tolist(),
        "faces": np.asarray(mesh.faces).tolist(),
        "colors": np.asarray(mesh.visual.vertex_colors)[:, :3].tolist(),  # drop alpha
        "stats": stats,
    }


@app.post("/api/session/{session_id}/drape")
def session_drape(
    session_id: str,
    product_key: str = Form(...),
    size: str = Form(...),
    engine: str = Form("simple"),
):
    """Drapes a real garment (already-precomputed pattern, per-product/size)
    onto this session's already-fitted body -- no photos needed here, reuses
    the betas from the earlier /submit call.

    engine: "simple" (default, fast, ~seconds) is the wrap+relax geometric
        approximation. "pbd" is the new real-physics solver (distance +
        bending constraints, gravity, body collision + friction) -- only
        validated on "top"-kind garments so far, and noticeably slower
        (~50s on a real polo pattern, pure Python/numpy, no GPU) since it
        runs a genuine multi-step physics simulation per request rather
        than a fixed small number of relaxation passes."""

    session = SESSIONS.get(session_id)
    if session is None or session["status"] != "done":
        raise HTTPException(status_code=400, detail="No completed avatar for this session yet")

    return _drape(
        product_key, size, engine,
        session["result"]["betas"], session["result"]["height_cm"], session["result"]["measurements"],
        user_tag=session_id,
    )


class DrapeBody(BaseModel):
    product_key: str
    size: str
    engine: str = "simple"
    measurements: dict
    betas: list
    height_cm: float


@app.post("/api/drape")
def drape_stateless(body: DrapeBody):
    """Same computation as the session-based endpoint above, but takes the
    scan result directly in the request body -- needed for the same reason
    /api/fit_heatmap_by_fabric needed a stateless twin: a logged-in user's
    "use my saved measurements" path never creates a session.

    user_tag is a fresh uuid per request (not a fixed string) -- it names
    this request's own output file, and drape_real_garment_on_user writes
    to <out_dir>/<dress_id>/<user_tag>_drape_body.obj; a shared fixed tag
    would let two concurrent stateless requests overwrite each other's
    in-progress files."""
    return _drape(
        body.product_key, body.size, body.engine,
        body.betas, body.height_cm, body.measurements,
        user_tag=uuid.uuid4().hex[:10],
    )


def _drape_outfit(top_product_key, top_size, bottom_product_key, bottom_size, engine, tuck_in,
                   betas, height_cm, measurements, user_tag):
    """Shared by both outfit-drape endpoints below, mirroring _drape's own
    session/stateless split. Drapes a top + pants-kind bottom together on
    one body and returns BOTH garment meshes plus the shared body mesh in
    one response, so the frontend can render a whole outfit in a single
    viewer scene instead of one garment at a time -- see drape_outfit_
    on_user's own docstring for what this does and does NOT simulate
    (independent drapes, not real cloth-on-cloth contact between the two
    garments)."""
    for key in (top_product_key, bottom_product_key):
        if key not in PRODUCTS and key not in GENERIC_PRODUCTS:
            raise HTTPException(status_code=404, detail=f"Unknown product: {key}")

    try:
        result = drape_outfit_on_user(
            top_product_key, top_size, bottom_product_key, bottom_size, betas, height_cm,
            user_tag=user_tag, out_dir=DRAPE_OUT_DIR,
            user_measurements=measurements, engine=engine, tuck_in=tuck_in,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    def _mesh_payload(stats):
        ply_path = os.path.join(PROJECT_ROOT, "third_party", "GarmentCode", stats["draped_mesh"])
        mesh = trimesh.load(ply_path, process=False)
        return {
            "vertices": np.asarray(mesh.vertices).tolist(),
            "faces": np.asarray(mesh.faces).tolist(),
            "colors": np.asarray(mesh.visual.vertex_colors)[:, :3].tolist(),
            "stats": stats,
        }

    # Both pieces were draped on the identical body (same betas/height/
    # pose), so either one's own body_obj_path would do -- drape_outfit_
    # on_user already picks the top's. Same meters->centimeters scaling
    # _drape_pose_check already needs for the same reason (export_for_
    # draping's own meters-scale .obj vs. the engines' internal cm
    # convention -- see that function's comment).
    body_mesh = trimesh.load(result["body_obj_path"], process=False)
    body_vertices_cm = np.asarray(body_mesh.vertices) * 100.0

    return {
        "top": _mesh_payload(result["top"]),
        "bottom": _mesh_payload(result["bottom"]),
        "body": {
            "vertices": body_vertices_cm.tolist(),
            "faces": np.asarray(body_mesh.faces).tolist(),
        },
        "tucked_in": result["tucked_in"],
    }


@app.post("/api/session/{session_id}/drape_outfit")
def session_drape_outfit(
    session_id: str,
    top_product_key: str = Form(...),
    top_size: str = Form(...),
    bottom_product_key: str = Form(...),
    bottom_size: str = Form(...),
    engine: str = Form("simple"),
    tuck_in: bool = Form(False),
):
    """Drapes a top + pants-kind bottom together on this session's already-
    fitted body -- no photos needed here, reuses the betas from the earlier
    /submit call. See _drape_outfit's docstring."""
    session = SESSIONS.get(session_id)
    if session is None or session["status"] != "done":
        raise HTTPException(status_code=400, detail="No completed avatar for this session yet")

    return _drape_outfit(
        top_product_key, top_size, bottom_product_key, bottom_size, engine, tuck_in,
        session["result"]["betas"], session["result"]["height_cm"], session["result"]["measurements"],
        user_tag=session_id,
    )


class DrapeOutfitBody(BaseModel):
    top_product_key: str
    top_size: str
    bottom_product_key: str
    bottom_size: str
    engine: str = "simple"
    tuck_in: bool = False
    measurements: dict
    betas: list
    height_cm: float


@app.post("/api/drape_outfit")
def drape_outfit_stateless(body: DrapeOutfitBody):
    """Same computation as the session-based endpoint above, but takes the
    scan result directly in the request body -- same reason /api/drape
    needed a stateless twin (the "use my saved measurements" login path
    never creates a session)."""
    return _drape_outfit(
        body.top_product_key, body.top_size, body.bottom_product_key, body.bottom_size,
        body.engine, body.tuck_in,
        body.betas, body.height_cm, body.measurements,
        user_tag=uuid.uuid4().hex[:10],
    )


def _drape_pose_check(product_key, size, engine, betas, height_cm, measurements, user_tag):
    """Shared by both pose-check endpoints below, mirroring _drape's own
    session/stateless split. Separate from _drape itself (rather than a
    flag on it) since this is an opt-in, much slower (~3x the real-GPU-sim
    cost for a top's 3 poses, ~5x for pants' 5 -- two independent leg axes,
    knee-bend and leg-spread, each with its own 2 check angles) follow-up
    call -- see check_fit_pose_robustness's own docstring for why it's
    never run as part of the normal drape.

    Unlike an earlier version of this endpoint, each pose's full mesh
    (avatar body + heatmap-colored garment) is loaded and returned here, not
    just its fit_label as text -- the whole point of checking several poses
    is to actually SEE each one draped, the same way the single-pose result
    already can be, not just read a number."""
    if product_key not in PRODUCTS and product_key not in GENERIC_PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Unknown product: {product_key}")
    if engine != "warp":
        raise HTTPException(
            status_code=400,
            detail="Pose robustness check requires engine=\"warp\" (the only engine validated for pants "
                   "as well as tops, and the one this feature's own pose-angle calibration was done against).",
        )

    try:
        result = check_fit_pose_robustness(
            product_key, size, betas, height_cm,
            user_tag=user_tag, out_dir=DRAPE_OUT_DIR,
            user_measurements=measurements,
            engine=engine,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    poses_out = []
    for stats in result["poses"]:
        ply_path = os.path.join(PROJECT_ROOT, "third_party", "GarmentCode", stats["draped_mesh"])
        garment_mesh = trimesh.load(ply_path, process=False)
        body_mesh = trimesh.load(stats["body_obj_path"], process=False)
        # body_obj_path is export_for_draping()'s own meters-scale export
        # (see its docstring); the garment mesh comes back in GarmentCode's
        # internal centimeter convention (their load_body() multiplies our
        # meters input by 100). Scale the body to match here so the two
        # don't land 100x apart in the viewer -- the single-pose endpoint
        # never had this problem since it only ever returns the garment.
        body_vertices_cm = np.asarray(body_mesh.vertices) * 100.0

        poses_out.append({
            "pose": stats["pose"],
            "description": stats["pose_description"],
            "fit_label": stats["fit_label"],
            "achieved_ease_cm": stats["achieved_ease_cm"],
            "expected_ease_cm": stats["expected_ease_cm"],
            "gap_cm_mean": round(float(stats["gap_cm_mean"]), 2),
            "garment": {
                "vertices": np.asarray(garment_mesh.vertices).tolist(),
                "faces": np.asarray(garment_mesh.faces).tolist(),
                "colors": np.asarray(garment_mesh.visual.vertex_colors)[:, :3].tolist(),
            },
            "body": {
                "vertices": body_vertices_cm.tolist(),
                "faces": np.asarray(body_mesh.faces).tolist(),
            },
        })

    return {
        "poses": poses_out,
        "agreement": result["agreement"],
        "confidence": result["confidence"],
        "good_fit_confidence": result["good_fit_confidence"],
    }


@app.post("/api/session/{session_id}/drape_pose_check")
def session_drape_pose_check(
    session_id: str,
    product_key: str = Form(...),
    size: str = Form(...),
):
    """Re-drapes the same garment+size at two modest arm-angle variants
    (see garment_export.check_fit_pose_robustness) to check whether the fit
    verdict holds up under realistic pose variation. Opt-in: ~3x the
    latency of a normal drape, so the frontend only calls this from an
    explicit "check fit robustness" action, never automatically."""
    session = SESSIONS.get(session_id)
    if session is None or session["status"] != "done":
        raise HTTPException(status_code=400, detail="No completed avatar for this session yet")

    return _drape_pose_check(
        product_key, size, "warp",
        session["result"]["betas"], session["result"]["height_cm"], session["result"]["measurements"],
        user_tag=session_id,
    )


class DrapePoseCheckBody(BaseModel):
    product_key: str
    size: str
    measurements: dict
    betas: list
    height_cm: float


@app.post("/api/drape_pose_check")
def drape_pose_check_stateless(body: DrapePoseCheckBody):
    """Stateless twin of session_drape_pose_check, same reason /api/drape
    needed one (the "use my saved measurements" login path never creates a
    session)."""
    return _drape_pose_check(
        body.product_key, body.size, "warp",
        body.betas, body.height_cm, body.measurements,
        user_tag=uuid.uuid4().hex[:10],
    )


def _recommend_size(product_key, measurements):
    """Shared by both recommend-size endpoints below, mirroring _drape's
    own session/stateless split. Pure arithmetic (see garment_export.
    recommend_best_size) -- no body mesh, no physics sim, so unlike every
    other garment endpoint here this one responds in milliseconds and
    never needs the WSL2/GPU bridge at all."""
    if product_key not in PRODUCTS and product_key not in GENERIC_PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Unknown product: {product_key}")

    try:
        return recommend_best_size(product_key, measurements)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/recommend_size")
def session_recommend_size(session_id: str, product_key: str = Form(...)):
    """Which size of this product fits this session's user best, across
    every size of the product at once -- see garment_export.
    recommend_best_size. Instant (pure arithmetic), so this is meant to run
    automatically whenever a product is picked, not gated behind an opt-in
    button the way the much slower physics-based checks are."""
    session = SESSIONS.get(session_id)
    if session is None or session["status"] != "done":
        raise HTTPException(status_code=400, detail="No completed avatar for this session yet")

    return _recommend_size(product_key, session["result"]["measurements"])


class RecommendSizeBody(BaseModel):
    product_key: str
    measurements: dict


@app.post("/api/recommend_size")
def recommend_size_stateless(body: RecommendSizeBody):
    """Stateless twin of session_recommend_size, same reason /api/drape
    needed one (the "use my saved measurements" login path never creates a
    session)."""
    return _recommend_size(body.product_key, body.measurements)


@app.get("/api/fabrics")
def list_fabrics():
    """Catalog for the garment-category + fabric-type + size dropdowns used
    by the generic (no-specific-product) fit heatmap -- lets someone check
    "if I bought a size M in this fabric, would it fit me" without picking
    a real garment listing first. Returns both "top" and "bottom" catalogs
    since the fabric list and size chart are entirely different per
    category (a shirt fabric and a pants fabric don't share ease fields)."""
    def _catalog(fabric_dict, chart):
        return {
            "fabrics": {
                key: {"label": f["label"], "category": f["category"], "fit_style": f["fit_style"]}
                for key, f in fabric_dict.items()
            },
            "sizes": list(chart.keys()),
        }

    return {
        "top": _catalog(FABRIC_TYPES, STANDARD_BODY_MEASUREMENTS_CM),
        "bottom": _catalog(BOTTOMS_FABRIC_TYPES, STANDARD_BOTTOMS_MEASUREMENTS_CM),
    }


def _fit_heatmap_by_fabric(fabric_key, size, garment_kind, measurements, betas, height_cm):
    """Shared by both fit-heatmap-by-fabric endpoints below -- validates the
    fabric/size/garment_kind combination and runs the computation. Split out
    because this app ended up with two different ways an avatar can be on
    screen (the ephemeral QR/session capture flow, and a logged-in user's
    saved scan fetched via /api/auth/scan) and both need the exact same
    validation + computation, just sourcing measurements/betas/height_cm
    differently."""
    if garment_kind == "bottom":
        fabric_catalog, standard_chart = BOTTOMS_FABRIC_TYPES, STANDARD_BOTTOMS_MEASUREMENTS_CM
    elif garment_kind == "top":
        fabric_catalog, standard_chart = FABRIC_TYPES, STANDARD_BODY_MEASUREMENTS_CM
    else:
        raise HTTPException(status_code=400, detail=f"Unknown garment_kind: {garment_kind}")

    if fabric_key not in fabric_catalog:
        raise HTTPException(status_code=404, detail=f"Unknown fabric type: {fabric_key}")

    if size not in standard_chart:
        raise HTTPException(status_code=400, detail=f"Unknown size: {size}")

    try:
        return compute_fit_heatmap_by_fabric(fabric_key, size, measurements, betas, height_cm, garment_kind=garment_kind)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/fit_heatmap_by_fabric")
def session_fit_heatmap_by_fabric(
    session_id: str,
    fabric_key: str = Form(...),
    size: str = Form(...),
    garment_kind: str = Form("top"),
):
    """Same measurement-driven fit heatmap as /fit_heatmap, but built from a
    garment-category + fabric-type + size selection and the standardized
    size chart instead of a specific real product's own hand-measured spec."""

    session = SESSIONS.get(session_id)
    if session is None or session["status"] != "done":
        raise HTTPException(status_code=400, detail="No completed avatar for this session yet")

    return _fit_heatmap_by_fabric(
        fabric_key, size, garment_kind,
        session["result"]["measurements"], session["result"]["betas"], session["result"]["height_cm"],
    )


class FitHeatmapByFabricBody(BaseModel):
    fabric_key: str
    size: str
    garment_kind: str = "top"
    measurements: dict
    betas: list
    height_cm: float


@app.post("/api/fit_heatmap_by_fabric")
def fit_heatmap_by_fabric_stateless(body: FitHeatmapByFabricBody):
    """Same computation as the session-based endpoint above, but takes the
    scan result directly in the request body instead of looking it up by
    session_id. Needed because a logged-in user's "use my saved
    measurements" path (GET /api/auth/scan) never creates a session at
    all -- it skips the QR/photo capture flow entirely -- so there's no
    session_id for that avatar to look the heatmap request up by. This
    endpoint works off whatever avatar is currently on screen regardless
    of which path put it there."""
    return _fit_heatmap_by_fabric(
        body.fabric_key, body.size, body.garment_kind, body.measurements, body.betas, body.height_cm,
    )


@app.post("/api/session/{session_id}/fit_heatmap")
def session_fit_heatmap(session_id: str, product_key: str = Form(...), size: str = Form(...)):
    """Measurement-driven fit heatmap: colors THIS SESSION'S OWN AVATAR BODY
    (not a draped garment mesh) by comparing a real garment's published
    measurements against what its fabric type should need for this user's
    own predicted measurements. Independent of the 3D garment-draping
    pipeline above -- no garment pattern, no wrap/relax physics, just
    measurement arithmetic mapped onto the avatar's existing vertices."""

    session = SESSIONS.get(session_id)
    if session is None or session["status"] != "done":
        raise HTTPException(status_code=400, detail="No completed avatar for this session yet")

    if product_key not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Unknown product: {product_key}")

    if size not in PRODUCTS[product_key]["size_guide"]:
        raise HTTPException(status_code=400, detail=f"Unknown size {size} for {product_key}")

    betas = session["result"]["betas"]
    height_cm = session["result"]["height_cm"]
    measurements = session["result"]["measurements"]

    try:
        result = compute_fit_heatmap(
            product_key, PRODUCTS[product_key], size, measurements, betas, height_cm,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get("/api/ngrok-url")
def ngrok_url():
    """Best-effort lookup of the current public ngrok URL via ngrok's local
    admin API (runs on the same machine as this server), so the laptop page
    doesn't need the URL pasted in manually for the QR code."""
    try:
        resp = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=1.5)
        tunnels = resp.json().get("tunnels", [])
        for t in tunnels:
            if t.get("public_url", "").startswith("https://"):
                return {"url": t["public_url"]}
    except Exception:
        pass
    return {"url": None}


# =========================================================
# Static pages
# =========================================================

@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/capture")
def capture_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "capture.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
