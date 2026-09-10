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

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from avatar.pipeline import generate_avatar, startup as pipeline_startup  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend", "public")

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


def _run_pipeline(front_bytes, side_bytes, height_cm):
    return generate_avatar(front_bytes, side_bytes, height_cm)


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
