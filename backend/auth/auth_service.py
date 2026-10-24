"""
Signup/login/token issuance and per-user measurement persistence, backed
by the MySQL schema in auth_db.py.

Passwords are bcrypt-hashed (auto-salted per user) -- the plaintext
password is never stored, logged, or kept around past the hashing/
verification call itself. Session tokens are random opaque strings kept
in an in-memory dict mapping token -> user_id, the same pattern
backend/api/main.py already uses for its phone/laptop-handoff SESSIONS
dict: this project has no need for a token that verifies itself without
a server round-trip (that's what a JWT buys you), so a signed token would
be more machinery than the problem calls for.
"""

import secrets
import time
import json

import bcrypt

from auth_db import get_connection

# Python side uses hyphenated keys (matching the BodyM measurement columns
# used everywhere else in this project); SQL column names can't contain
# hyphens, hence the mapping.
MEASUREMENT_FIELD_MAP = {
    "chest": "chest",
    "waist": "waist",
    "hip": "hip",
    "shoulder-breadth": "shoulder_breadth",
    "arm-length": "arm_length",
    "ankle": "ankle",
    "bicep": "bicep",
    "calf": "calf",
    "forearm": "forearm",
    "leg-length": "leg_length",
    "shoulder-to-crotch": "shoulder_to_crotch",
    "thigh": "thigh",
    "wrist": "wrist",
}

AUTH_TOKENS = {}  # token -> {"user_id": int, "created": float}
TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60  # a week


class AuthError(Exception):
    """Raised for any expected auth failure (bad password, duplicate
    username, ...) -- the API layer turns this into a 400, not a 500."""


def _issue_token(user_id):
    token = secrets.token_hex(24)
    AUTH_TOKENS[token] = {"user_id": user_id, "created": time.time()}
    return token


def resolve_token(token):
    """Returns the user_id for a valid, non-expired token, else None."""
    entry = AUTH_TOKENS.get(token)
    if entry is None:
        return None
    if time.time() - entry["created"] > TOKEN_TTL_SECONDS:
        del AUTH_TOKENS[token]
        return None
    return entry["user_id"]


def signup(username, password):
    username = (username or "").strip()
    if not username or not password:
        raise AuthError("Username and password are required")
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")

    conn = get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash),
            )
        except Exception as e:
            if "Duplicate entry" in str(e):
                raise AuthError("That username is already taken") from e
            raise
        conn.commit()
        user_id = cur.lastrowid
    finally:
        conn.close()

    return _issue_token(user_id), user_id


def login(username, password):
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, password_hash FROM users WHERE username = %s",
            ((username or "").strip(),),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None or not bcrypt.checkpw(
        (password or "").encode("utf-8"), row["password_hash"].encode("ascii")
    ):
        raise AuthError("Invalid username or password")

    return _issue_token(row["id"]), row["id"]


def save_measurements(user_id, measurements, betas, height_cm):
    """measurements: the 13-key dict from avatar.pipeline.generate_avatar();
    betas: the fitted SMPL shape params for the same scan. Overwrites this
    user's previously saved result (ON DUPLICATE KEY UPDATE) -- this
    project only needs each user's CURRENT measurements, not a history of
    every past scan."""
    columns = ["user_id", "height_cm", "betas_json"]
    values = [user_id, height_cm, json.dumps(list(betas))]
    for measurement_key, column in MEASUREMENT_FIELD_MAP.items():
        columns.append(column)
        values.append(measurements.get(measurement_key))

    placeholders = ", ".join(["%s"] * len(columns))
    update_clause = ", ".join(f"{c} = VALUES({c})" for c in columns if c != "user_id")

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO user_measurements ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {update_clause}",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def get_measurements(user_id):
    """Returns {"measurements": {...}, "betas": [...], "height_cm": ...,
    "updated_at": iso-string} for this user's most recently saved scan, or
    None if they've never saved one -- the caller (the API layer) uses
    None to mean "ask this user whether to reuse an existing scan" isn't
    even an option yet, they must take a photo."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM user_measurements WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    measurements = {
        measurement_key: row[column]
        for measurement_key, column in MEASUREMENT_FIELD_MAP.items()
        if row[column] is not None
    }
    return {
        "measurements": measurements,
        "betas": json.loads(row["betas_json"]),
        "height_cm": row["height_cm"],
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }
