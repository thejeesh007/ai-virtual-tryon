"""
MySQL connection + schema for user accounts and their saved
measurements/avatar shape.

Credentials come from a .env file at the project root (gitignored -- see
.env.example for the expected shape), read here with a small manual
parser rather than pulling in a new dependency for something this simple.
Never hardcode a real password in this file or any other tracked one.
"""

import os

import mysql.connector

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def _load_env():
    env = dict(os.environ)
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    return env


_ENV = _load_env()

DB_CONFIG = {
    "host": _ENV.get("MYSQL_HOST", "localhost"),
    "port": int(_ENV.get("MYSQL_PORT", "3306")),
    "user": _ENV.get("MYSQL_USER", "root"),
    "password": _ENV.get("MYSQL_PASSWORD", ""),
    "database": _ENV.get("MYSQL_DATABASE", "dressfit"),
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# user_measurements has one row per user (PRIMARY KEY user_id, not an
# auto-increment id) -- a fresh photo scan overwrites the previous result
# via ON DUPLICATE KEY UPDATE (see auth_service.save_measurements), since
# this project only needs each user's CURRENT measurements, not a history
# of past scans.
SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(64) NOT NULL UNIQUE,
        password_hash VARCHAR(60) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_measurements (
        user_id INT PRIMARY KEY,
        height_cm FLOAT NOT NULL,
        chest FLOAT, waist FLOAT, hip FLOAT,
        shoulder_breadth FLOAT, arm_length FLOAT,
        ankle FLOAT, bicep FLOAT, calf FLOAT, forearm FLOAT,
        leg_length FLOAT, shoulder_to_crotch FLOAT, thigh FLOAT, wrist FLOAT,
        betas_json VARCHAR(512) NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
]


def init_schema():
    """Creates the tables if they don't already exist. Safe to call every
    time the API server starts."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            cur.execute(statement)
        conn.commit()
    finally:
        conn.close()
