"""Thin HTTP client the Streamlit pages use to talk to the MediShield API."""

from __future__ import annotations

import io
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


_EMBEDDED_PORT = 8765
_embedded_proc: subprocess.Popen | None = None
_embedded_lock = threading.Lock()
_EMBEDDED_LOG = Path(tempfile.gettempdir()) / "medishield_api.log"


def _start_embedded_backend() -> None:
    """Launch the API child process (non-blocking) if it isn't running."""
    global _embedded_proc
    with _embedded_lock:
        if _embedded_proc is not None and _embedded_proc.poll() is None:
            return
        env = dict(os.environ)
        env.setdefault("SECRET_KEY", secrets.token_urlsafe(48))
        log = open(_EMBEDDED_LOG, "ab")
        _embedded_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "backend.main:app",
                "--host", "127.0.0.1", "--port", str(_EMBEDDED_PORT),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def warm_up() -> None:
    """Begin starting the embedded API early, e.g. while the user logs in."""
    if not os.environ.get("API_URL"):
        _start_embedded_backend()


def _ensure_embedded_backend() -> str:
    """Return the embedded API's URL, waiting until it is ready.

    This lets a single Streamlit service host the whole app. Without
    DATABASE_URL the API uses a local SQLite file, so accounts reset
    whenever the service restarts.
    """
    url = f"http://127.0.0.1:{_EMBEDDED_PORT}"
    _start_embedded_backend()

    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if requests.get(url + "/health", timeout=1).ok:
                return url
        except requests.RequestException:
            pass
        if _embedded_proc.poll() is not None:
            try:
                tail = _EMBEDDED_LOG.read_text(errors="replace")[-800:]
            except OSError:
                tail = ""
            raise ApiError(
                "The MediShield server failed to start. "
                f"Details: {tail.strip() or 'no log output'}"
            )
        time.sleep(0.3)

    raise ApiError("The MediShield server is taking too long to start.")


def _base_url() -> str:
    if not os.environ.get("API_URL"):
        return _ensure_embedded_backend()
    url = os.environ["API_URL"].rstrip("/")
    # Render's private-network value is a bare host:port.
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


class ApiError(Exception):
    """An error response from the API, with a user-readable message."""


def _request(method: str, path: str, *, auth: bool = True, **kwargs):
    headers = {}
    token = st.session_state.get("token")
    if auth and token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Generous read timeout: free-tier cold starts and training are slow.
        response = requests.request(
            method,
            _base_url() + path,
            headers=headers,
            timeout=(10, 600),
            **kwargs,
        )
    except requests.RequestException as exc:
        raise ApiError(
            "Could not reach the MediShield server. It may be waking up; "
            f"try again in a moment. ({exc.__class__.__name__})"
        ) from exc

    if response.status_code == 401 and auth:
        st.session_state.clear()
        st.rerun()

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        raise ApiError(str(detail))

    return response


# --------------------------------------------------------------------- auth

def login(username: str, password: str) -> dict:
    return _request(
        "POST", "/auth/login", auth=False,
        json={"username": username, "password": password},
    ).json()


def signup(username: str, password: str) -> dict:
    return _request(
        "POST", "/auth/signup", auth=False,
        json={"username": username, "password": password},
    ).json()


# ------------------------------------------------------------------ catalog

def catalog() -> dict:
    if "catalog" not in st.session_state:
        st.session_state["catalog"] = _request(
            "GET", "/catalog/hyperparameters"
        ).json()
    return st.session_state["catalog"]


# ----------------------------------------------------------------- datasets

def sheet_names(file_bytes: bytes, filename: str) -> list[str]:
    return _request(
        "POST", "/datasets/sheets",
        files={"file": (filename, file_bytes)},
    ).json()["sheets"]


def upload_dataset(
    file_bytes: bytes, filename: str, sheet_name: str | None = None
) -> dict:
    return _request(
        "POST", "/datasets",
        files={"file": (filename, file_bytes)},
        data={"sheet_name": sheet_name} if sheet_name else None,
    ).json()


def download_dataset(dataset_id: int) -> pd.DataFrame:
    text = _request("GET", f"/datasets/{dataset_id}/csv").text
    return pd.read_csv(io.StringIO(text))


def clean_dataset(dataset_id: int, options: dict) -> dict:
    return _request(
        "POST", f"/datasets/{dataset_id}/clean", json=options
    ).json()


def split_check(
    dataset_id: int, target_column: str, test_size: float, random_state: int
) -> dict:
    return _request(
        "POST", f"/datasets/{dataset_id}/split-check",
        json={
            "target_column": target_column,
            "test_size": test_size,
            "random_state": random_state,
        },
    ).json()


# ------------------------------------------------------------------- models

def train_model(payload: dict) -> dict:
    return _request("POST", "/models/train", json=payload).json()


def list_models() -> list[dict]:
    return _request("GET", "/models").json()


def get_model(slug: str) -> dict:
    return _request("GET", f"/models/{slug}").json()


def delete_model(slug: str) -> None:
    _request("DELETE", f"/models/{slug}")


def predict(slug: str, csv_bytes: bytes) -> pd.DataFrame:
    data = _request(
        "POST", f"/models/{slug}/predict",
        files={"file": ("records.csv", csv_bytes)},
    ).json()
    return pd.DataFrame(data["results"])


def explain(slug: str, csv_bytes: bytes, max_rows: int) -> dict:
    return _request(
        "POST", f"/models/{slug}/explain",
        files={"file": ("records.csv", csv_bytes)},
        data={"max_rows": max_rows},
    ).json()
