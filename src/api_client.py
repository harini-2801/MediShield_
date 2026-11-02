"""Thin HTTP client the Streamlit pages use to talk to the MediShield API."""

from __future__ import annotations

import io
import os
import secrets
import threading
import time
import traceback

import pandas as pd
import requests
import streamlit as st


_EMBEDDED_PORT = 8765
_embedded_thread: threading.Thread | None = None
_embedded_server = None
_embedded_error: str | None = None
_embedded_lock = threading.Lock()


def _run_embedded_backend() -> None:
    """Thread target: import the API and serve it on localhost."""
    global _embedded_server, _embedded_error
    try:
        import uvicorn

        from backend.main import app

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=_EMBEDDED_PORT,
            log_level="warning",
            loop="asyncio",
        )
        _embedded_server = uvicorn.Server(config)
        _embedded_server.run()
    except BaseException:
        _embedded_error = traceback.format_exc()


def _start_embedded_backend() -> None:
    """Start the API in a background thread of this process (non-blocking).

    Sharing the process with Streamlit avoids a second Python interpreter
    and its copy of pandas/numpy, which matters on 512 MB instances.
    """
    global _embedded_thread, _embedded_error
    with _embedded_lock:
        if _embedded_thread is not None and _embedded_thread.is_alive():
            return
        _embedded_error = None
        os.environ.setdefault("SECRET_KEY", secrets.token_urlsafe(48))
        _embedded_thread = threading.Thread(
            target=_run_embedded_backend, name="medishield-api", daemon=True
        )
        _embedded_thread.start()


def warm_up() -> None:
    """Begin starting the embedded API early, e.g. while the user logs in."""
    if not os.environ.get("MEDISHIELD_API_URL"):
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
        server = _embedded_server
        if server is not None and server.started:
            return url
        if not _embedded_thread.is_alive():
            raise ApiError(
                "The MediShield server failed to start. Details: "
                + (_embedded_error or "no error output")[-800:]
            )
        time.sleep(0.2)

    raise ApiError("The MediShield server is taking too long to start.")


def _base_url() -> str:
    if not os.environ.get("MEDISHIELD_API_URL"):
        return _ensure_embedded_backend()
    url = os.environ["MEDISHIELD_API_URL"].rstrip("/")
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
