"""End-to-end test of the REST API using an in-process client."""

import io
import os

import numpy as np
import pandas as pd
import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test_medishield.db"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256"

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    for suffix in ("", "-journal"):
        try:
            os.remove(f"test_medishield.db{suffix}")
        except OSError:
            pass


def _csv(df):
    return df.to_csv(index=False).encode()


def _auth(client, name):
    r = client.post(
        "/auth/signup", json={"username": name, "password": "password123"}
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_auth_rules(client):
    _auth(client, "alice")
    dup = client.post(
        "/auth/signup", json={"username": "alice", "password": "password123"}
    )
    assert dup.status_code == 409
    bad = client.post(
        "/auth/login", json={"username": "alice", "password": "wrongpass1"}
    )
    assert bad.status_code == 401
    ok = client.post(
        "/auth/login", json={"username": "alice", "password": "password123"}
    )
    assert ok.status_code == 200
    assert client.get("/datasets").status_code == 401


def test_full_workflow_and_isolation(client):
    alice = _auth(client, "bob_a")
    eve = _auth(client, "eve_e")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    df = pd.DataFrame(x, columns=list("abcd"))
    df["y"] = np.where(x[:, 0] + x[:, 1] > 0, "yes", "no")

    up = client.post(
        "/datasets", headers=alice,
        files={"file": ("d.csv", _csv(df))},
    )
    assert up.status_code == 200, up.text
    ds = up.json()["id"]

    cleaned = client.post(
        f"/datasets/{ds}/clean", headers=alice, json={}
    )
    assert cleaned.status_code == 200, cleaned.text

    check = client.post(
        f"/datasets/{cleaned.json()['id']}/split-check", headers=alice,
        json={"target_column": "y", "test_size": 0.25, "random_state": 3},
    )
    assert check.status_code == 200 and check.json()["valid"], check.text
    assert check.json()["test_rows"] == 50

    train = client.post("/models/train", headers=alice, json={
        "dataset_id": cleaned.json()["id"],
        "target_column": "y",
        "disease_name": "Demo Disease",
        "selected_models": ["Random Forest", "Logistic Regression"],
        "hyperparameters": {"Random Forest": {"n_estimators": 30}},
        "save_model": "Logistic Regression",
    })
    assert train.status_code == 200, train.text
    assert train.json()["package"]["best_model_name"] == "Logistic Regression"

    slug = "demo_disease"
    assert [m["slug"] for m in client.get("/models", headers=alice).json()] == [slug]

    features = df.drop(columns="y").head(5)
    pred = client.post(
        f"/models/{slug}/predict", headers=alice,
        files={"file": ("p.csv", _csv(features))},
    )
    assert pred.status_code == 200, pred.text
    assert len(pred.json()["results"]) == 5

    exp = client.post(
        f"/models/{slug}/explain", headers=alice,
        files={"file": ("p.csv", _csv(features))}, data={"max_rows": 3},
    )
    assert exp.status_code == 200, exp.text
    assert exp.json()["rows"] == 3

    # Another user cannot see or use Alice's data.
    assert client.get("/models", headers=eve).json() == []
    assert client.get(f"/models/{slug}", headers=eve).status_code == 404
    assert client.get(f"/datasets/{ds}/csv", headers=eve).status_code == 404
