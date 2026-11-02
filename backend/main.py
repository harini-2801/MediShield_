"""MediShield REST API: accounts, datasets, training, prediction, SHAP."""

import io
import json
import pickle
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.db import Dataset, TrainedModel, User, get_db, init_db
from backend.security import (
    create_token,
    current_user,
    hash_password,
    verify_password,
)
from src.data_cleaning import clean_dataset
from src.hyperparameters import HYPERPARAMETERS, MODEL_INFO
from src.ingestion import get_excel_sheet_names, load_dataset, profile_dataset
from src.model_training import train_disease_model, validate_package
from src.prediction import predict_records, prepare_prediction_input

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="MediShield API", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ helpers

def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return json.loads(df.to_json(orient="records"))


def read_upload(file: UploadFile) -> bytes:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    return data


def dump_df(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)


def get_dataset(db: Session, user: User, dataset_id: int) -> Dataset:
    row = db.get(Dataset, dataset_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(404, "Dataset not found.")
    return row


def load_df(row: Dataset) -> pd.DataFrame:
    # Blobs are written only by this service for the owning user.
    return pickle.loads(row.data)


def get_model_row(db: Session, user: User, slug: str) -> TrainedModel:
    row = db.scalar(
        select(TrainedModel).where(
            TrainedModel.user_id == user.id, TrainedModel.slug == slug
        )
    )
    if row is None:
        raise HTTPException(404, "Model not found.")
    return row


def load_package(row: TrainedModel) -> dict:
    return validate_package(joblib.load(io.BytesIO(row.blob)))


def package_summary(package: dict) -> dict:
    comparison = package.get("comparison_results")
    return {
        "disease_name": package["disease_name"],
        "model_slug": package["model_slug"],
        "target_column": package["target_column"],
        "best_model_name": package["best_model_name"],
        "selection_mode": package.get("selection_mode", "auto"),
        "class_names": [str(c) for c in package["class_names"]],
        "feature_columns": list(package["feature_columns"]),
        "metrics": package["metrics"],
        "hyperparameters": package.get("hyperparameters", {}),
        "comparison_results": (
            records(comparison) if comparison is not None else []
        ),
        "training_rows": package.get("training_rows"),
        "test_rows": package.get("test_rows"),
    }


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    data = read_upload(file)
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(400, f"Could not read the CSV: {exc}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------- auth

class Credentials(BaseModel):
    username: str
    password: str = Field(min_length=8, max_length=128)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup")
def signup(body: Credentials, db: Session = Depends(get_db)):
    username = body.username.strip()
    if not USERNAME_RE.match(username):
        raise HTTPException(
            422,
            "Username must be 3-40 characters: letters, numbers, . _ -",
        )
    user = User(username=username, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "That username is already taken.")
    return {"token": create_token(user.id), "username": user.username}


@app.post("/auth/login")
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.scalar(
        select(User).where(User.username == body.username.strip())
    )
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect username or password.")
    return {"token": create_token(user.id), "username": user.username}


@app.get("/auth/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username}


# ------------------------------------------------------------------ catalog

@app.get("/catalog/hyperparameters")
def catalog(_: User = Depends(current_user)):
    return {"hyperparameters": HYPERPARAMETERS, "models": MODEL_INFO}


# ----------------------------------------------------------------- datasets

@app.post("/datasets/sheets")
def sheets(file: UploadFile = File(...), _: User = Depends(current_user)):
    try:
        return {"sheets": get_excel_sheet_names(read_upload(file))}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/datasets")
def upload_dataset(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    try:
        df = load_dataset(read_upload(file), file.filename or "", sheet_name)
        profile = profile_dataset(df)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    row = Dataset(
        user_id=user.id,
        name=file.filename,
        kind="raw",
        rows=len(df),
        columns=len(df.columns),
        data=dump_df(df),
    )
    db.add(row)
    db.commit()

    profile["column_profiles"] = records(profile["column_profiles"])
    return {"id": row.id, "name": row.name, "profile": profile}


@app.get("/datasets")
def list_datasets(
    db: Session = Depends(get_db), user: User = Depends(current_user)
):
    rows = db.scalars(
        select(Dataset)
        .where(Dataset.user_id == user.id)
        .order_by(Dataset.created_at.desc())
    )
    return [
        {
            "id": r.id, "name": r.name, "kind": r.kind,
            "rows": r.rows, "columns": r.columns,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.get("/datasets/{dataset_id}/csv")
def dataset_csv(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    df = load_df(get_dataset(db, user, dataset_id))
    return Response(df.to_csv(index=False), media_type="text/csv")


@app.delete("/datasets/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    db.delete(get_dataset(db, user, dataset_id))
    db.commit()
    return {"deleted": dataset_id}


class CleanRequest(BaseModel):
    normalize_numeric: bool = True
    normalize_categories: bool = True
    report_outliers: bool = True
    outlier_iqr_multiplier: float = Field(1.5, ge=0.5, le=10)


@app.post("/datasets/{dataset_id}/clean")
def clean(
    dataset_id: int,
    body: CleanRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    source = get_dataset(db, user, dataset_id)
    try:
        cleaned, report = clean_dataset(
            load_df(source),
            normalize_numeric=body.normalize_numeric,
            normalize_categories=body.normalize_categories,
            report_outliers=body.report_outliers,
            outlier_iqr_multiplier=body.outlier_iqr_multiplier,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Could not clean this dataset: {exc}")

    row = Dataset(
        user_id=user.id,
        name=f"{source.name} (cleaned)",
        kind="cleaned",
        rows=len(cleaned),
        columns=len(cleaned.columns),
        data=dump_df(cleaned),
    )
    db.add(row)
    db.commit()

    report_records = (
        records(report) if isinstance(report, pd.DataFrame) else []
    )
    return {"id": row.id, "report": report_records}


# ------------------------------------------------------------------- models

class TrainRequest(BaseModel):
    dataset_id: int
    target_column: str
    disease_name: str
    test_size: float = 0.2
    random_state: int = Field(42, ge=0, le=9999)
    selected_models: list[str] | None = None
    hyperparameters: dict[str, dict] | None = None
    save_model: str | None = None


@app.post("/models/train")
def train(
    body: TrainRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    df = load_df(get_dataset(db, user, body.dataset_id))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            output = train_disease_model(
                df=df,
                target_column=body.target_column,
                disease_name=body.disease_name,
                test_size=body.test_size,
                random_state=body.random_state,
                selected_models=body.selected_models,
                hyperparameters=body.hyperparameters,
                save_model=body.save_model,
                model_dir=Path(tmp),
            )
            blob = Path(output["model_path"]).read_bytes()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))

    package = output["package"]
    existing = db.scalar(
        select(TrainedModel).where(
            TrainedModel.user_id == user.id,
            TrainedModel.slug == package["model_slug"],
        )
    )
    if existing:
        existing.disease_name = package["disease_name"]
        existing.algorithm = package["best_model_name"]
        existing.target_column = package["target_column"]
        existing.blob = blob
    else:
        db.add(TrainedModel(
            user_id=user.id,
            slug=package["model_slug"],
            disease_name=package["disease_name"],
            algorithm=package["best_model_name"],
            target_column=package["target_column"],
            blob=blob,
        ))
    db.commit()

    return {
        "package": package_summary(package),
        "results": records(output["results"]),
        "errors": output["errors"],
    }


@app.get("/models")
def list_models(
    db: Session = Depends(get_db), user: User = Depends(current_user)
):
    rows = db.scalars(
        select(TrainedModel)
        .where(TrainedModel.user_id == user.id)
        .order_by(TrainedModel.created_at.desc())
    )
    return [
        {
            "name": r.disease_name, "slug": r.slug,
            "target": r.target_column, "model": r.algorithm,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.get("/models/{slug}")
def model_detail(
    slug: str,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    return package_summary(load_package(get_model_row(db, user, slug)))


@app.delete("/models/{slug}")
def delete_model(
    slug: str,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    db.delete(get_model_row(db, user, slug))
    db.commit()
    return {"deleted": slug}


@app.post("/models/{slug}/predict")
def predict(
    slug: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    package = load_package(get_model_row(db, user, slug))
    df = read_csv_upload(file)
    try:
        results = predict_records(df, slug, package=package)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"results": records(results)}


@app.post("/models/{slug}/explain")
def explain(
    slug: str,
    file: UploadFile = File(...),
    max_rows: int = Form(10),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    # Imported lazily: shap is heavy and only needed for explanations.
    from src.explainability import (
        explain_records,
        global_importance,
        individual_importance,
    )

    package = load_package(get_model_row(db, user, slug))
    df = read_csv_upload(file)
    max_rows = max(1, min(max_rows, 20))

    try:
        features = prepare_prediction_input(df, package)
        explanation = explain_records(package, features, max_rows=max_rows)
        importance = global_importance(explanation)

        values = np.asarray(explanation["shap_values"].values)
        class_count = values.shape[2] if values.ndim == 3 else 1

        details = [
            [
                records(individual_importance(explanation, row, cls))
                for cls in range(class_count)
            ]
            for row in range(explanation["rows"])
        ]
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))

    return {
        "rows": explanation["rows"],
        "class_names": [str(c) for c in explanation["class_names"]],
        "multiclass": values.ndim == 3,
        "global_importance": records(importance),
        "details": details,
    }
