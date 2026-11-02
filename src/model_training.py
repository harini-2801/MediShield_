
"""Train, evaluate, and manage disease-specific model packages."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.hyperparameters import HYPERPARAMETERS, build_models, default_params
from src.preprocessing import build_preprocessor
from src.splitting import IDENTIFIER_COLUMNS, prepare_split


# Anchor to the project root so saved models are found no matter which
# directory Streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

def slugify(value: str) -> str:
    """Convert a disease name into a safe filename."""
    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")

    if not slug:
        raise ValueError("Enter a valid disease model name.")

    return slug


def list_saved_models() -> list[dict]:
    """Return available disease model packages."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []

    for path in sorted(MODEL_DIR.glob("*.joblib")):
        try:
            package = joblib.load(path)

            if not isinstance(package, dict):
                continue

            models.append({
                "name": package.get(
                    "disease_name",
                    path.stem.replace("_", " ").title(),
                ),
                "slug": package.get("model_slug", path.stem),
                "target": package.get("target_column", "Unknown"),
                "model": package.get(
                    "best_model_name",
                    "Unknown",
                ),
                "path": str(path),
            })

        except Exception:
            # Ignore incompatible artifacts in the model directory.
            continue

    return models


def load_model_package(model_slug: str) -> dict:
    """Load a saved disease model using its disease slug."""
    slug = slugify(model_slug)
    path = MODEL_DIR / f"{slug}.joblib"

    if not path.is_file():
        available = [
            item["slug"] for item in list_saved_models()
        ]
        raise FileNotFoundError(
            f"No saved disease model exists for '{slug}'. "
            f"Available models: {available or 'none'}. "
            "Train and save this disease model first."
        )

    # Only load artifacts created by a trusted local training workflow.
    return validate_package(joblib.load(path))


def validate_package(package) -> dict:
    """Check that a loaded model package has every required entry."""
    if not isinstance(package, dict):
        raise ValueError("The saved model package is not valid.")

    required = {
        "pipeline",
        "disease_name",
        "model_slug",
        "target_column",
        "class_names",
        "feature_columns",
        "best_model_name",
        "metrics",
        "background_transformed",
        "transformed_feature_names",
    }

    missing = required - set(package.keys())

    if missing:
        raise ValueError(
            f"The saved model package is incomplete: {sorted(missing)}. "
            "Retrain this disease model."
        )

    return package


def train_disease_model(
    df: pd.DataFrame,
    target_column: str,
    disease_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    selected_models: list[str] | None = None,
    hyperparameters: dict[str, dict] | None = None,
    save_model: str | None = None,
    model_dir: Path | None = None,
) -> dict:
    """Train the chosen classifiers and save one disease model.

    selected_models: algorithms to train (default: all available).
    hyperparameters: {algorithm: {param: value}} overrides of the defaults.
    save_model: algorithm to deploy; None picks the best by macro F1.
    model_dir: where to write the artifact (default: artifacts/models).
    """
    selected_models = selected_models or list(HYPERPARAMETERS)

    if save_model is not None and save_model not in selected_models:
        raise ValueError(
            f"'{save_model}' must be among the trained models."
        )

    disease_name = disease_name.strip()

    if not disease_name:
        raise ValueError("Enter a disease model name.")

    # Same function the upload-time split check uses.
    split = prepare_split(df, target_column, test_size, random_state)
    X = split["X"]
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    label_encoder = split["label_encoder"]
    identifier_columns = split["identifier_columns"]

    models = build_models(
        selected_models,
        hyperparameters,
        random_state,
    )

    preprocessor_template = build_preprocessor(X_train)

    results = []
    successful_pipelines = {}
    errors = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor_template)),
            ("model", estimator),
        ])

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            results.append({
                "model": model_name,
                "accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, predictions)
                ),
                "macro_f1": float(
                    f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
            })

            successful_pipelines[model_name] = pipeline

        except Exception as exc:
            errors[model_name] = str(exc)

    if not results:
        raise RuntimeError(
            f"All model training attempts failed: {errors}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            ["macro_f1", "balanced_accuracy"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if save_model is not None:
        if save_model not in successful_pipelines:
            raise RuntimeError(
                f"'{save_model}' failed to train: "
                f"{errors.get(save_model, 'unknown error')}"
            )
        best_name = save_model
    else:
        best_name = str(results_df.iloc[0]["model"])

    best_pipeline = successful_pipelines[best_name]
    chosen_row = results_df.loc[results_df["model"] == best_name].iloc[0]

    # Store transformed reference records for later SHAP explanations.
    preprocessor = best_pipeline.named_steps["preprocessor"]

    background = preprocessor.transform(
        X_train.head(min(50, len(X_train)))
    )

    if hasattr(background, "toarray"):
        background = background.toarray()

    background = np.asarray(background, dtype=float)

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    model_slug = slugify(disease_name)

    metrics = {
        key: float(value)
        for key, value in chosen_row.to_dict().items()
        if key != "model"
    }

    used_params = {
        name: {**default_params(name), **(hyperparameters or {}).get(name, {})}
        for name in models
    }

    package = {
        "pipeline": best_pipeline,
        "disease_name": disease_name,
        "model_slug": model_slug,
        "target_column": target_column,
        "class_names": label_encoder.classes_.tolist(),
        "feature_columns": X.columns.tolist(),
        "identifier_columns": identifier_columns,
        "best_model_name": best_name,
        "metrics": metrics,
        "comparison_results": results_df,
        "hyperparameters": used_params,
        "selection_mode": "manual" if save_model else "auto",
        "fitted_params": best_pipeline.named_steps["model"].get_params(),
        "background_transformed": background,
        "transformed_feature_names": feature_names,
        "evaluation": {
            "y_true": np.asarray(y_test),
            "y_pred": np.asarray(
                best_pipeline.predict(X_test)
            ),
            "class_names": label_encoder.classes_.tolist(),
        },
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / f"{model_slug}.joblib"
    joblib.dump(package, model_path)

    return {
        "package": package,
        "results": results_df,
        "model_path": str(model_path),
        "errors": errors,
    }
