
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.hyperparameters import HYPERPARAMETERS, build_models, default_params
from src.preprocessing import build_preprocessor, split_features_target


# Anchor to the project root so saved models are found no matter which
# directory Streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

IDENTIFIER_COLUMNS = {
    "id",
    "patient_id",
    "record_id",
}


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
    package = joblib.load(path)

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
) -> dict:
    """Train the chosen classifiers and save one disease model.

    selected_models: algorithms to train (default: all available).
    hyperparameters: {algorithm: {param: value}} overrides of the defaults.
    save_model: algorithm to deploy; None picks the best by macro F1.
    """
    selected_models = selected_models or list(HYPERPARAMETERS)

    if save_model is not None and save_model not in selected_models:
        raise ValueError(
            f"'{save_model}' must be among the trained models."
        )

    if not 0.1 <= test_size <= 0.4:
        raise ValueError(
            "Test size must be between 0.1 and 0.4."
        )

    disease_name = disease_name.strip()

    if not disease_name:
        raise ValueError("Enter a disease model name.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    # Remove rows with missing target labels.
    labeled_df = df.loc[df[target_column].notna()].copy()

    if labeled_df.empty:
        raise ValueError(
            "The selected target contains no labeled records."
        )

    X, y = split_features_target(
        labeled_df,
        target_column,
    )

    # Identifiers should identify records, not predict disease.
    identifier_columns = [
        column
        for column in X.columns
        if str(column).strip().lower() in IDENTIFIER_COLUMNS
    ]

    X = X.drop(columns=identifier_columns)

    if X.empty:
        raise ValueError(
            "No predictive features remain after excluding IDs."
        )

    # Encode target labels consistently for the saved pipeline.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    class_counts = pd.Series(y_encoded).value_counts()
    n_classes = len(class_counts)

    if n_classes < 2:
        raise ValueError(
            "The target must contain at least two classes."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Every target class needs at least two records "
            "for a stratified train/test split."
        )

    n_test = max(
        int(np.ceil(len(y_encoded) * test_size)),
        n_classes,
    )

    if n_test >= len(y_encoded) or len(y_encoded) - n_test < n_classes:
        raise ValueError(
            "There are too few records for this stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_slug}.joblib"
    joblib.dump(package, model_path)

    return {
        "package": package,
        "results": results_df,
        "model_path": str(model_path),
        "errors": errors,
    }
