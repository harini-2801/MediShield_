
"""Batch predictions using saved disease-specific models."""

import streamlit as st

from src.theme import apply_theme
import pandas as pd

from src.model_training import (
    list_saved_models,
    load_model_package,
)
from src.prediction import predict_records


st.set_page_config(
    page_title="Prediction | Medical AutoML",
    page_icon="🔎",
    layout="wide",
)
apply_theme()

st.title("Patient Batch Prediction")

st.warning(
    "Research use only. Predictions are not medical diagnoses "
    "or validated clinical risk estimates."
)

saved_models = list_saved_models()

if not saved_models:
    st.info(
        "No disease models are available. Open Model Training, "
        "train a disease model, and save it first."
    )
    st.stop()

# IMPORTANT: Display name and algorithm, but pass the disease slug.
model_options = {
    f"{item['name']} — {item['model']}": item["slug"]
    for item in saved_models
}

labels = list(model_options.keys())

previous_slug = st.session_state.get("active_model_slug")

default_index = next(
    (
        index
        for index, label in enumerate(labels)
        if model_options[label] == previous_slug
    ),
    0,
)

selected_label = st.selectbox(
    "Select disease model",
    options=labels,
    index=default_index,
)

# This is the disease slug, e.g. "stroke", NOT "random_forest".
model_slug = model_options[selected_label]

try:
    package = load_model_package(model_slug)
except Exception as exc:
    st.error(f"Could not load selected model: {exc}")
    st.stop()

st.session_state["active_model_slug"] = model_slug

st.caption(
    f"Disease: {package['disease_name']} | "
    f"Target: {package['target_column']} | "
    f"Algorithm: {package['best_model_name']}"
)

with st.expander("Model hyperparameters"):
    from src.hyperparameters import HYPERPARAMETERS

    algo = package["best_model_name"]
    used = package.get("hyperparameters", {}).get(algo)
    if used is None:
        st.write("This model was saved before hyperparameters were recorded.")
    else:
        specs = {s["name"]: s for s in HYPERPARAMETERS[algo]}
        st.dataframe(
            pd.DataFrame({
                "Hyperparameter": [specs[k]["label"] for k in used],
                "Value": [str(v) for v in used.values()],
                "What it does": [specs[k]["help"] for k in used],
            }),
            width="stretch",
            hide_index=True,
        )

uploaded_file = st.file_uploader(
    "Upload a CSV containing new patient records",
    type=["csv"],
    key="prediction_csv",
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Normalize header whitespace.
        input_df.columns = [
            str(column).strip()
            for column in input_df.columns
        ]

        if input_df.columns.duplicated().any():
            st.error(
                "The uploaded CSV contains duplicate column names."
            )
            st.stop()

        st.subheader("Uploaded records")
        st.dataframe(
            input_df.head(10),
            width="stretch",
        )

        st.caption(
            f"{len(input_df):,} records | "
            f"{len(input_df.columns):,} columns"
        )

        if st.button(
            "Generate predictions",
            type="primary",
        ):
            with st.spinner("Generating predictions..."):
                results = predict_records(
                    input_data=input_df,
                    model_slug=model_slug,
                    package=package,
                )

                # Preserve exactly the feature columns used by the model.
                feature_df = input_df.loc[
                    :, package["feature_columns"]
                ].copy()

                st.session_state["prediction_package"] = package
                st.session_state["prediction_features"] = feature_df
                st.session_state["prediction_results"] = results
                st.session_state["prediction_model_slug"] = model_slug

                # Invalidate old explanations when new predictions are made.
                for key in (
                    "active_explanation",
                    "active_global_importance",
                    "explanation_model_slug",
                ):
                    st.session_state.pop(key, None)

    except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")

results = st.session_state.get("prediction_results")
same_model = (
    st.session_state.get("prediction_model_slug") == model_slug
)

if results is not None and same_model:
    st.divider()
    st.subheader("Prediction results")

    st.dataframe(
        results,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_slug}_predictions.csv",
        mime="text/csv",
    )

    st.success(
        "These records are available to the Explainability page. "
        "You do not need to upload the CSV again."
    )
