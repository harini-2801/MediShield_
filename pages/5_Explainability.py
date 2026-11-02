
"""Explain predictions from the existing prediction session."""

import streamlit as st

from src.theme import apply_theme
import pandas as pd
import plotly.express as px

from src.explainability import (
    explain_records,
    global_importance,
    individual_importance,
)


st.set_page_config(
    page_title="Explainability | Medical AutoML",
    page_icon="🔍",
    layout="wide",
)
apply_theme()

st.title("Prediction Explainability")

package = st.session_state.get("prediction_package")
features = st.session_state.get("prediction_features")
results = st.session_state.get("prediction_results")
model_slug = st.session_state.get("prediction_model_slug")

if package is None or features is None or results is None:
    st.info(
        "First open Prediction, select a disease model, upload a CSV, "
        "and generate predictions. This page will then use those same records."
    )
    st.stop()

if st.session_state.get("active_model_slug") != model_slug:
    st.warning(
        "The selected disease model has changed. Generate predictions "
        "again on the Prediction page before explaining them."
    )
    st.stop()

st.write(f"**Disease model:** {package['disease_name']}")
st.write(f"**Algorithm:** {package['best_model_name']}")
st.write(f"**Records available:** {len(features)}")

st.subheader("Prediction batch")
st.dataframe(results, width="stretch", hide_index=True)

max_rows = min(20, len(features))

# A slider needs min_value < max_value, so a single record skips the slider.
if max_rows <= 1:
    rows_to_explain = max_rows
    st.caption("Explaining the only available record.")
else:
    rows_to_explain = st.slider(
        "Number of records to explain",
        min_value=1,
        max_value=max_rows,
        value=min(10, max_rows),
    )

if st.button("Generate explanations", type="primary"):
    try:
        with st.spinner("Calculating SHAP explanations..."):
            explanation = explain_records(
                package=package,
                input_data=features,
                max_rows=rows_to_explain,
            )
            importance_df = global_importance(explanation)

        st.session_state["active_explanation"] = explanation
        st.session_state["active_global_importance"] = importance_df
        st.session_state["explanation_model_slug"] = model_slug

    except Exception as exc:
        st.error(f"Could not generate explanations: {exc}")

explanation = st.session_state.get("active_explanation")
importance_df = st.session_state.get("active_global_importance")

if (
    explanation is not None
    and importance_df is not None
    and st.session_state.get("explanation_model_slug") == model_slug
):
    st.divider()
    st.subheader("Global feature importance")

    top_count = min(20, len(importance_df))
    global_chart_data = (
        importance_df.head(top_count)
        .sort_values("mean_absolute_shap", ascending=True)
    )

    fig = px.bar(
        global_chart_data,
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        labels={
            "mean_absolute_shap": "Mean absolute SHAP value",
            "feature": "Feature",
        },
        title="Features with the Largest Average Contributions",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        importance_df,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download global importance",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_slug}_global_importance.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Individual prediction explanation")

    row_index = st.selectbox(
        "Select record",
        options=list(range(explanation["rows"])),
        format_func=lambda index: f"Record {index + 1}",
    )

    class_names = explanation["class_names"]
    shap_values = explanation["shap_values"].values

    if getattr(shap_values, "ndim", 0) == 3:
        class_index = st.selectbox(
            "Class to explain",
            options=list(range(len(class_names))),
            format_func=lambda index: str(class_names[index]),
        )
    else:
        class_index = 0

    detail_df = individual_importance(
        explanation=explanation,
        row_index=row_index,
        class_index=class_index,
    )

    detail_chart_data = (
        detail_df.head(20)
        .sort_values("shap_value", ascending=True)
    )

    detail_fig = px.bar(
        detail_chart_data,
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={
            "shap_value": "SHAP value",
            "feature": "Feature",
        },
        title=f"Feature Contributions — Record {row_index + 1}",
    )
    detail_fig.update_layout(template="plotly_white")
    st.plotly_chart(detail_fig, width="stretch")

    st.dataframe(
        detail_df,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download individual explanation",
        data=detail_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_slug}_record_{row_index + 1}_explanation.csv",
        mime="text/csv",
    )

    st.caption(
        "SHAP values explain the model's output relative to a reference sample. "
        "They do not establish causation, clinical risk, or a diagnosis."
    )
