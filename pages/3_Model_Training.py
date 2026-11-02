
"""Train and register disease-specific models with manual control."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import api_client
from src.auth_ui import require_login
from src.split_ui import render_split_report
from src.theme import apply_theme, hero


st.set_page_config(
    page_title="Model Training | MediShield",
    page_icon="🧠",
    layout="wide",
)
apply_theme()
require_login()

try:
    _catalog = api_client.catalog()
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

HYPERPARAMETERS = _catalog["hyperparameters"]
MODEL_INFO = _catalog["models"]


def default_params(model_name: str) -> dict:
    return {s["name"]: s["default"] for s in HYPERPARAMETERS[model_name]}


hero(
    "Disease Model Training",
    "Choose algorithms, tune every hyperparameter, and decide which model "
    "gets deployed.",
)

df = st.session_state.get("cleaned_dataset")
if df is None:
    df = st.session_state.get("loaded_dataset")

if df is None or df.empty:
    st.info("Upload a historical labeled dataset before training.")
    st.stop()

target_options = list(df.columns)
previous_target = st.session_state.get("selected_target")
target_index = (
    target_options.index(previous_target)
    if previous_target in target_options
    else 0
)


def render_param_control(model_name: str, spec: dict):
    """Draw one widget for a hyperparameter and return its value."""
    key = f"hp_{model_name}_{spec['name']}"
    kind = spec["kind"]

    if kind == "choice":
        options = spec["options"]
        return st.selectbox(
            spec["label"],
            options,
            index=options.index(spec["default"]),
            help=spec["help"],
            key=key,
        )

    if kind == "int":
        return st.slider(
            spec["label"],
            min_value=spec["min"],
            max_value=spec["max"],
            value=spec["default"],
            step=spec["step"],
            help=spec["help"],
            key=key,
        )

    return st.number_input(
        spec["label"],
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        value=float(spec["default"]),
        step=float(spec["min"]),
        format="%.3f",
        help=spec["help"],
        key=key,
    )


st.subheader("1 · Dataset & split")
c1, c2, c3 = st.columns([2, 2, 3])
disease_name = c1.text_input(
    "Disease model name",
    placeholder="e.g. Stroke",
    help="Use a distinct name for each disease model.",
)
target_column = c2.selectbox(
    "Target label column",
    target_options,
    index=target_index,
)
test_size = c3.slider(
    "Held-out test proportion",
    min_value=0.1,
    max_value=0.4,
    value=st.session_state.get("split_test_size", 0.2),
    step=0.05,
    help="Share of records kept aside to evaluate the models.",
)

c4, c5 = st.columns(2)
random_state = c4.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=st.session_state.get("split_random_state", 42),
    step=1,
    help="Fixes the shuffling so results are reproducible.",
)

with st.expander("Preview the train / test split"):
    if st.button("Check split", key="check_split_button"):
        try:
            with st.spinner("Checking the split..."):
                render_split_report(api_client.split_check(
                    st.session_state.get("cleaned_dataset_id")
                    if st.session_state.get("cleaned_dataset") is not None
                    else st.session_state.get("loaded_dataset_id"),
                    target_column,
                    test_size,
                    int(random_state),
                ))
        except api_client.ApiError as exc:
            st.error(str(exc))

st.subheader("2 · Algorithms & hyperparameters")
st.markdown(
    '<div class="ms-note">Tick the algorithms to train, then open each tab '
    "to adjust its hyperparameters. Hover the <b>?</b> next to any control "
    "for a plain-language explanation. Defaults are sensible starting "
    "points.</div>",
    unsafe_allow_html=True,
)
st.write("")

selected_models = [
    name
    for name in HYPERPARAMETERS
    if st.checkbox(
        f"{MODEL_INFO[name]['icon']} {name}",
        value=True,
        key=f"use_{name}",
    )
]

hyperparameters = {}

if selected_models:
    tabs = st.tabs(
        [f"{MODEL_INFO[name]['icon']} {name}" for name in selected_models]
    )
    for tab, name in zip(tabs, selected_models):
        with tab:
            st.caption(MODEL_INFO[name]["summary"])
            cols = st.columns(2)
            values = {}
            for i, spec in enumerate(HYPERPARAMETERS[name]):
                with cols[i % 2]:
                    values[spec["name"]] = render_param_control(name, spec)
            hyperparameters[name] = values

            if values != default_params(name):
                st.caption("Customised: differs from the defaults.")

st.subheader("3 · Which model to deploy")
deploy_mode = st.radio(
    "Model selection",
    ["Automatic: best macro F1", "Manual: I choose"],
    horizontal=True,
    help=(
        "Automatic saves whichever trained model scores highest macro F1. "
        "Manual lets you pick the algorithm yourself."
    ),
)

save_model = None
if deploy_mode.startswith("Manual") and selected_models:
    save_model = st.selectbox(
        "Algorithm to save and use for predictions",
        selected_models,
    )

submitted = st.button(
    "Train and save disease model",
    type="primary",
    disabled=not selected_models,
)

if not selected_models:
    st.warning("Select at least one algorithm.")

if submitted:
    try:
        with st.spinner("Training models and preparing explanations..."):
            dataset_id = (
                st.session_state.get("cleaned_dataset_id")
                if st.session_state.get("cleaned_dataset") is not None
                else st.session_state.get("loaded_dataset_id")
            )
            output = api_client.train_model({
                "dataset_id": dataset_id,
                "target_column": target_column,
                "disease_name": disease_name,
                "test_size": test_size,
                "random_state": int(random_state),
                "selected_models": selected_models,
                "hyperparameters": hyperparameters,
                "save_model": save_model,
            })

        package = output["package"]
        st.session_state["last_trained_model_slug"] = package["model_slug"]

        st.success(
            f"Saved {package['disease_name']} "
            f"({package['best_model_name']}, "
            f"{package['selection_mode']} selection) to your account."
        )

        metrics = package["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric(
            "Balanced accuracy",
            f"{metrics['balanced_accuracy']:.3f}",
        )
        col3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

        st.subheader("Model comparison")
        results = pd.DataFrame(output["results"]).round(4)
        left, right = st.columns([2, 3])
        left.dataframe(results, width="stretch", hide_index=True)

        melted = results.melt(
            id_vars="model", var_name="metric", value_name="score"
        )
        fig = px.bar(
            melted,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            color_discrete_sequence=["#0B3C49", "#0E7C86", "#14B8A6"],
        )
        fig.update_layout(
            yaxis_range=[0, 1],
            xaxis_title=None,
            legend_title=None,
            margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        right.plotly_chart(fig, width="stretch")

        st.subheader("Hyperparameters used")
        for name, params in package["hyperparameters"].items():
            marker = " ⭐ saved" if name == package["best_model_name"] else ""
            with st.expander(f"{name}{marker}"):
                specs = {s["name"]: s for s in HYPERPARAMETERS[name]}
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Hyperparameter": [
                                specs[k]["label"] for k in params
                            ],
                            "Value": [str(v) for v in params.values()],
                            "What it does": [
                                specs[k]["help"] for k in params
                            ],
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

        if output["errors"]:
            with st.expander("Models that could not be trained"):
                for name, error in output["errors"].items():
                    st.write(f"**{name}:** {error}")

    except api_client.ApiError as exc:
        st.error(f"Training failed: {exc}")

st.divider()
st.subheader("Saved disease models")

try:
    saved_models = api_client.list_models()
except api_client.ApiError as exc:
    st.error(str(exc))
    saved_models = []

if not saved_models:
    st.info("No disease models have been saved yet.")
else:
    st.dataframe(
        [
            {
                "Disease": item["name"],
                "Target column": item["target"],
                "Selected algorithm": item["model"],
            }
            for item in saved_models
        ],
        width="stretch",
        hide_index=True,
    )
