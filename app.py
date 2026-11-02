
"""MediShield application home page."""

import streamlit as st

from src.theme import apply_theme, card, hero


st.set_page_config(
    page_title="MediShield",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

hero(
    "🩺 MediShield",
    "Medical dataset analysis and machine learning: upload, clean, train, "
    "predict and explain.",
)

st.warning(
    "Research and development use only. This application is not a "
    "diagnostic device and must not be used to make clinical decisions "
    "without appropriate validation, governance, and professional oversight."
)

st.header("Application workflow")

steps = [
    ("1", "Dataset Upload",
     "Upload CSV or Excel files, review columns, missing values and "
     "duplicates, and select the target label."),
    ("2", "Data Cleaning",
     "Normalise names and whitespace, detect missing-value markers and "
     "review the cleaning report."),
    ("3", "Model Training",
     "Pick algorithms, tune explained hyperparameters, compare metrics and "
     "choose which model to deploy."),
    ("4", "Prediction",
     "Upload new records, generate model outputs and export them to CSV."),
    ("5", "Explainability",
     "Review global feature importance and individual SHAP explanations."),
]

cols = st.columns(len(steps))
for col, (num, title, body) in zip(cols, steps):
    with col:
        card(title, body, step=num)

st.write("")
st.header("What's new")
n1, n2 = st.columns(2)
with n1:
    card(
        "Hyperparameter control",
        "Every model exposes its hyperparameters with plain-language "
        "explanations, so you can see and decide how it learns.",
    )
with n2:
    card(
        "Manual model selection",
        "Train any subset of algorithms and either let MediShield pick the "
        "best by macro F1 or choose the one to deploy yourself.",
    )

st.write("")
st.info(
    "Only use data you are authorized to process. Remove direct "
    "patient identifiers where possible, protect sensitive health "
    "information, and use trusted model artifacts."
)

st.caption("MediShield | Experimental machine learning platform")
