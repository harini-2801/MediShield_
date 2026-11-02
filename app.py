
"""MediShield application home page."""

import streamlit as st

from src.auth_ui import require_login
from src.theme import apply_theme, card, hero


st.set_page_config(
    page_title="MediShield",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
require_login()

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
st.info(
    "Only use data you are authorized to process. Remove direct "
    "patient identifiers where possible, protect sensitive health "
    "information, and use trusted model artifacts."
)

st.caption("MediShield | Experimental machine learning platform")
