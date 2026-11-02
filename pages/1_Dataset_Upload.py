
"""Medical AutoML: dataset upload and inspection page."""

import streamlit as st

import hashlib

import pandas as pd

from src import api_client
from src.auth_ui import require_login
from src.split_ui import render_split_report
from src.theme import apply_theme


st.set_page_config(
    page_title="Dataset Upload | Medical AutoML",
    page_icon="📂",
    layout="wide",
)
apply_theme()
require_login()

st.title("Dataset Upload")
st.write(
    "Upload a medical dataset, inspect its quality, "
    "and select the disease label for supervised learning."
)

st.caption(
    "This page inspects the data only. It does not train a model "
    "or produce a medical prediction."
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX, and XLS.",
)

if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
filename = uploaded_file.name

# Use a content-based key so changing files invalidates the
# previously selected sheet and target.
file_key = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("ingestion_file_key") != file_key:
    st.session_state["ingestion_file_key"] = file_key
    for key in (
        "selected_target", "selected_sheet", "loaded_dataset",
        "loaded_dataset_id", "loaded_profile", "upload_key",
        "cleaned_dataset", "cleaned_dataset_id", "cleaning_report",
    ):
        st.session_state.pop(key, None)

sheet_name = None

try:
    if filename.lower().endswith((".xlsx", ".xls")):
        sheet_names = api_client.sheet_names(file_bytes, filename)

        if not sheet_names:
            st.error("No worksheets were found in this workbook.")
            st.stop()

        previous_sheet = st.session_state.get("selected_sheet")
        default_index = (
            sheet_names.index(previous_sheet)
            if previous_sheet in sheet_names
            else 0
        )

        sheet_name = st.selectbox(
            "Choose an Excel worksheet",
            options=sheet_names,
            index=default_index,
            key="excel_sheet_selector",
        )

        st.session_state["selected_sheet"] = sheet_name

    # Upload once per file/sheet, not on every Streamlit rerun.
    if st.session_state.get("upload_key") != (file_key, sheet_name):
        with st.spinner("Uploading and profiling dataset..."):
            uploaded = api_client.upload_dataset(
                file_bytes, filename, sheet_name
            )
            st.session_state["loaded_dataset"] = (
                api_client.download_dataset(uploaded["id"])
            )
        st.session_state["loaded_dataset_id"] = uploaded["id"]
        st.session_state["loaded_profile"] = uploaded["profile"]
        st.session_state["upload_key"] = (file_key, sheet_name)
        for key in (
            "cleaned_dataset", "cleaned_dataset_id", "cleaning_report",
        ):
            st.session_state.pop(key, None)

    df = st.session_state["loaded_dataset"]

except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Successfully loaded: {filename}")

# Dataset overview
st.subheader("Dataset overview")

profile = dict(st.session_state["loaded_profile"])
profile["column_profiles"] = pd.DataFrame(profile["column_profiles"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", f"{profile['row_count']:,}")
col2.metric("Columns", f"{profile['column_count']:,}")
col3.metric(
    "Missing cells",
    f"{profile['total_missing_cells']:,}",
)
col4.metric(
    "Duplicate rows",
    f"{profile['duplicate_rows']:,}",
)

st.caption(
    f"Columns with missing values: "
    f"{profile['columns_with_missing']}"
)

# Data preview
st.subheader("Data preview")

# A slider needs min_value < max_value, so small datasets are shown whole.
preview_limit = min(50, len(df))

if preview_limit <= 5:
    preview_rows = len(df)
    st.caption(f"Showing all {len(df):,} rows.")
else:
    preview_rows = st.slider(
        "Number of rows to preview",
        min_value=5,
        max_value=preview_limit,
        value=min(10, preview_limit),
    )

st.dataframe(
    df.head(preview_rows),
    width="stretch",
)

# Column profile
st.subheader("Column profile")

column_profile = profile["column_profiles"]

st.dataframe(
    column_profile,
    width="stretch",
    hide_index=True,
)

# Missing-value detail
st.subheader("Missing-value analysis")

missing_profile = column_profile[
    column_profile["missing_count"] > 0
].sort_values("missing_count", ascending=False)

if missing_profile.empty:
    st.success("No missing values or recognized placeholders detected.")
else:
    st.dataframe(
        missing_profile[
            ["column", "missing_count", "missing_pct"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        missing_profile.set_index("column")["missing_count"]
    )

# Target configuration
st.subheader("Configure prediction target")

st.write(
    "Choose the column containing the known disease labels. "
    "It will be used later as the supervised training target, "
    "not as a patient prediction input."
)

target_options = list(df.columns)

previous_target = st.session_state.get("selected_target")
target_index = (
    target_options.index(previous_target)
    if previous_target in target_options
    else 0
)

selected_target = st.selectbox(
    "Disease / target column",
    options=target_options,
    index=target_index,
    key="target_selector",
)

if st.button("Confirm target column", type="primary"):
    st.session_state["selected_target"] = selected_target
    st.success(f"Target column selected: {selected_target}")

if st.session_state.get("selected_target") in target_options:
    target = st.session_state["selected_target"]

    st.subheader("Target label distribution")

    target_counts = (
        df[target]
        .fillna("(missing)")
        .astype(str)
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    st.dataframe(
        target_counts,
        width="stretch",
        hide_index=True,
    )

    if df[target].nunique(dropna=True) < 2:
        st.warning(
            "This target has fewer than two observed classes. "
            "It may not be suitable for classification."
        )

    st.subheader("Train / test split check")
    st.write(
        "Models are trained on the training set and scored on the held-out "
        "test set. This is the exact split that will be used for training, "
        "so you can check it before you train."
    )

    s1, s2 = st.columns([3, 1])
    split_test_size = s1.slider(
        "Held-out test proportion",
        min_value=0.1,
        max_value=0.4,
        value=st.session_state.get("split_test_size", 0.2),
        step=0.05,
        key="upload_split_test_size",
        help="Share of labeled records kept aside to evaluate the models.",
    )
    split_seed = s2.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=st.session_state.get("split_random_state", 42),
        step=1,
        key="upload_split_seed",
        help="Fixes the shuffling so the split is reproducible.",
    )
    st.session_state["split_test_size"] = split_test_size
    st.session_state["split_random_state"] = int(split_seed)

    try:
        with st.spinner("Checking the split..."):
            report = api_client.split_check(
                st.session_state["loaded_dataset_id"],
                target,
                split_test_size,
                int(split_seed),
            )
        render_split_report(report)
    except api_client.ApiError as exc:
        st.error(str(exc))

st.divider()

st.caption(
    "Next stage: controlled data cleaning, target validation, "
    "and leakage-safe preprocessing."
)
