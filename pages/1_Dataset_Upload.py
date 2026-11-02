
"""Medical AutoML: dataset upload and inspection page."""

import streamlit as st

from src.theme import apply_theme

from src.ingestion import (
    get_excel_sheet_names,
    load_dataset,
    profile_dataset,
)


st.set_page_config(
    page_title="Dataset Upload | Medical AutoML",
    page_icon="📂",
    layout="wide",
)
apply_theme()

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
import hashlib

file_key = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("ingestion_file_key") != file_key:
    st.session_state["ingestion_file_key"] = file_key
    st.session_state.pop("selected_target", None)
    st.session_state.pop("selected_sheet", None)
    st.session_state.pop("loaded_dataset", None)

sheet_name = None

try:
    if filename.lower().endswith((".xlsx", ".xls")):
        sheet_names = get_excel_sheet_names(file_bytes)

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

    df = load_dataset(
        file_bytes=file_bytes,
        filename=filename,
        sheet_name=sheet_name,
    )

except ValueError as exc:
    st.error(str(exc))
    st.stop()

except Exception as exc:
    st.error(f"Unexpected ingestion error: {exc}")
    st.stop()

st.session_state["loaded_dataset"] = df

st.success(f"Successfully loaded: {filename}")

# Dataset overview
st.subheader("Dataset overview")

profile = profile_dataset(df)

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

st.divider()

st.caption(
    "Next stage: controlled data cleaning, target validation, "
    "and leakage-safe preprocessing."
)
