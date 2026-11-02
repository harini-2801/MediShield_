
"""Medical AutoML: data cleaning and quality review page."""

import pandas as pd
import streamlit as st

from src.theme import apply_theme

from src.data_cleaning import clean_dataset


st.set_page_config(
    page_title="Data Cleaning | Medical AutoML",
    page_icon="🧹",
    layout="wide",
)
apply_theme()

st.title("Data Cleaning")
st.caption(
    "Review data quality, correct safe formatting issues, "
    "and prepare your dataset for machine learning."
)

# ---------------------------------------------------------
# 1. Retrieve and validate the uploaded dataset
# ---------------------------------------------------------

df = st.session_state.get("loaded_dataset")

if not isinstance(df, pd.DataFrame):
    st.warning(
        "No dataset is available. Open the Dataset Upload page "
        "and upload a CSV file first."
    )
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The uploaded dataset contains no usable rows or columns.")
    st.stop()

# ---------------------------------------------------------
# 2. Invalidate stale cleaning results when the source changes
# ---------------------------------------------------------

previous_source = st.session_state.get("cleaning_source_dataset")

if previous_source is not df:
    st.session_state.pop("cleaned_dataset", None)
    st.session_state.pop("cleaning_report", None)
    st.session_state["cleaning_source_dataset"] = df

    # Record that the previous cleaning results are no longer current.
    st.session_state.pop("cleaning_summary", None)

# ---------------------------------------------------------
# 3. Display original dataset information
# ---------------------------------------------------------

missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Rows", f"{len(df):,}")

with metric2:
    st.metric("Columns", f"{len(df.columns):,}")

with metric3:
    st.metric("Missing cells", f"{missing_cells:,}")

with metric4:
    st.metric("Duplicate rows", f"{duplicate_rows:,}")

with st.expander("Preview original dataset", expanded=True):
    st.dataframe(
        df.head(20),
        width="stretch",
        hide_index=True,
    )

with st.expander("Original column information"):
    column_info = pd.DataFrame({
        "column": df.columns.astype(str),
        "data_type": [
            str(dtype) for dtype in df.dtypes
        ],
        "missing_values": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
        "unique_values": [
            int(df[column].nunique(dropna=True))
            for column in df.columns
        ],
    })

    st.dataframe(
        column_info,
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------
# 4. Cleaning options
# ---------------------------------------------------------

st.subheader("Cleaning options")

st.write(
    "Choose which safe data-quality operations to apply. "
    "Original records and potential outliers are preserved."
)

with st.form("data_cleaning_form"):
    normalize_numeric = st.checkbox(
        "Convert numeric-looking text columns to numeric values",
        value=True,
        help=(
            "Converts a column only when all non-missing values "
            "can be interpreted as numbers. Identifier columns "
            "such as patient_id are excluded."
        ),
    )

    normalize_categories = st.checkbox(
        "Normalize repeated whitespace in categorical values",
        value=True,
        help=(
            "For example, 'Full  time' becomes 'Full time'. "
            "Category capitalization is preserved."
        ),
    )

    report_outliers = st.checkbox(
        "Detect and report potential numeric outliers",
        value=True,
        help=(
            "Uses the interquartile range (IQR) rule. "
            "Outlier values are reported, not removed."
        ),
    )

    outlier_multiplier = st.selectbox(
        "Outlier detection sensitivity",
        options=[1.5, 2.0, 3.0],
        index=0,
        help=(
            "A multiplier of 1.5 is the conventional IQR rule. "
            "Higher values flag fewer observations."
        ),
        format_func=lambda value: {
            1.5: "Standard — 1.5 × IQR",
            2.0: "Moderate — 2.0 × IQR",
            3.0: "Conservative — 3.0 × IQR",
        }[value],
    )

    clean_clicked = st.form_submit_button(
        "Clean dataset",
        type="primary",
        width="stretch",
    )

# ---------------------------------------------------------
# 5. Run cleaning
# ---------------------------------------------------------

if clean_clicked:
    try:
        with st.spinner(
            "Cleaning the dataset and preparing the quality report..."
        ):
            cleaned_df, cleaning_report = clean_dataset(
                df,
                normalize_numeric=normalize_numeric,
                normalize_categories=normalize_categories,
                report_outliers=report_outliers,
                outlier_iqr_multiplier=outlier_multiplier,
            )

        st.session_state["cleaned_dataset"] = cleaned_df
        st.session_state["cleaning_report"] = cleaning_report
        st.session_state["cleaning_source_dataset"] = df

        st.session_state["cleaning_summary"] = {
            "rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "missing_cells": int(
                cleaned_df.isna().sum().sum()
            ),
            "duplicate_rows": int(
                cleaned_df.duplicated().sum()
            ),
        }

        # A newly cleaned dataset invalidates explanations from
        # previous predictions, since their source data may differ.
        for key in (
            "active_explanation",
            "active_global_importance",
            "explanation_model_slug",
        ):
            st.session_state.pop(key, None)

        st.success("Data cleaning completed successfully.")

    except (ValueError, TypeError) as exc:
        st.error(f"Could not clean this dataset: {exc}")

    except Exception as exc:
        st.error(
            "An unexpected error occurred during data cleaning. "
            "No new cleaning results were saved."
        )
        st.exception(exc)

# ---------------------------------------------------------
# 6. Display cleaned dataset and report
# ---------------------------------------------------------

cleaned_df = st.session_state.get("cleaned_dataset")
cleaning_report = st.session_state.get("cleaning_report")

if not isinstance(cleaned_df, pd.DataFrame):
    st.info(
        "Choose your cleaning options above and select "
        "'Clean dataset' to generate the cleaned version."
    )
    st.stop()

st.divider()
st.subheader("Cleaning results")

summary = st.session_state.get("cleaning_summary", {})

result1, result2, result3, result4 = st.columns(4)

with result1:
    st.metric(
        "Cleaned rows",
        f"{len(cleaned_df):,}",
        delta=f"{len(cleaned_df) - len(df):+,} rows",
    )

with result2:
    st.metric(
        "Columns",
        f"{len(cleaned_df.columns):,}",
    )

with result3:
    st.metric(
        "Missing cells",
        f"{int(cleaned_df.isna().sum().sum()):,}",
        delta=(
            f"{int(cleaned_df.isna().sum().sum()) - missing_cells:+,}"
            if missing_cells != int(cleaned_df.isna().sum().sum())
            else "No change"
        ),
        delta_color="inverse",
    )

with result4:
    st.metric(
        "Duplicate rows",
        f"{int(cleaned_df.duplicated().sum()):,}",
    )

st.subheader("Cleaning report")

if (
    not isinstance(cleaning_report, pd.DataFrame)
    or cleaning_report.empty
):
    st.info("No changes or data-quality issues were detected.")
else:
    st.dataframe(
        cleaning_report,
        width="stretch",
        hide_index=True,
    )

    report_csv = cleaning_report.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download cleaning report",
        data=report_csv,
        file_name="medical_data_cleaning_report.csv",
        mime="text/csv",
    )

st.subheader("Cleaned dataset")

st.dataframe(
    cleaned_df.head(50),
    width="stretch",
    hide_index=True,
)

cleaned_csv = cleaned_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download cleaned dataset",
    data=cleaned_csv,
    file_name="cleaned_medical_dataset.csv",
    mime="text/csv",
)

# ---------------------------------------------------------
# 7. Explain the next stage
# ---------------------------------------------------------

with st.expander("What still happens during model preprocessing?"):
    st.markdown(
        """
        Cleaning does not automatically remove records or fill missing
        feature values.

        During model training, the preprocessing pipeline should handle:

        - **Missing numerical values:** median imputation.
        - **Missing categorical values:** most-frequent imputation.
        - **Categorical features:** one-hot encoding.
        - **Numerical features:** optional standard scaling, especially
          for Logistic Regression and SVM.
        - **Unknown categories in prediction data:** safely ignored by
          the fitted encoder.

        These transformations must be fitted on the training split only
        and reused for validation, testing, prediction, and explainability.

        Keep your target column intact. Do not automatically delete
        outliers or duplicate rows without first determining whether
        they represent errors or legitimate patient records.
        """
    )
