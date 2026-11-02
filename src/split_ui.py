"""Streamlit rendering of the train/test split report."""

import pandas as pd
import plotly.express as px
import streamlit as st


def render_split_report(report: dict) -> None:
    """Show split sizes, per-class balance, and any warnings."""
    if not report.get("valid"):
        st.error(f"This split cannot be created: {report.get('error')}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labeled rows", f"{report['total_rows']:,}")
    c2.metric("Training rows", f"{report['train_rows']:,}")
    c3.metric(
        "Test rows",
        f"{report['test_rows']:,}",
        help=f"{report['test_pct']}% held out for evaluation.",
    )
    c4.metric("Features", f"{report['feature_count']:,}")

    st.caption(
        f"Stratified split (each class is divided in the same proportion "
        f"into train and test), random seed {report['random_state']}."
    )

    classes = pd.DataFrame(report["classes"])
    long = classes.melt(
        id_vars="class",
        value_vars=["train", "test"],
        var_name="split",
        value_name="records",
    )
    fig = px.bar(
        long,
        x="class",
        y="records",
        color="split",
        barmode="group",
        color_discrete_map={"train": "#0E7C86", "test": "#F59E0B"},
    )
    fig.update_layout(
        xaxis_title=None,
        legend_title=None,
        margin=dict(t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, width="stretch")
    right.dataframe(
        classes.rename(columns={
            "class": "Class",
            "total": "Total",
            "train": "Train",
            "test": "Test",
            "share_of_data_pct": "% of data",
            "train_pct": "% of train",
            "test_pct": "% of test",
        }),
        width="stretch",
        hide_index=True,
    )

    if report["warnings"]:
        for message in report["warnings"]:
            st.warning(message)
    else:
        st.success(
            "The split looks healthy: every class appears in both sets "
            "in matching proportions, with no duplicate leakage."
        )
