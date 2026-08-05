import streamlit as st
import pandas as pd
from typing import Callable
from CORE.models import AnalysisResult


def render_dataset_overview(df):
    if df is not None:

        st.header("📊 Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Rows", len(df))

        with c2:
            st.metric("Columns", len(df.columns))

        with c3:
            st.metric(
                "Numeric Columns",
                len(df.select_dtypes(include="number").columns)
            )

        with c4:
            st.metric(
                "Categorical Columns",
                len(df.select_dtypes(exclude="number").columns)
            )

        # ---------------------------------------------------------
        #                     Missing Values
        # ---------------------------------------------------------

        missing = df.isnull().sum().sum()

        st.metric(
            "Missing Values",
            int(missing)
        )

        # ---------------------------------------------------------
        #                     Dataset Preview
        # ---------------------------------------------------------

        st.header("👀 Dataset Preview")

        st.dataframe(
            df.head(10),
            width="stretch"
        )

        # ---------------------------------------------------------
        #                  Dataset Information
        # ---------------------------------------------------------

        with st.expander("📋 Dataset Information", expanded=False):

            st.subheader("Dataset Summary")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Rows", len(df))

            with c2:
                st.metric("Columns", len(df.columns))

            with c3:
                memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
                st.metric("Memory Usage", f"{memory:.2f} MB")

            st.divider()

            st.subheader("Column Details")

            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isnull().sum().values,
                "Unique Values": df.nunique(dropna=True).values
            })

            st.dataframe(
                info_df,
                width="stretch",
                hide_index=True
            )

            st.divider()

            st.subheader("Data Types")

            dtype_counts = df.dtypes.astype(str).value_counts()

            st.dataframe(
                dtype_counts.rename_axis("Data Type")
                            .reset_index(name="Count"),
                hide_index=True,
                width="stretch"
            )

            st.divider()

            st.subheader("Missing Values")

            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing": df.isnull().sum().values,
                "Percentage (%)":
                    (
                        df.isnull().sum()
                        / len(df)
                        * 100
                    ).round(2).values
            })

            st.dataframe(
                missing_df,
                hide_index=True,
                width="stretch"
            )

            st.divider()

            st.subheader("Preview")

            st.dataframe(
                df.head(10),
                width="stretch"
            )


def render_configuration(df):
    """
    Render dataset configuration widgets.

    Returns
    -------
    target_col : str
    protected_col : str
    analyze : bool
    """

    st.header("⚙ Dataset Configuration")

    target_col = st.selectbox(
        "Target Column",
        options=df.columns,
        key="target_column"
    )

    protected_col = st.selectbox(
        "Protected Attribute",
        options=df.columns,
        key="protected_column"
    )

    analyze = st.button(
        "🚀 Analyze Bias",
        width="stretch",
        type="primary",
        key="analyze_button"
    )

    return target_col, protected_col, analyze

def render_ai_summary(ai_summary: str):
    """
    Display AI-generated fairness explanation.
    """

    st.header("🤖 AI Fairness Explanation")

    if not ai_summary:
        st.warning("AI explanation could not be generated.")
        return

    with st.expander("View AI Report", expanded=True):
        st.markdown(ai_summary)

def render_export_card(title:str, description:str, status:str, body_renderer:Callable):
    with st.container(border=True):

        st.subheader(title)

        st.write(description)

        st.markdown(f"🟢 **{status}**")

        st.write("")

        body_renderer()

def render_export_center(analysis, ai_summary, pdf_report):
    st.header("📦 Export Centre")

    # -----------------------------
    #         AI REPORT
    # -----------------------------

    # Converted once
    mitigated_df = analysis.mitigated_dataset.convert_to_dataframe()[0]

    # Preparing Export formats once
    mitigated_csv = mitigated_df.to_csv(index=False)
    mitigated_json = mitigated_df.to_json(orient="records", indent=4)

    def render_pdf_button():

        st.download_button(
            "⬇ Download PDF",
            data=pdf_report,
            file_name="AI_Fairness_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    render_export_card(
        title="📄 AI Fairness Report",
        description="Comprehensive fairness audit with AI explanation and fairness metrics.",
        status="Ready",
        body_renderer=render_pdf_button
    )

    st.write("")

    # -----------------------------
    #         DATASET
    # -----------------------------
    def render_dataset_buttons():

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "⬇ CSV",
                data=mitigated_csv,
                file_name="Mitigated_Dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:

            st.download_button(
                "⬇ JSON",
                data=mitigated_json,
                file_name="Mitigated_Dataset.json",
                mime="application/json",
                use_container_width=True
            )

    render_export_card(
        title="📁 Mitigated Dataset",
        description="Dataset after IBM AIF360 Reweighing.",
        status="Ready",
        body_renderer=render_dataset_buttons
    ) 

def render_downloads(analysis: AnalysisResult, ai_summary, pdf_report):
    render_export_center(analysis, ai_summary, pdf_report)