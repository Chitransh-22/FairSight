import streamlit as st
import pandas as pd
import json
from models import AnalysisResult

def render_sidebar():
    """
    Render the application sidebar.
    """

    with st.sidebar:

        st.title("⚖ AI Fairness Auditor")

        st.markdown("---")

        st.markdown(
            """
Detect, analyze and mitigate bias in machine learning datasets
using **IBM AIF360**, **Reweighing**, and **Local LLM-powered**
fairness explanations.
"""
        )

        st.markdown("---")

        st.subheader("🛠 Technology Stack")

        st.markdown(
            """
- 🐍 Python
- 📊 Streamlit
- ⚖ IBM AIF360
- 🤖 Ollama + Qwen
- 🧠 LangChain
- 📈 Plotly
- 🐼 Pandas
"""
        )

        st.markdown("---")

        st.subheader("📂 Supported Formats")

        st.markdown(
            """
- CSV (.csv)
- JSON (.json)
"""
        )

        st.markdown("---")

        st.subheader("📋 Workflow")

        st.markdown(
            """
1. Upload Dataset
2. Configure Analysis
3. Detect Bias
4. Mitigate Bias
5. Visualize Results
6. Generate AI Report
7. Download Results
"""
        )

        st.markdown("---")

        st.caption("Version 1.0")



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


def render_downloads(analysis: AnalysisResult, ai_summary, pdf_report):
    """
    Render download buttons for fairness analysis.
    """

    st.header("📥 Download Results")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # Download Mitigated Dataset
    # --------------------------------------------------

    with col1:

        mitigated_df = analysis.mitigated_dataset.convert_to_dataframe()[0]

        csv = mitigated_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Mitigated Dataset",
            data=csv,
            file_name="mitigated_dataset.csv",
            mime="text/csv",
            width="stretch"
        )

    # --------------------------------------------------
    # Download Metrics JSON
    # --------------------------------------------------

    with col2:

        metrics = {
            "context": analysis.context,
            "group_info": analysis.group_info,
            "metrics_before": analysis.metrics_before,
            "metrics_after": analysis.metrics_after,
            "rates_before": analysis.rates_before,
            "rates_after": analysis.rates_after,
        }

        st.download_button(
            label="⬇ Metrics JSON",
            data=json.dumps(metrics, indent=4,default=str),
            file_name="fairness_metrics.json",
            mime="application/json",
            width="stretch"
        )

    # --------------------------------------------------
    # Download AI Report
    # --------------------------------------------------

    with col3:

        st.download_button(
            label="📄 Download AI Report (PDF)",
            data=pdf_report,
            file_name="AI_Fairness_Report.pdf",
            mime="application/pdf"
        )