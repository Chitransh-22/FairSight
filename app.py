import streamlit as st
from CORE.dataset_loader import render_upload
from UI.visualization import plot_dashboard
from CORE.llm_utils import get_ai_explanation
from CORE.analysis_pipeline import run_analysis
from UI.ui import (render_dataset_overview,render_configuration, 
                   render_ai_summary, render_downloads)
from CORE.report_generator import (build_full_report, generate_pdf_report)
from UI.theme import load_css
from app_state import (initialize_state, update_status)
from UI.sidebar import render_sidebar

st.set_page_config(
    page_title="AI Bias Detection System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

st.title("⚖️ AI Bias Detection System")

st.markdown("""
Detect, analyze and mitigate bias in machine learning datasets using
AIF360, Reweighing and Local LLM-powered fairness explanations.
""")

st.divider()

initialize_state()

defaults = {

    "df": None,
    "analysis": None,
    "ai_summary": None,
    "pdf_report": None,

}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

"""

                    RUN ANALYSIS
                         │
                         ▼
                  run_analysis()
                         │
                         ▼
                session_state.analysis
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Dashboard   AI Report   Export
              │          │          │
              │          ▼          │
              │        Ollama       │
              │          │          │
              ▼          ▼          ▼
           Results    Explanation   Files

"""

df = render_upload()

if df is not None:

    update_status("dataset_loaded")

    dataset_tab, dashboard_tab, ai_tab, export_tab = st.tabs(
        [
            "📂 Dataset",
            "📊 Dashboard",
            "🤖 AI Report",
            "📦 Export"
        ]
    )

    # =====================================================
    # DATASET TAB
    # =====================================================

    with dataset_tab:

        render_dataset_overview(df)

        target_col, protected_col, analyze = render_configuration(df)

        if analyze:

            update_status("dataset_configured")

            try:

                analysis = run_analysis(
                    df,
                    target_col,
                    protected_col
                )

                st.session_state.analysis = analysis

                update_status(
                    "fairness_analysis_completed"
                )

                st.success(
                    "✅ Fairness analysis completed successfully."
                )

            except Exception as e:

                st.error("❌ Analysis failed.")

                st.exception(e)


    # =====================================================
    # DASHBOARD TAB
    # =====================================================

    with dashboard_tab:

        if st.session_state.analysis is not None:

            plot_dashboard(
                st.session_state.analysis
            )

        else:

            st.info(
                "👈 Run the analysis from the Dataset tab "
                "to view the fairness dashboard."
            )


    # =====================================================
    # AI REPORT TAB
    # =====================================================

    with ai_tab:

        if st.session_state.analysis is None:

            st.info(
                "👈 Run the analysis first."
            )

        else:

            if st.session_state.ai_summary is None:

                st.info(
                    "The fairness analysis is ready. "
                    "Generate the AI explanation below."
                )

                generate_ai = st.button(
                    "🤖 Generate AI Report",
                    type="primary",
                    width="stretch"
                )

                if generate_ai:

                    try:

                        llm_response = get_ai_explanation(
                            st.session_state.analysis
                        )

                        ai_summary = build_full_report(
                            st.session_state.analysis,
                            llm_response
                        )

                        pdf_report = generate_pdf_report(
                            ai_summary
                        )

                        st.session_state.ai_summary = ai_summary

                        st.session_state.pdf_report = pdf_report

                        update_status(
                            "ai_report_generated"
                        )

                        update_status(
                            "export_ready"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            "❌ AI report generation failed."
                        )

                        st.exception(e)

            else:

                render_ai_summary(
                    st.session_state.ai_summary
                )


    # =====================================================
    # EXPORT TAB
    # =====================================================

    with export_tab:

        if (
            st.session_state.analysis is not None
            and st.session_state.ai_summary is not None
            and st.session_state.pdf_report is not None
        ):

            render_downloads(
                st.session_state.analysis,
                st.session_state.ai_summary,
                st.session_state.pdf_report
            )

        else:

            st.info(
                "👈 Complete the analysis and AI report "
                "before exporting."
            )


render_sidebar()