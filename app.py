import streamlit as st
from CORE.dataset_loader import render_upload
from UI.visualization import plot_dashboard
from CORE.llm_utils import get_ai_explanation
from CORE.analysis_pipeline import run_analysis
from UI.ui import (
    render_dataset_overview,
    render_configuration,
    render_ai_summary,
    render_downloads,
)
from CORE.report_generator import build_full_report, generate_pdf_report
from UI.theme import load_css
from app_state import initialize_state, update_status
from UI.sidebar import render_sidebar
from UI.tabs import (
    render_hero,
    render_workflow_progress,
    render_tab_empty_state,
    get_tab_labels,
)

st.set_page_config(
    page_title="FairSight — AI Bias Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
render_hero()

initialize_state()

for key, value in {
    "df": None,
    "analysis": None,
    "ai_summary": None,
    "pdf_report": None,
    "upload_file_id": None,
}.items():
    st.session_state.setdefault(key, value)

render_workflow_progress()

dataset_tab, dashboard_tab, ai_tab, export_tab = st.tabs(get_tab_labels())

# ── Dataset ──────────────────────────────────────────────────────────

with dataset_tab:
    df = render_upload()

    if df is not None:
        update_status("dataset_loaded")
        render_dataset_overview(df)
        target_col, protected_col, analyze = render_configuration(df)

        if analyze:
            update_status("dataset_configured")
            try:
                analysis = run_analysis(df, target_col, protected_col)
                st.session_state.analysis = analysis
                update_status("fairness_analysis_completed")
                st.success("✅ Fairness analysis completed — switch to the **Dashboard** tab to review results.")
            except Exception as e:
                st.error("❌ Analysis failed.")
                st.exception(e)
    else:
        render_tab_empty_state(
            icon="📂",
            title="Upload a dataset to begin",
            message="Drop a CSV or JSON file above to start your fairness audit. "
            "Common test datasets include Adult Income, COMPAS, and German Credit.",
            action_hint="Supported formats: .csv · .json",
        )

# ── Dashboard ────────────────────────────────────────────────────────

with dashboard_tab:
    if st.session_state.analysis is not None:
        plot_dashboard(st.session_state.analysis)
    else:
        render_tab_empty_state(
            icon="📊",
            title="Dashboard not ready yet",
            message="Configure your target and protected columns in the Dataset tab, "
            "then click Analyze Bias to generate fairness metrics and charts.",
            action_hint="← Go to the Dataset tab to run analysis",
            variant="info",
        )

# ── AI Report ────────────────────────────────────────────────────────

with ai_tab:

    if st.session_state.analysis is None:

        render_tab_empty_state(
            icon="🤖",
            title="AI report requires analysis",
            message="Complete the fairness analysis in the Dataset tab first.",
            action_hint="← Complete analysis in the Dataset tab",
            variant="info",
        )

    elif st.session_state.ai_summary is None:

        st.header("🤖 AI Fairness Report")

        st.write(
            "Generate a plain-language explanation of the fairness analysis "
            "using the local Ollama LLM."
        )

        if st.button(
            "🤖 Generate AI Report",
            type="primary",
            width="stretch"
        ):

            try:

                with st.spinner("Generating AI fairness explanation..."):

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

                    update_status("ai_report_generated")
                    update_status("export_ready")

                st.rerun()

            except Exception as e:

                st.error("❌ AI report generation failed.")
                st.exception(e)

    else:

        render_ai_summary(
            st.session_state.ai_summary
        )

# ── Export ───────────────────────────────────────────────────────────

with export_tab:

    analysis = st.session_state.analysis
    ai_summary = st.session_state.ai_summary
    pdf_report = st.session_state.pdf_report

    if (
        analysis is not None
        and ai_summary is not None
        and pdf_report is not None
    ):

        render_downloads(
            analysis,
            ai_summary,
            pdf_report,
        )

    else:

        missing = []

        if analysis is None:
            missing.append("fairness analysis")

        if ai_summary is None:
            missing.append("AI report")

        if pdf_report is None:
            missing.append("PDF report")

        st.info(
            "Complete the following before exporting: "
            + ", ".join(missing)
        )
