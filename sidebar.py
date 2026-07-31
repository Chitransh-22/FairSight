import streamlit as st
from app_state import get_status


def render_project_info():
    st.sidebar.title("⚖ AI Fairness Auditor")

    st.sidebar.caption("Version 1.0")

    st.sidebar.info(
        """
Detect, analyze and mitigate bias in
machine learning datasets using IBM
AIF360 and Local LLMs.
"""
    )


def render_workflow():

    st.sidebar.subheader("🔄 Workflow")

    workflow = [
        "Upload Dataset",
        "Validate",
        "Preprocess",
        "Encode",
        "Fairness Analysis",
        "Bias Mitigation",
        "AI Explanation",
        "Export Report"
    ]

    for step in workflow:
        st.sidebar.write(f"• {step}")


def render_analysis_status():

    status = get_status()

    st.sidebar.subheader("📊 Analysis Status")

    stages = [
        ("dataset_loaded", "Dataset Loaded"),
        ("configuration_complete", "Configuration Complete"),
        ("analysis_complete", "Analysis Complete"),
        ("report_generated", "Report Generated"),
        ("export_ready", "Ready to Export"),
    ]

    for key, label in stages:

        if status[key]:
            st.sidebar.success(f"✅ {label}")

        else:
            st.sidebar.warning(f"⏳ {label}")


def render_ai_engine():

    st.sidebar.subheader("🤖 AI Engine")

    st.sidebar.write("Model: Gemma")
    st.sidebar.write("Framework: LangChain")
    st.sidebar.write("Inference: Ollama")


def render_resources():

    st.sidebar.subheader("📚 Resources")

    st.sidebar.markdown("- GitHub")
    st.sidebar.markdown("- Documentation")
    st.sidebar.markdown("- About")


def render_sidebar():

    render_project_info()

    st.sidebar.divider()

    render_analysis_status()

    st.sidebar.divider()

    render_workflow()

    st.sidebar.divider()

    render_ai_engine()

    st.sidebar.divider()

    render_resources()