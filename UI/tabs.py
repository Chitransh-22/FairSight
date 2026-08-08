"""Tab workflow UI — progress stepper, dynamic labels, and empty states."""

import streamlit as st
from app_state import get_status, reset_status


WORKFLOW_STEPS = [
    ("dataset_loaded", "Upload", "📂"),
    ("fairness_analysis_completed", "Analyze", "📊"),
    ("ai_report_generated", "AI Report", "🤖"),
    ("export_ready", "Export", "📦"),
]


def _step_complete(status: dict, step_key: str) -> bool:
    return bool(status.get(step_key))


def get_tab_labels() -> list[str]:
    """Return tab labels with completion badges based on pipeline status."""
    status = get_status()
    analysis = st.session_state.get("analysis")
    ai_summary = st.session_state.get("ai_summary")
    pdf_report = st.session_state.get("pdf_report")

    dataset_badge = " ✓" if status.get("fairness_analysis_completed") else ""
    dashboard_badge = " ✓" if analysis is not None else ""
    ai_badge = " ✓" if ai_summary is not None else ""
    export_badge = " ✓" if pdf_report is not None else ""

    return [
        f"📂 Dataset{dataset_badge}",
        f"📊 Dashboard{dashboard_badge}",
        f"🤖 AI Report{ai_badge}",
        f"📦 Export{export_badge}",
    ]


def render_hero() -> None:
    """Compact hero banner for the main content area."""
    st.markdown(
        """
        <div class="fs-hero">
            <div class="fs-hero-content">
                <span class="fs-hero-icon">⚖️</span>
                <div>
                    <h1 class="fs-hero-title">FairSight</h1>
                    <p class="fs-hero-subtitle">
                        Detect, analyze, and mitigate algorithmic bias —
                        powered by AIF360, Reweighing, and local LLM explanations.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_progress() -> None:
    """Horizontal stepper showing pipeline progress above the tabs."""
    status = get_status()

    steps_html = []
    completed_count = 0

    for i, (key, label, icon) in enumerate(WORKFLOW_STEPS):
        done = _step_complete(status, key)
        if done:
            completed_count += 1

        state = "completed" if done else "pending"
        if not done and completed_count == i:
            state = "active"

        steps_html.append(
            f"""
            <div class="fs-step fs-step-{state}">
                <div class="fs-step-circle">
                    {"✓" if done else str(i + 1)}
                </div>
                <span class="fs-step-icon">{icon}</span>
                <span class="fs-step-label">{label}</span>
            </div>
            """
        )

        if i < len(WORKFLOW_STEPS) - 1:
            line_done = done and _step_complete(status, WORKFLOW_STEPS[i + 1][0])
            line_class = "fs-step-line-done" if done else ""
            steps_html.append(f'<div class="fs-step-line {line_class}"></div>')

    st.markdown(
        f'<div class="fs-workflow">{"".join(steps_html)}</div>',
        unsafe_allow_html=True,
    )


def render_tab_empty_state(
    icon: str,
    title: str,
    message: str,
    action_hint: str | None = None,
    variant: str = "info",
) -> None:
    """Rich empty-state card for locked or incomplete tabs."""
    hint_html = ""
    if action_hint:
        hint_html = f'<p class="fs-empty-action">{action_hint}</p>'

    st.markdown(
        f"""
        <div class="fs-empty-state fs-empty-{variant}">
            <div class="fs-empty-icon">{icon}</div>
            <h3 class="fs-empty-title">{title}</h3>
            <p class="fs-empty-message">{message}</p>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_pipeline_state() -> None:
    """Clear analysis outputs and pipeline status."""
    reset_status()
    st.session_state.analysis = None
    st.session_state.ai_summary = None
    st.session_state.pdf_report = None


def handle_new_upload(uploaded_file) -> None:
    """Reset pipeline when a different file is uploaded."""
    if uploaded_file is None:
        return

    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    previous_id = st.session_state.get("upload_file_id")

    if previous_id is not None and previous_id != file_id:
        reset_pipeline_state()

    st.session_state.upload_file_id = file_id
