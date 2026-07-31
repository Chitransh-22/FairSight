import streamlit as st


DEFAULT_STATUS = {
    "dataset_loaded": False,
    "configuration_complete": False,
    "analysis_complete": False,
    "report_generated": False,
    "export_ready": False,
}


def initialize_state():
    """Initialize application state."""

    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = DEFAULT_STATUS.copy()


def update_status(stage, value=True):
    """Update a specific analysis stage."""

    if "analysis_status" not in st.session_state:
        initialize_state()

    st.session_state.analysis_status[stage] = value


def get_status():
    """Return the current analysis status."""

    if "analysis_status" not in st.session_state:
        initialize_state()

    return st.session_state.analysis_status


def reset_status():
    """Reset all analysis stages."""

    st.session_state.analysis_status = DEFAULT_STATUS.copy()