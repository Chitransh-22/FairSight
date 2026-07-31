from pathlib import Path
import streamlit as st

def load_css():
    css_path = Path("styles/theme.css")

    with open(css_path) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )