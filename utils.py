"""Shared helpers for the Streamlit application."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent


def asset_path(*parts: str) -> Path:
    """Return a project-relative path that also works from Streamlit pages."""
    return PROJECT_ROOT.joinpath(*parts)


def apply_styles() -> None:
    """Apply the shared stylesheet when it is available."""
    css_path = asset_path("style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

import streamlit as st

def render_sidebar():

    st.sidebar.markdown(
        """
        <div class="brand">
            <span class="brand-mark">S</span>
            <span class="brand-name">SmartHome AI</span>

            <p class="brand-copy">
            A data-informed estimate for Ames residential properties.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    st.sidebar.page_link(
        "Home.py",
        label="🏠 Home"
    )

    st.sidebar.page_link(
        "pages/Analytics.py",
        label="📊 Analytics"
    )

    st.sidebar.page_link(
        "pages/Predict.py",
        label="🏡 Predict"
    )

    st.sidebar.page_link(
        "pages/About.py",
        label="ℹ️ About"
    )


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Load and cache the fitted estimator for the current Streamlit session."""
    return joblib.load(asset_path("models", "house_price_model.pkl"))


@st.cache_data(show_spinner="Loading housing data...")
def load_training_data() -> pd.DataFrame:
    """Load and cache the Ames training data used by the dashboard."""
    return pd.read_csv(asset_path("data", "train.csv"))
