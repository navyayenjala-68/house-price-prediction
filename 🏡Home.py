import streamlit as st

from utils import apply_styles, asset_path, render_sidebar


st.set_page_config(
    page_title="SmartHome AI | Property Valuation",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()
render_sidebar()

st.markdown("<p class='eyebrow'>Ames Housing Explorer</p>", unsafe_allow_html=True)
st.markdown(
    """
    <section class="hero">
        <h1>Understand a home's potential value.</h1>
        <h3>A simple, data-informed property valuation experience.</h3>
        <p>SmartHome AI uses a trained machine-learning model and the Ames Housing dataset to turn key property details into an estimated sale price.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

metrics = st.columns(3)
metrics[0].metric("Properties analysed", "1,460")
metrics[1].metric("Property attributes", "79")
metrics[2].metric("Valuation inputs", "10")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## Built for clear decisions")
st.markdown("<p class='section-intro'>Explore the data, then create an estimate using a focused set of property characteristics.</p>", unsafe_allow_html=True)

cards = st.columns(3)
card_content = [
    ("Focused valuation", "Enter the property details that have the strongest practical relationship with sale price."),
    ("Transparent analysis", "Review sale-price patterns and the model's feature importance in the Analytics page."),
    ("Practical output", "Receive a clear price estimate and value band to support an initial conversation or comparison."),
]
for column, (title, text) in zip(cards, card_content):
    column.markdown(f"<div class='feature-card'><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
left, right = st.columns([1.25, 1])
with left:
    st.markdown("## Start with the right context")
    st.write(
        "Use **Predict** to generate an estimate for an Ames-style property. Use **Analytics** to examine the training data and understand which characteristics matter most."
    )
    st.caption("Estimates are decision-support tools, not professional appraisals or guarantees of sale price.")
with right:
    st.image(str(asset_path("assets", "house.jpg")), use_container_width=True)

st.markdown("<p class='disclaimer'>Model: Random Forest regression &bull; Data source: Ames Housing dataset &bull; Built with Streamlit and scikit-learn</p>", unsafe_allow_html=True)
