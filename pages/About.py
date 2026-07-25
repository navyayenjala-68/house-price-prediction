import streamlit as st

from utils import apply_styles, render_sidebar


st.set_page_config(page_title="About | SmartHome AI", page_icon="S", layout="wide")
apply_styles()
render_sidebar()

st.markdown("<p class='eyebrow'>About the project</p>", unsafe_allow_html=True)
st.title("A focused property-valuation prototype")
st.markdown("<p class='section-intro'>SmartHome AI demonstrates an end-to-end machine-learning workflow, from housing data exploration to an accessible valuation interface.</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown("## The data")
    st.write("The application uses the Ames Housing dataset, a detailed collection of 1,460 residential property sales in Ames, Iowa. The sale price is the target variable.")
    st.markdown("## The model")
    st.write("A Random Forest regression model estimates sale price from ten practical features, including quality, living area, year built, garage capacity, and basement area. The training data covers homes built through 2010; estimates for later construction years are extrapolations.")
with right:
    st.markdown("## The workflow")
    st.markdown("1. Prepare and explore housing data  \n2. Select practical valuation features  \n3. Train and evaluate candidate models  \n4. Present the selected model in Streamlit")
    st.markdown("## Intended use")
    st.write("This prototype is designed for learning, portfolio demonstration, and initial price exploration. It is not a substitute for a licensed appraisal, inspection, or current local market analysis.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## Technology")
technology = st.columns(3)
technology[0].markdown("<div class='feature-card'><h3>Python</h3><p>Pandas and NumPy support data preparation and numerical processing.</p></div>", unsafe_allow_html=True)
technology[1].markdown("<div class='feature-card'><h3>scikit-learn</h3><p>Random Forest regression powers the trained price-estimation model.</p></div>", unsafe_allow_html=True)
technology[2].markdown("<div class='feature-card'><h3>Streamlit</h3><p>The interface makes model inputs, outputs, and data exploration accessible.</p></div>", unsafe_allow_html=True)

st.markdown("<p class='disclaimer'>SmartHome AI &bull; Ames Housing dataset &bull; Machine-learning internship project</p>", unsafe_allow_html=True)
