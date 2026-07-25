import numpy as np
import streamlit as st

from utils import apply_styles, load_model, load_training_data, render_sidebar


DEFAULT_VALUES = {
    "neighborhood": "NAmes",
    "house_style": "1Story",
    "overallqual": 5,
    "grlivarea": 1500,
    "lotarea": 9000,
    "yearbuilt": 2005,
    "overallcond": 5,
    "fullbath": 2,
    "totrms": 7,
    "totalbsmtsf": 900,
    "garagecars": 2,
    "garagearea": 500,
}


def reset_form() -> None:
    """Restore every property input to its initial value."""
    for key in DEFAULT_VALUES:
        st.session_state.pop(key, None)


def typical_range(data, column: str) -> tuple[int, int]:
    """Return a practical 1st-99th percentile range for an input field."""
    return int(data[column].quantile(0.01)), int(data[column].quantile(0.99))


def show_range_notice(label: str, value: int, bounds: tuple[int, int]) -> None:
    """Surface a light-touch warning for unusual, but accepted, input values."""
    lower, upper = bounds
    if value < lower or value > upper:
        st.caption(f"Note: {label} is outside the typical training range ({lower:,}-{upper:,}).")


st.set_page_config(page_title="Create an Estimate | SmartHome AI", page_icon="S", layout="wide")
apply_styles()
render_sidebar()

for key, value in DEFAULT_VALUES.items():
    st.session_state.setdefault(key, value)

st.markdown("<p class='eyebrow'>Property valuation</p>", unsafe_allow_html=True)
st.title("Create an estimate")
st.markdown("<p class='section-intro'>Provide core property details to generate an Ames Housing price estimate. Field guidance updates as you work.</p>", unsafe_allow_html=True)

try:
    model = load_model()
    training_data = load_training_data()
except Exception as exc:
    st.error("The valuation model or training data could not be loaded. Confirm that the project files are available and try again.")
    st.exception(exc)
    st.stop()

neighborhoods = sorted(training_data["Neighborhood"].dropna().unique())
house_styles = sorted(training_data["HouseStyle"].dropna().unique())
if st.session_state.neighborhood not in neighborhoods:
    st.session_state.neighborhood = neighborhoods[0]
if st.session_state.house_style not in house_styles:
    st.session_state.house_style = house_styles[0]

st.markdown("<div class='valuation-form'>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown("#### Location and home")
    neighborhood = st.selectbox("Neighborhood", neighborhoods, key="neighborhood", help="Ames neighborhood recorded for this property scenario.")
    house_style = st.segmented_control("Property style", house_styles, key="house_style", help="Property style from the Ames Housing dataset.")
    st.caption("Location and style are captured for context. The current saved model uses the ten numeric inputs below; retraining is required before these fields can influence the estimate.")

    overallqual = st.slider("Overall quality", min_value=1, max_value=10, key="overallqual", help="Material and finish quality, from 1 (very poor) to 10 (excellent).")
    st.caption(f"Selected quality: **{overallqual}/10**")
    grlivarea = st.number_input("Above-ground living area (sq ft)", min_value=334, max_value=15000, step=25, key="grlivarea")
    show_range_notice("Living area", grlivarea, typical_range(training_data, "GrLivArea"))
    lotarea = st.number_input("Lot area (sq ft)", min_value=1300, max_value=250000, step=100, key="lotarea")
    show_range_notice("Lot area", lotarea, typical_range(training_data, "LotArea"))
    yearbuilt = st.number_input(
    "Year Built",
    min_value=1872,
    max_value=2010,
    step=1,
    key="yearbuilt",
    help="Construction year of the property."
)
overallcond = st.slider(
    "Overall Condition",
    min_value=1,
    max_value=10,
    key="overallcond"
)
st.caption(f"Selected condition: **{overallcond}/10**")
with right:
    st.markdown("#### Rooms and parking")
    fullbath = st.slider("Full bathrooms", min_value=0, max_value=4, key="fullbath")
    st.caption(f"Selected bathrooms: **{fullbath}**")
    totrms = st.slider("Rooms above ground", min_value=2, max_value=15, key="totrms")
    st.caption(f"Selected rooms: **{totrms}**")
    totalbsmtsf = st.number_input("Total basement area (sq ft)", min_value=0, max_value=10000, step=25, key="totalbsmtsf")
    show_range_notice("Basement area", totalbsmtsf, typical_range(training_data, "TotalBsmtSF"))
    garagecars = st.slider("Garage capacity (cars)", min_value=0, max_value=4, key="garagecars")
    st.caption(f"Selected garage capacity: **{garagecars} cars**")
    garagearea = st.number_input("Garage area (sq ft)", min_value=0, max_value=2000, step=25, key="garagearea")
    show_range_notice("Garage area", garagearea, typical_range(training_data, "GarageArea"))

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='sticky-action-bar'>", unsafe_allow_html=True)
generate_column, reset_column = st.columns([3, 1])
with generate_column:
    submitted = st.button("Generate estimate", type="primary", use_container_width=True)
with reset_column:
    st.button("Reset to default", use_container_width=True, on_click=reset_form)
st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    features = np.array([[
        overallqual, grlivarea, garagecars, garagearea, totalbsmtsf,
        fullbath, yearbuilt, totrms, lotarea, overallcond,
    ]])

    expected_features = getattr(model, "n_features_in_", features.shape[1])
    if expected_features != features.shape[1]:
        st.error("The current model does not match this valuation form.")
        st.stop()

    try:
        with st.spinner("Reviewing property details and generating an estimate..."):
            prediction = float(model.predict(features)[0])
    except Exception as exc:
        st.error("The estimate could not be generated.")
        st.exception(exc)
        st.stop()

    if prediction < 150000:
        segment = "Entry range"
    elif prediction < 300000:
        segment = "Mid range"
    else:
        segment = "Premium range"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='eyebrow'>Estimated Sale Price</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Estimated Value", f"${prediction:,.0f}")

    with col2:
        st.metric("Value Brand", segment)

    with col3:
        st.metric("Model Accuracy (R²)", "89.5%")

    st.info(
        "This estimated value was generated using a Random Forest Regression model trained on the Ames Housing Dataset."
        "The prediction is intended as an approximate market valuation based on the selected property characteristics and should be used for informational purposes only."
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### How to use this result")
st.write(
    "Compare several plausible property scenarios, then review the Analytics page "
    "to understand the sale-price distribution and the inputs that most influence the model."
)