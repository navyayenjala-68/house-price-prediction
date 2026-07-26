import numpy as np
import streamlit as st

from utils import (
    apply_styles,
    load_model,
    load_training_data,
    render_sidebar,
)

# ==========================================================
# DEFAULT VALUES
# ==========================================================

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


def reset_form():
    for key in DEFAULT_VALUES:
        st.session_state.pop(key, None)


def typical_range(data, column):
    return (
        int(data[column].quantile(0.01)),
        int(data[column].quantile(0.99)),
    )


def show_range_notice(label, value, bounds):
    low, high = bounds
    if value < low or value > high:
        st.caption(
            f"Typical training range: {low:,} – {high:,}"
        )


# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="Property Valuation | SmartHome AI",
    page_icon="🏠",
    layout="wide",
)
apply_styles()
render_sidebar()
for key, value in DEFAULT_VALUES.items():
    st.session_state.setdefault(key, value)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
"""
<div class="predict-hero">

<div class="eyebrow">
PROPERTY VALUATION
</div>

<h1>
Estimate Residential Property Value
</h1>

<p>
Generate an estimated selling price using our trained Random Forest
Regression model built on the Ames Housing Dataset.
</p>

</div>
""",
unsafe_allow_html=True,
)

# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = load_model()
    training_data = load_training_data()

except Exception as exc:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(exc)

    st.stop()

neighborhoods = sorted(
    training_data["Neighborhood"].dropna().unique()
)

house_styles = sorted(
    training_data["HouseStyle"].dropna().unique()
)

if st.session_state.neighborhood not in neighborhoods:
    st.session_state.neighborhood = neighborhoods[0]

if st.session_state.house_style not in house_styles:
    st.session_state.house_style = house_styles[0]

# ==========================================================
# PROPERTY DETAILS
# ==========================================================

st.markdown(
"""
<div class="form-card">

<h2>🏡 Property Details</h2>

<p>
Fill in the property information below to estimate the market value.
</p>

""",
unsafe_allow_html=True,
)
left, right = st.columns(2)
st.markdown("</div></div>", unsafe_allow_html=True)

# ==========================================================
# LEFT COLUMN
# ==========================================================

with left:

    st.markdown("### Basic Information")

    neighborhood = st.selectbox(
        "Neighborhood",
        neighborhoods,
        key="neighborhood",
    )

    house_style = st.segmented_control(
        "House Style",
        house_styles,
        key="house_style",
    )

    overallqual = st.slider(
        "Overall Quality",
        1,
        10,
        key="overallqual",
    )

    grlivarea = st.number_input(
        "Above Ground Living Area (sq ft)",
        min_value=334,
        max_value=15000,
        step=25,
        key="grlivarea",
    )

    show_range_notice(
        "Living Area",
        grlivarea,
        typical_range(training_data, "GrLivArea"),
    )

    lotarea = st.number_input(
        "Lot Area (sq ft)",
        min_value=1300,
        max_value=250000,
        step=100,
        key="lotarea",
    )

    show_range_notice(
        "Lot Area",
        lotarea,
        typical_range(training_data, "LotArea"),
    )

# ==========================================================
# RIGHT COLUMN
# ==========================================================

with right:

    st.markdown("### Property Features")

    yearbuilt = st.number_input(
        "Year Built",
        min_value=1872,
        max_value=2010,
        step=1,
        key="yearbuilt",
    )

    overallcond = st.slider(
        "Overall Condition",
        1,
        10,
        key="overallcond",
    )

    fullbath = st.slider(
        "Full Bathrooms",
        0,
        4,
        key="fullbath",
    )

    totrms = st.slider(
        "Rooms Above Ground",
        2,
        15,
        key="totrms",
    )

    totalbsmtsf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=10000,
        step=25,
        key="totalbsmtsf",
    )

    show_range_notice(
        "Basement Area",
        totalbsmtsf,
        typical_range(training_data, "TotalBsmtSF"),
    )

    garagecars = st.slider(
        "Garage Capacity",
        0,
        4,
        key="garagecars",
    )

    garagearea = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=2000,
        step=25,
        key="garagearea",
    )

    show_range_notice(
        "Garage Area",
        garagearea,
        typical_range(training_data, "GarageArea"),
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# ACTION BUTTONS
# ==========================================================

col1, col2 = st.columns([4,1])

with col1:

    submitted = st.button(
        "🏠 Estimate Property Value",
        use_container_width=True,
        type="primary",
    )

with col2:

    st.button(
        "Reset",
        use_container_width=True,
        on_click=reset_form,
    )

st.markdown("<hr>", unsafe_allow_html=True)

if submitted:

    features = np.array([[
        overallqual,
        grlivarea,
        garagecars,
        garagearea,
        totalbsmtsf,
        fullbath,
        yearbuilt,
        totrms,
        lotarea,
        overallcond,
    ]])

    expected_features = getattr(
        model,
        "n_features_in_",
        features.shape[1]
    )

    if expected_features != features.shape[1]:

        st.error(
            "The saved model does not match the current prediction form."
        )

        st.stop()

    try:

        with st.spinner("Generating property valuation..."):

            prediction = float(
                model.predict(features)[0]
            )

    except Exception as exc:

        st.error("Prediction failed.")

        st.exception(exc)

        st.stop()

    # =====================================================
    # MARKET SEGMENT
    # =====================================================

    if prediction < 150000:

        segment = "Entry Level"

        colour = "#16A34A"

    elif prediction < 300000:

        segment = "Mid Market"

        colour = "#2563EB"

    else:

        segment = "Premium"

        colour = "#D97706"

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="result-card">

        <div style="font-size:14px;
        letter-spacing:1px;
        color:#64748B;
        font-weight:700;
        text-transform:uppercase;">

        Estimated Market Value
        Based on selected property characteristics

        </div>

        <div style="
        font-size:52px;
        font-weight:800;
        color:#183B63;
        margin-top:8px;
        ">

        ${:,.0f}

        </div>

        </div>
        """.format(prediction),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Estimated Segment",
            segment,
        )

    with c2:

        st.metric(
            "Model",
            "Random Forest",
        )

    with c3:

        st.metric(
            "Model Performance (R²)",
            "89.5%",
        )

    st.success(
        f"""
The predicted selling price for this property is approximately
**${prediction:,.0f}**.

This estimate was generated using a Random Forest Regression model trained on
the Ames Housing Dataset and should be used as an indicative market valuation,
not as a professional appraisal.
"""
    )

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("## About this estimate")

st.write(
"""
SmartHome AI predicts residential property values using the
Ames Housing Dataset and a trained Random Forest Regression model.

The estimate is based on ten important property characteristics
including:

• Overall Quality

• Living Area

• Lot Area

• Basement Area

• Garage Capacity

• Year Built

• Room Count

The prediction should be treated as an informed estimate rather
than an official property appraisal.
"""
)

st.info(
"💡 Tip: Try changing one feature at a time to understand how each property characteristic influences the estimated selling price."
)
