import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils import apply_styles, load_model, load_training_data, render_sidebar


st.set_page_config(page_title="Analytics | SmartHome AI", page_icon="S", layout="wide")
apply_styles()
render_sidebar()

try:
    df = load_training_data()
    model = load_model()
except Exception as exc:
    st.error("The analytics data could not be loaded. Check the data and model files, then try again.")
    st.exception(exc)
    st.stop()

st.markdown(
"""
<p class="eyebrow">
MARKET ANALYTICS
</p>

<h1 class="page-title">
Ames Housing Dashboard
</h1>

<p class="page-description">
Explore historical housing trends, understand property characteristics,
and discover which factors influence residential prices the most.
</p>
""",
unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("Properties", f"{len(df):,}")

with c2:
    st.metric(
        "Average Price",
        f"${df['SalePrice'].mean():,.0f}"
    )

with c3:
    st.metric(
        "Median Price",
        f"${df['SalePrice'].median():,.0f}"
    )

with c4:
    st.metric(
        "Price Range",
        f"${df['SalePrice'].min()/1000:.0f}K - ${df['SalePrice'].max()/1000:.0f}K"
    )
st.markdown("<hr>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.subheader("Distribution of Residential Sale Prices")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["SalePrice"], bins=30, color="#0f8b8d", edgecolor="white")
    ax.set_xlabel("Sale price (USD)")
    ax.set_ylabel("Number of properties")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)
with right:
    st.subheader("Features Most Associated with Sale Price")
    correlations = df.corr(numeric_only=True)["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(10)
    st.bar_chart(correlations, color="#0f8b8d", horizontal=True)
    st.caption("Absolute correlation shows the strength of a linear relationship; it does not prove causation.")

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Correlation Between Key Housing Features")
heatmap_features = ["SalePrice", *correlations.head(8).index.tolist()]
heatmap_data = df[heatmap_features].corr()
fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(heatmap_data, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(heatmap_features)), heatmap_features, rotation=40, ha="right")
ax.set_yticks(range(len(heatmap_features)), heatmap_features)
for row in range(len(heatmap_features)):
    for column in range(len(heatmap_features)):
        value = heatmap_data.iloc[row, column]
        text_color = "white" if abs(value) > .55 else "#172238"
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Pearson correlation")
ax.set_title("Top numeric relationships with sale price", pad=14)
st.pyplot(fig, clear_figure=True)
st.caption("Values closer to 1 or -1 indicate stronger linear relationships. This view is descriptive and does not establish causation.")

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Interactive Price Analysis")
available_features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "GarageCars", "FullBath", "LotArea"]
selected_feature = st.selectbox("Compare sale price with", available_features, index=1)
plot_data = df[[selected_feature, "SalePrice"]].dropna()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.scatter(plot_data[selected_feature], plot_data["SalePrice"], alpha=.45, color="#183b63", edgecolors="none")
ax.set_xlabel(selected_feature)
ax.set_ylabel("Sale price (USD)")
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, clear_figure=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Random Forest Feature Importance")
feature_names = ["Overall quality", "Living area", "Garage capacity", "Garage area", "Basement area", "Full bathrooms", "Year built", "Rooms above ground", "Lot area", "Overall condition"]
model_importance = getattr(model, "feature_importances_", None)
if model_importance is not None and len(model_importance) == len(feature_names):
    importance = pd.DataFrame({"Feature": feature_names, "Importance": model_importance}).sort_values("Importance", ascending=False)
    st.bar_chart(importance.set_index("Feature"), color="#183b63", horizontal=True)
    st.caption("Feature importance reflects how often and how effectively an input helps the trained Random Forest reduce prediction error. It is not a measure of causality.")
else:
    st.info("Feature importance is not available for the currently loaded model.")
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
"""
<div class="insight-card">

<h3>Key Insights</h3>

<ul>

<li>Overall Quality has the strongest influence on selling price.</li>

<li>Larger living areas generally correspond to higher property values.</li>

<li>Garage capacity and basement area contribute positively to price.</li>

<li>Most homes in the Ames dataset are priced between $120K and $220K.</li>

<li>The Random Forest model captures complex relationships between property features.</li>

</ul>

</div>
""",
unsafe_allow_html=True,
)
with st.expander("Preview Training Dataset"):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
