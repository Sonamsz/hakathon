import streamlit as st
import pandas as pd
import joblib
import pickle

st.set_page_config(
    page_title="Agriculture crop prediction",
    page_icon=":seedling:",
    layout="wide"
)

@st.cache_resource
def load_model():
    model=joblib.load(
        "agricultural_production_model.pkl"
    )
    return model

model = load_model()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🌾 Agricultural Farm Production Predictor")

st.write(
    """
    This application predicts agricultural farm production
    based on farm characteristics such as crop type, farm area,
    rainfall, fertilizer usage, labor, region, and irrigation.
    """
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Farm Information")

region = st.sidebar.selectbox(
    "Region",
    [
        "Bumthang",
        "Chhukha",
        "Dagana",
        "Gasa",
        "Haa",
        "Lhuentse",
        "Mongar",
        "Paro",
        "Pema Gatshel",
        "Punakha",
        "Samdrup Jongkhar",
        "Samtse",
        "Sarpang",
        "Thimphu",
        "Trashigang",
        "Trashiyangtse",
        "Trongsa",
        "Tsirang",
        "Wangdue Phodrang",
        "Zhemgang"
    ]
)


crop = st.sidebar.selectbox(
    "Crop",
    [
        "Rice",
        "Maize",
        "Wheat",
        "Potato",
        "Vegetables"
    ]
)


farm_area = st.sidebar.number_input(
    "Farm Area (Acres)",
    min_value=0.1,
    max_value=1000.0,
    value=5.0,
    step=0.1
)


rainfall = st.sidebar.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=1000.0,
    step=10.0
)


fertilizer = st.sidebar.number_input(
    "Fertilizer (kg)",
    min_value=0.0,
    max_value=5000.0,
    value=100.0,
    step=5.0
)


labor_days = st.sidebar.number_input(
    "Labor Days",
    min_value=1,
    max_value=1000,
    value=50,
    step=1
)


irrigation = st.sidebar.selectbox(
    "Irrigation",
    [
        "Yes",
        "No"
    ]
)


# ---------------------------------------------------
# INPUT DATA
# ---------------------------------------------------

input_data = pd.DataFrame({
    "Region": [region],
    "Crop": [crop],
    "Farm_Area_Acres": [farm_area],
    "Rainfall_mm": [rainfall],
    "Fertilizer_kg": [fertilizer],
    "Labor_Days": [labor_days],
    "Irrigation": [irrigation]
})


# ---------------------------------------------------
# DISPLAY INPUT
# ---------------------------------------------------

st.subheader("Farm Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Region",
        region
    )

with col2:
    st.metric(
        "Crop",
        crop
    )

with col3:
    st.metric(
        "Farm Area",
        f"{farm_area:.2f} acres"
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Rainfall",
        f"{rainfall:.0f} mm"
    )

with col5:
    st.metric(
        "Fertilizer",
        f"{fertilizer:.0f} kg"
    )

with col6:
    st.metric(
        "Labor",
        f"{labor_days} days"
    )


# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

st.divider()

if st.button(
    "🌾 Predict Production",
    use_container_width=True
):

    prediction = model.predict(
        input_data
    )

    predicted_value = prediction[0]

    st.success(
        "Prediction completed successfully!"
    )

    st.subheader(
        "Predicted Agricultural Production"
    )

    st.metric(
        "Production",
        f"{predicted_value:,.2f} kg"
    )


# ---------------------------------------------------
# SHOW INPUT DATA
# ---------------------------------------------------

with st.expander(
    "View Input Data"
):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ---------------------------------------------------
# INFORMATION
# ---------------------------------------------------

st.divider()

st.subheader("About the Model")

st.write(
    """
    The application uses a Random Forest Regression model
    trained on agricultural farm production data.

    The model considers:

    • Region
    • Crop
    • Farm Area
    • Rainfall
    • Fertilizer Usage
    • Labor Days
    • Irrigation

    The output represents the predicted agricultural
    production in kilograms.
    """
)