
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# Load the trained AI model
saved = joblib.load("crop_model.pkl")
model = saved["model"]
features = saved["features"]

st.title("🌱 AI-Based Crop Recommendation System")
st.write(
    "Find suitable crops based on your soil and "
    "weather conditions."
)

st.info(
    "Enter your soil test and local weather values. "
    "The results are suggestions, not a guarantee of yield."
)

with st.form("crop_form"):

    st.subheader("🧪 Soil Nutrients")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input(
            "Nitrogen (N)", min_value=0.0,
            max_value=300.0, value=90.0
        )

    with col2:
        P = st.number_input(
            "Phosphorus (P)", min_value=0.0,
            max_value=300.0, value=42.0
        )

    with col3:
        K = st.number_input(
            "Potassium (K)", min_value=0.0,
            max_value=300.0, value=43.0
        )

    st.subheader("🌦️ Weather and Soil pH")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=60.0,
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0,
        value=80.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0, max_value=14.0,
        value=6.5
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0, max_value=2000.0,
        value=200.0
    )

    submitted = st.form_submit_button(
        "🌾 Recommend Crops",
        use_container_width=True
    )

if submitted:

    values = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])[features]

    probabilities = model.predict_proba(values)[0]
    top_indices = probabilities.argsort()[::-1][:3]

    st.subheader("Your Crop Recommendations")

    for rank, index in enumerate(top_indices, start=1):
        crop = model.classes_[index]
        score = probabilities[index] * 100

        st.markdown(f"### {rank}. {crop.title()}")
        st.progress(float(probabilities[index]))
        st.write(f"Model score: {score:.2f}%")

    st.caption(
        "The scores describe the model's predictions. "
        "They do not measure actual yield or profitability. "
        "Confirm crop suitability using local agricultural "
        "advice and field conditions."
    )
