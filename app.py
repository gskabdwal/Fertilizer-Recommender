import joblib
import numpy as np
import streamlit as st

# Load the trained Random Forest model and LabelEncoder
model = joblib.load("fertilizer_recommendation_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("🌱 Fertilizer Recommendation System")
st.write("Enter soil and environmental conditions to get the best fertilizer recommendation.")

# Input fields for user data
temperature = st.number_input("🌡 Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
moisture = st.number_input("💦 Moisture Level (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
soil_type = st.selectbox("🌍 Soil Type", options=[0, 1, 2, 3, 4])  # Modify based on encoding
crop_type = st.selectbox("🌾 Crop Type", options=[0, 1, 2, 3, 4])  # Modify based on encoding
nitrogen = st.number_input("🧪 Nitrogen Level", min_value=0, max_value=100, value=40, step=1)
potassium = st.number_input("🧂 Potassium Level", min_value=0, max_value=100, value=20, step=1)
phosphorous = st.number_input("🔬 Phosphorous Level", min_value=0, max_value=100, value=30, step=1)

# Predict function
if st.button("🔍 Recommend Fertilizer"):
    # Prepare input array
    input_data = np.array([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]])
    
    # Predict
    prediction = model.predict(input_data)
    recommended_fertilizer = label_encoder.inverse_transform([prediction[0]])[0]
    
    # Display result
    st.success(f"✅ Recommended Fertilizer: **{recommended_fertilizer}**")
