# prop_app.py - Streamlit Deployment for House Price Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Load model and feature order
model, feature_order = joblib.load('price_model.pkl')

# ✅ Manual encodings (must match training mappings)
property_type_map = {'Flat': 0, 'House': 1, 'Upper Portion': 2, 'Lower Portion': 3, 'Farm House': 4}
purpose_map = {'For Rent': 0, 'For Sale': 1}
province_map = {'Islamabad Capital': 0, 'Punjab': 1, 'Sindh': 2, 'Khyber Pakhtunkhwa': 3, 'Balochistan': 4}
city_map = {'Islamabad': 0, 'Lahore': 1, 'Karachi': 2, 'Peshawar': 3, 'Quetta': 4}
location_map = {'G-10': 0, 'E-11': 1, 'G-15': 2, 'Bani Gala': 3, 'DHA Valley': 4}

# ✅ Area conversion function
def convert_to_marla(size, unit):
    unit = unit.strip().lower()
    try:
        size = float(size)
    except:
        return None

    if unit == "marla":
        return size
    elif unit == "kanal":
        return size * 20
    elif unit in ["sq. yards", "square yards"]:
        return size / 30.25
    elif unit in ["sq. ft.", "square feet"]:
        return size / 225
    return None

# ✅ Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction (Pakistan)")

st.header("Enter Property Details:")

property_type = st.selectbox("Property Type", list(property_type_map.keys()))
purpose = st.selectbox("Purpose", list(purpose_map.keys()))
province = st.selectbox("Province", list(province_map.keys()))
city = st.selectbox("City", list(city_map.keys()))
location = st.selectbox("Location", list(location_map.keys()))

latitude = st.number_input("Latitude", value=33.7)
longitude = st.number_input("Longitude", value=73.0)

col1, col2 = st.columns(2)
with col1:
    baths = st.slider("Bathrooms", 1, 10, 2)
with col2:
    bedrooms = st.slider("Bedrooms", 1, 10, 3)

st.subheader("Area Information")
area_size = st.number_input("Area Size", value=5.0)
area_unit = st.selectbox("Area Unit", ["Marla", "Kanal", "Sq. Yards", "Sq. Ft."])

st.subheader("Date Property Was Listed")
year_added = st.number_input("Year Added", min_value=2015, max_value=2025, value=2023)
month_added = st.slider("Month", 1, 12, 6)
day_added = st.slider("Day", 1, 31, 15)

# ✅ Convert and encode input
total_area = convert_to_marla(area_size, area_unit)

if total_area is None:
    st.error("⚠️ Invalid area size/unit.")
else:
    encoded_data = {
        'property_type': property_type_map[property_type],
        'location': location_map[location],
        'city': city_map[city],
        'province_name': province_map[province],
        'purpose': purpose_map[purpose],
        'latitude': latitude,
        'longitude': longitude,
        'baths': baths,
        'bedrooms': bedrooms,
        'year_added': year_added,
        'month_added': month_added,
        'day_added': day_added,
        'total_area_marla': total_area
    }

    input_df = pd.DataFrame([encoded_data])
    input_df = input_df[feature_order]  # ✅ Ensure feature order matches training

    # ✅ Predict
    log_price = model.predict(input_df)[0]
    actual_price = np.expm1(log_price)  # reverse log1p

    st.success(f"💰 **Estimated Price:** Rs. {actual_price:,.0f}")
