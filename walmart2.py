#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# Load trained models
sales_model = joblib.load("xgboost_sales_forecast_model.pkl")
demand_model = joblib.load("gradient_boosting_demand_classifier.pkl")
meta_model = joblib.load("meta_demand_classifier.pkl")

# Demand level mapping
demand_map = {0: "Low 📉", 1: "Medium 📊", 2: "High 📈"}

# Custom CSS styling for a fancy layout
def set_css():
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #6495ED;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        color: #003366;
    }
    .sub-header {
        font-size: 22px;
        text-align: center;
        color: #2f4f4f;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #0056b3;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-weight: bold;
    }
    .result-box {
        background-color: #87CEFA;
        border-left: 5px solid #0056b3;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        font-size: 20px;
    }
    .image-box {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

set_css()

# Navigation
page = st.sidebar.selectbox("Navigate", ["\U0001F3E0 Home", "\U0001F4C8 Predict Weekly Sales"])

# Home Page
if page == "🏠 Home":
    st.markdown("<div class='main-header'>Walmart Weekly Sales Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Powered by Stacked Models: XGBoost + Meta-Learning</div>", unsafe_allow_html=True)

    # Image Placeholder
    st.markdown("<div class='image-box'>", unsafe_allow_html=True)
    st.image(r"assets/Walmart.jpg", width=1000)  
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; font-size: 18px; margin-top: 30px;'>
        This smart AI system predicts <strong>Weekly Sales</strong> and classifies <strong>Demand Level</strong> for Walmart stores<br>
        using a stacked architecture combining regression, classification, and meta-learning models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Click the button below to begin forecasting.</div>", unsafe_allow_html=True)

    if st.button("Start Prediction ✨"):
        st.session_state["page"] = "predict"

# Prediction Page
if page == "📈 Predict Weekly Sales" or st.session_state.get("page") == "predict":
    st.markdown("<div class='main-header'>Predict Weekly Sales</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("\U0001f3ec Store ID", min_value=1)
        month = st.number_input("\U0001f4c5 Month", min_value=1, max_value=12)
        holiday_flag = st.selectbox("\U0001f389 Holiday Flag", [0, 1])
        temperature = st.slider("\U0001f321️ Temperature (°F)", 0.0, 120.0, 65.0)
        fuel_price = st.slider("⛽ Fuel Price ($)", 1.0, 5.0, 2.5)

    with col2:
        cpi = st.number_input("\U0001f4c8 Consumer Price Index (CPI)", value=210.0)
        unemployment = st.number_input("\U0001f4c9 Unemployment Rate (%)", value=8.0)
        rolling_mean_4 = st.number_input("\U0001f4c8 Rolling Mean (4 Weeks) [$]", value=1_500_000.0)
        season = st.radio("\U0001f4c6 Season", ["Spring 🌸", "Summer ☀️", "Winter ❄️"])
        day_of_week = st.selectbox("\U0001f4c6 Day of Week (0=Mon, 6=Sun)", list(range(7)))

    season_spring = season_summer = season_winter = 0
    if "Spring" in season:
        season_spring = 1
    elif "Summer" in season:
        season_summer = 1
    elif "Winter" in season:
        season_winter = 1

    is_weekend = 1 if day_of_week in [5, 6] else 0

    if st.button("🚀 Predict"):
        input_data = np.array([[store, holiday_flag, temperature, fuel_price, cpi, unemployment,
                                month, day_of_week, is_weekend, rolling_mean_4,
                                season_spring, season_summer, season_winter]])

        sales_pred = sales_model.predict(input_data)[0]
        demand_pred = demand_model.predict(input_data)[0]

        def bucket_sales(y):
            return np.digitize([y], [669926.827, 1279637.663])[0]

        sales_bucket = bucket_sales(sales_pred)
        meta_input = np.append(input_data, [sales_pred, demand_pred, sales_bucket]).reshape(1, -1)

        corrected_demand_class = meta_model.predict(meta_input)[0]
        final_demand_label = demand_map[corrected_demand_class]

        st.markdown(f"""
        <div class="result-box">
            📊 <strong>Predicted Weekly Sales:</strong> <span style="color:green;">${sales_pred:,.2f}</span><br><br>
            📦 <strong>Final Demand Level:</strong> <span style="color:#003366;">{final_demand_label}</span>
        </div>
        """, unsafe_allow_html=True)
# streamlit run walmart2.py