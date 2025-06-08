import streamlit as st
import pandas as pd
import joblib

def run_prediction_app():
    st.title("Prediksi Total Service Time")

    # Load model, scaler, dan kolom
    model = joblib.load("lasso_model.pkl")
    scaler = joblib.load("scaler.pkl")
    saved_columns = joblib.load("saved_columns.pkl")

    # Kolom numerik untuk scaling
    numerik_cols = [
        'Distance_km',
        'Preparation_Time_min',
        'Courier_Experience_yrs',
        'Max_Allowed_Time'
    ]

    # =========================== Input =============================
    st.header("🔧 Input Fitur Numerik")
    distance = st.number_input("Distance_km", value=2.0)
    prep_time = st.number_input("Preparation_Time_min", value=300.0)
    courier_exp = st.number_input("Courier_Experience_yrs", value=1.0)
    max_time = st.number_input("Max_Allowed_Time", value=500.0)

    st.header("☁️ Kondisi Cuaca")
    weather_clear = st.checkbox("Weather_Clear")
    weather_foggy = st.checkbox("Weather_Foggy")
    weather_rainy = st.checkbox("Weather_Rainy")
    weather_snowy = st.checkbox("Weather_Snowy")
    weather_windy = st.checkbox("Weather_Windy")

    st.header("🚦 Traffic Level")
    traffic_high = st.checkbox("Traffic_Level_High")
    traffic_low = st.checkbox("Traffic_Level_Low")
    traffic_medium = st.checkbox("Traffic_Level_Medium")

    st.header("🕒 Time of Day")
    tod_afternoon = st.checkbox("Time_of_Day_Afternoon")
    tod_evening = st.checkbox("Time_of_Day_Evening")
    tod_morning = st.checkbox("Time_of_Day_Morning")
    tod_night = st.checkbox("Time_of_Day_Night")

    st.header("🛵 Jenis Kendaraan")
    veh_bike = st.checkbox("Vehicle_Type_Bike")
    veh_car = st.checkbox("Vehicle_Type_Car")
    veh_scooter = st.checkbox("Vehicle_Type_Scooter")

    # =========================== DataFrame =============================
    input_data = pd.DataFrame([{
        'Distance_km': distance,
        'Preparation_Time_min': prep_time,
        'Courier_Experience_yrs': courier_exp,
        'Max_Allowed_Time': max_time,
        'Weather_Clear': weather_clear,
        'Weather_Foggy': weather_foggy,
        'Weather_Rainy': weather_rainy,
        'Weather_Snowy': weather_snowy,
        'Weather_Windy': weather_windy,
        'Traffic_Level_High': traffic_high,
        'Traffic_Level_Low': traffic_low,
        'Traffic_Level_Medium': traffic_medium,
        'Time_of_Day_Afternoon': tod_afternoon,
        'Time_of_Day_Evening': tod_evening,
        'Time_of_Day_Morning': tod_morning,
        'Time_of_Day_Night': tod_night,
        'Vehicle_Type_Bike': veh_bike,
        'Vehicle_Type_Car': veh_car,
        'Vehicle_Type_Scooter': veh_scooter
    }])

    # Reindex dan scaling
    input_data = input_data.reindex(columns=saved_columns, fill_value=0)
    input_data[numerik_cols] = scaler.transform(input_data[numerik_cols])

    # =========================== Prediksi =============================
    if st.button("🚚 Prediksi Total Service Time"):
        prediction = model.predict(input_data)
        st.success(f"⏱️ Prediksi Total Service Time: {prediction[0]:.2f} menit")

        # Tampilkan data input akhir
        st.subheader("📋 Data yang Digunakan untuk Prediksi")
        st.dataframe(input_data)
