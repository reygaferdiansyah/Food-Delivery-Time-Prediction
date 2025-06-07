import streamlit as st
import pandas as pd
import joblib

def run_prediction_app():
    st.title("Prediksi Total Service Time")

    # Load model, scaler, dan kolom
    model = joblib.load("linear_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    saved_columns = joblib.load("saved_columns.pkl")

    # Kolom numerik yang akan di-scaling
    numerik_cols = [
        'Preparation_Time_min',
        'Courier_Experience_yrs',
        'Delivery_Time_min',
        'Distance_per_Minute',
        'Time_Efficiency',
        'Experience_Effectiveness'
    ]

    st.header("🔧 Input Fitur Numerik")
    prep_time = st.number_input("Preparation_Time_min", value=500.0)
    courier_exp = st.number_input("Courier_Experience_yrs", value=1.0)
    delivery_time = st.number_input("Delivery_Time_min", value=0.5)
    distance_per_min = st.number_input("Distance_per_Minute", value=0.1)
    time_eff = st.number_input("Time_Efficiency", value=0.0)
    exp_eff = st.number_input("Experience_Effectiveness", value=0.0)

    # ==========================================================
    # Tampilkan checkbox kategori secara vertikal
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

    # Bangun DataFrame dari input
    input_data = pd.DataFrame([{
        'Preparation_Time_min': prep_time,
        'Courier_Experience_yrs': courier_exp,
        'Delivery_Time_min': delivery_time,
        'Distance_per_Minute': distance_per_min,
        'Time_Efficiency': time_eff,
        'Experience_Effectiveness': exp_eff,
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

    # Reindex kolom sesuai yang dipakai saat training model
    input_data = input_data.reindex(columns=saved_columns, fill_value=0)

    # Scaling fitur numerik
    input_data[numerik_cols] = scaler.transform(input_data[numerik_cols])

    # Tombol Prediksi
    if st.button("🚚 Prediksi Total Service Time"):
        prediction = model.predict(input_data)
        st.success(f"⏱️ Prediksi Total Service Time: {prediction[0]:.2f} menit")

        # Tampilkan tabel input final yang dipakai model
        st.subheader("📋 Data yang Digunakan untuk Prediksi")
        st.dataframe(input_data)
