import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def show_prediction_visualization():
    st.title("📈 Visualisasi Prediksi Total Service Time")

    # 🔍 Background Project
    st.markdown("""
    ### 🎯 Latar Belakang Proyek  
    Efisiensi waktu pengantaran merupakan salah satu faktor penting dalam layanan logistik makanan.  
    Berdasarkan eksplorasi data, ditemukan bahwa waktu layanan (Total Service Time) dipengaruhi oleh beberapa faktor seperti waktu persiapan, jarak tempuh, cuaca, dan tingkat kemacetan.  
    Untuk meningkatkan ketepatan estimasi waktu sampai (ETA), proyek ini membangun model prediksi berbasis **Linear Regression** yang mampu memproyeksikan durasi layanan secara real-time.  
    Model ini diharapkan membantu operasional logistik dalam mengoptimalkan rute, alokasi armada, serta memberikan transparansi waktu pengiriman kepada pelanggan.

    ---
    """)

    # Load model
    model = joblib.load("linear_regression_model.pkl")

    # Pilih dataset
    option = st.radio("Pilih Dataset", ["Training", "Testing"])
    x_path = "X_train_scaled.csv" if option == "Training" else "X_test_scaled.csv"
    y_path = "y_train.csv" if option == "Training" else "y_test.csv"

    try:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path).squeeze()  # Squeeze untuk Series
    except FileNotFoundError:
        st.error("❌ File X atau y tidak ditemukan.")
        return

    # Prediksi
    y_pred = model.predict(X)

    # Evaluasi
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Tampilkan metrik
    st.subheader("📊 Evaluasi Model Linear Regression")
    st.markdown(f"""
    - **MAE**: {mae:.2f} menit  
    - **MSE**: {mse:.2f}  
    - **RMSE**: {rmse:.2f}  
    - **R² Score**: {r2:.4f}
    """)

    st.markdown("""
    ---
    📘 **Penjelasan Evaluasi Model**  
    - **MAE**: Rata-rata selisih absolut antara nilai aktual dan prediksi.  
    - **MSE**: Rata-rata kuadrat selisih prediksi.  
    - **RMSE**: Akar dari MSE, menunjukkan besaran kesalahan dengan satuan yang sama dengan target.  
    - **R² Score**: Proporsi variansi yang bisa dijelaskan oleh model. Semakin mendekati 1, semakin baik.
    
    Hasil evaluasi menunjukkan bahwa **model Linear Regression** memberikan performa yang konsisten, akurat, dan mudah diinterpretasikan untuk kebutuhan operasional.
    ---
    """)

    # Scatter plot
    st.subheader("📌 Scatter Plot: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y, y=y_pred, alpha=0.7)
    plt.xlabel("Actual Total Service Time")
    plt.ylabel("Predicted Total Service Time")
    plt.title("Actual vs Predicted Total Service Time")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Prediction")
    plt.legend()
    st.pyplot(fig)

    st.markdown("""
    ---
    📘 **Interpretasi Scatter Plot**  
    Titik-titik yang mendekati garis merah putus-putus menunjukkan bahwa model memiliki presisi yang baik dalam memetakan hubungan antara fitur dan target.  
    ---
    """)

    # Tabel hasil
    st.subheader("📋 Tabel: Actual vs Predicted")
    result_df = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })
    st.dataframe(result_df)
