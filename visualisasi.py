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
    Untuk meningkatkan ketepatan estimasi waktu sampai (ETA), proyek ini membangun model prediksi berbasis **Lasso Regression** yang mampu memproyeksikan durasi layanan secara real-time.  
    Model ini diharapkan membantu operasional logistik dalam mengoptimalkan rute, alokasi armada, serta memberikan transparansi waktu pengiriman kepada pelanggan.

    ---
    """)

    # Load model
    model = joblib.load("lasso_model.pkl")

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
    st.subheader("📊 Evaluasi Model Lasso Regression")
    st.markdown(f"""
    - **MAE**: {mae:.2f} menit  
    - **MSE**: {mse:.2f}  
    - **RMSE**: {rmse:.2f}  
    - **R² Score**: {r2:.4f}
    """)

    st.markdown("""
    ---
    📘 **Interpretasi Hasil Evaluasi Model Lasso Regression (Data Training)**

    - **MAE (Mean Absolute Error)** = 6.53 menit  
    Rata-rata kesalahan absolut antara nilai aktual dan hasil prediksi adalah sekitar 6.53 menit. Ini berarti model cenderung meleset sekitar 6 hingga 7 menit dalam memperkirakan waktu layanan aktual.

    - **MSE (Mean Squared Error)** = 105.99  
    MSE mengkuadratkan error, sehingga memberikan penalti lebih besar terhadap kesalahan besar. Nilai ini menunjukkan masih ada beberapa prediksi yang cukup jauh dari nilai aktual.

    - **RMSE (Root Mean Squared Error)** = 10.30  
    Dengan satuan yang sama dengan target (menit), RMSE memberi gambaran bahwa prediksi model kadang bisa meleset sekitar 10 menit.

    - **R² Score (Koefisien Determinasi)** = 0.7750  
    Nilai ini mengindikasikan bahwa sekitar 77.5% variasi dalam `Delivery_Time_min` dapat dijelaskan oleh fitur yang digunakan dalam model. Semakin mendekati 1, semakin baik model dalam menangkap pola data.

    ---

    📌 **Kesimpulan**  
    Model Lasso menunjukkan performa yang cukup baik pada data training. Dengan kombinasi error yang rendah dan R² yang mendekati 0.8, model ini layak digunakan untuk estimasi awal waktu layanan pengiriman makanan, namun tetap disarankan untuk dievaluasi lebih lanjut pada data testing untuk validasi generalisasi model.
    """)


    st.markdown("""
    ---
    📘 **Interpretasi Hasil Evaluasi Model Lasso Regression (Data Testing)**

    - **MAE (Mean Absolute Error)** = 6.32 menit  
    Rata-rata kesalahan absolut antara hasil prediksi dan nilai aktual pada data testing adalah sekitar 6.32 menit. Ini menunjukkan tingkat kesalahan prediksi yang relatif rendah di luar data pelatihan.

    - **MSE (Mean Squared Error)** = 123.75  
    Nilai MSE yang cukup rendah menunjukkan bahwa model mampu menjaga kesalahan besar tetap minimal meskipun terdapat data baru yang belum pernah dilihat sebelumnya.

    - **RMSE (Root Mean Squared Error)** = 11.12  
    Dengan satuan menit, nilai RMSE mengindikasikan deviasi prediksi terhadap nilai aktual berada di kisaran 11 menit.

    - **R² Score (Koefisien Determinasi)** = 0.7745  
    Sekitar 77.45% variasi dalam variabel target (`Delivery_Time_min`) pada data testing berhasil dijelaskan oleh model. Ini menunjukkan bahwa model memiliki kemampuan generalisasi yang cukup baik.

    ---

    📌 **Kesimpulan**  
    Evaluasi pada data testing menunjukkan bahwa model Lasso mempertahankan performa prediksi yang stabil di luar data pelatihan. Dengan error yang tetap rendah dan nilai R² yang cukup tinggi, model ini cocok digunakan untuk memprediksi waktu layanan dalam skenario nyata.
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
