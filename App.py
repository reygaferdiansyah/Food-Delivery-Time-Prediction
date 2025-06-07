import streamlit as st

# ====================================
# 🛠️ Konfigurasi Halaman
# ====================================
st.set_page_config(page_title="Portfolio", layout="wide", page_icon=":rocket:")
st.title("Portfolio Saya")
st.header("Data Scientist & Developer")

# ====================================
# 📚 Sidebar Navigasi
# ====================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", [
    "Tentang Saya",
    "Model Machine Learning",
    "Predict Time",
    "Kontak"
])


# ====================================
# 🔄 Routing Halaman
# ====================================
if page == "Tentang Saya":
    import About_me
    About_me.tampilkan_tentang_saya()

elif page == "Kontak":
    import kontak
    kontak.tampilkan_kontak()

elif page == "Model Machine Learning":
    import visualisasi
    visualisasi.show_prediction_visualization()

elif page == "Predict Time":
    import prediksi
    prediksi.run_prediction_app()

