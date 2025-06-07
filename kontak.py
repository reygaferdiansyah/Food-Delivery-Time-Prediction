import streamlit as st

def tampilkan_kontak():
    st.title("📬 Kontak Saya")

    st.markdown("""
    Silakan hubungi saya atau lihat profil saya melalui link berikut:

    - 💼 [LinkedIn](https://www.linkedin.com/in/reyga-ferdiansyah)  
    - 🛠️ [GitHub](https://github.com/reygaferdiansyah)  
    """)

    st.markdown("---")
    st.subheader("📨 Kirim Pesan Langsung")

    with st.form("form_kontak"):
        nama = st.text_input("Nama")
        email = st.text_input("Email")
        pesan = st.text_area("Pesan")

        submit = st.form_submit_button("Kirim")

        if submit:
            st.success("Terima kasih! Pesan Anda sudah terkirim.")
