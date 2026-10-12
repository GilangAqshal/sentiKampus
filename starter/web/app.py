import streamlit as st

st.set_page_config(page_title="SentiKampus Starter")
st.title("SentiKampus — Starter Web")
text = st.text_area("Masukan mahasiswa")

if st.button("Analisis Sentimen"):
    # TODO P4: kirim POST JSON ke FastAPI /api/v1/predict.
    # Jangan menjalankan model langsung dari UI; pertahankan pemisahan frontend-backend.
    st.info(f"TODO: kirim teks ke API: {text!r}")
