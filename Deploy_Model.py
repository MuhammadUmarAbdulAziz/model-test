import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("used_car_price_model_mae.pkl")

# Judul Aplikasi
st.title("Prediksi Harga Mobil Bekas")

# Input dari pengguna
st.header("Masukkan informasi mobil")
brand = st.selectbox("Merek Mobil", ['Toyota', 'Honda', 'Suzuki', 'Daihatsu', 'BMW'])  # contoh
model_name = st.text_input("Model Mobil")
year = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, value=2015)
transmission = st.selectbox("Transmisi", ['Automatic', 'Manual'])
mileage = st.number_input("Jarak Tempuh (dalam KM)", min_value=0, value=50000)
fuel_type = st.selectbox("Jenis Bahan Bakar", ['Bensin', 'Diesel'])

# Button prediksi
if st.button("Prediksi"):
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [model_name],
        'year': [year],
        'transmission': [transmission],
        'mileage': [mileage],
        'fuel_type': [fuel_type]
    })

    # Prediksi harga
    prediction = model.predict(input_data)
    st.success(f"Perkiraan harga mobil bekas: Rp {int(prediction[0]):,}")