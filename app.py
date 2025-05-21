import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# --- CUSTOM CSS (Bersih & Profesional) ---
import streamlit as st

# CSS Styling
st.markdown("""
<style>
/* Latar belakang halaman */
body, .stApp {
    background-color: white !important;
    color: black !important;
}

/* Judul dan subjudul */
h1, h2, h3, h4 {
    color: black !important;
    font-weight: bold;
}

/* Semua input dan dropdown box */
input[type="text"], input[type="number"],
div[data-baseweb="select"] > div,
textarea {
    background-color: #e0f0ff !important; /* Biru muda */
    color: black !important;
    font-weight: 500;
    border-radius: 8px;
}

/* Label input */
label, .stTextInput > label, .stNumberInput > label, .stSelectbox > label {
    color: black !important;
    font-weight: bold;
}

/* Tombol */
.stButton > button {
    background-color: #e0f0ff !important;
    color: black !important;
    border: 2px solid #007bff !important;
    padding: 10px 24px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
}

/* Dropzone area upload file */
section[data-testid="stFileUploaderDropzone"] {
    background-color: #e0f0ff !important; /* Biru muda */
    color: black !important;
    border: 2px dashed #007bff !important;
    border-radius: 10px;
    padding: 20px;
}

/* Teks di dalam dropzone (termasuk "Drag and drop", limit, dll) */
section[data-testid="stFileUploaderDropzone"] p {
    color: black !important;
    font-weight: 500;
}

/* Tombol "Browse files" */
section[data-testid="stFileUploaderDropzone"] button {
    background-color: white !important;
    color: black !important;
    border: 2px solid #007bff !important;
    border-radius: 8px;
    font-weight: bold;
}

/* Pilihan radio dan label */
div[data-testid="stRadio"] label {
    color: black !important;
    font-weight: bold;
}

/* Pastikan teks dalam radio button tetap terlihat (tidak putih) */
div[data-testid="stRadio"] label {
    color: black !important;
    opacity: 1 !important;
}

/* Mengatasi teks yang tidak aktif */
div[data-testid="stRadio"] div[aria-disabled="true"] {
    color: black !important;
    opacity: 1 !important;
}

/* Mengatasi teks yang hilang pada elemen radio button aktif */
div[data-testid="stRadio"] div[role="radio"] {
    color: black !important;
    opacity: 1 !important;
}
            
/* ... semua kode CSS sebelumnya tetap sama ... */

/* RADIO BUTTON STYLING */
div[data-testid="stRadio"] label {
    color: black !important;
    font-weight: bold;
}
div[data-testid="stRadio"] label span {
    color: black !important;
}

/* MENGATASI TEKS RADIO BUTTON YANG TIDAK TERLIHAT JELAS */
div[data-testid="stRadio"] label {
    opacity: 1 !important;
}
div[data-testid="stRadio"] div[aria-disabled="true"] {
    color: black !important;
    opacity: 1 !important;
}

/* 🔴 TAMBAHKAN KODE BERIKUT UNTUK MENGUBAH WARNA HASIL PREDIKSI (DI DALAM st.success()) */
div[data-testid="stAlert"] p {
    color: black !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🚗 Used Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Estimate your car's price with Machine Learning!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent / 'data' / 'used_cars_cleaned.csv')

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / 'used_car_price_model_mae.pkl')

used_car = load_data()
model = load_model()
expected_columns = ['Gear_Type', 'Origin', 'Options', 'Type', 'Make', 'Region', 'Year', 'Engine_Size', 'Mileage']

# --- MODE SELECTION (Bukan Sidebar) ---
mode = st.radio("Pilih Mode Input:", ["🚗 🇮 🇳 🇵 🇺 🇹   🇮 🇳 🇫 🇴 🇷 🇲 🇦 🇸 🇮   🇲 🇴 🇧 🇮 🇱", "📄 🇺 🇵 🇱 🇴 🇦 🇩   🇫 🇮 🇱 🇪   🇧 🇦 🇹 🇨 🇭"], label_visibility="visible")
st.markdown("<br>", unsafe_allow_html=True)

# --- INPUT SATU MOBIL ---
if mode == "🚗 🇮 🇳 🇵 🇺 🇹   🇮 🇳 🇫 🇴 🇷 🇲 🇦 🇸 🇮   🇲 🇴 🇧 🇮 🇱":
    st.subheader("📋 Masukkan Informasi Mobil", divider="rainbow")

    def get_user_input(df):
        col1, col2 = st.columns(2)
        make_list = sorted(df['Make'].dropna().unique())
        origin_list = sorted(df['Origin'].dropna().unique())
        region_list = sorted(df['Region'].dropna().unique())
        gear_type_list = sorted(df['Gear_Type'].dropna().unique())
        option_list = sorted(df['Options'].dropna().unique())

        with col1:
            car_maker = st.selectbox("Manufacturer:", make_list)
            filtered_types = df[df['Make'] == car_maker]['Type'].dropna().unique()
            car_type = st.selectbox("Model/Type:", sorted(filtered_types))
            car_origin = st.selectbox("Origin/From:", origin_list)
            car_region = st.selectbox("Selling/Buying Region:", region_list)
            car_gear_type = st.selectbox("Gear Type:", gear_type_list)

        with col2:
            car_option = st.selectbox("Car's Option:", option_list)
            car_year = st.number_input("Year of Production:", value=2010, step=1, min_value=1960, max_value=2025)
            car_engine_size = st.number_input("Engine Size (L):", value=1.5, step=0.1, min_value=0.5, max_value=10.0)
            car_mileage = st.number_input("Mileage (km):", value=0, step=1000, min_value=0)

        return pd.DataFrame({
            'Gear_Type': [car_gear_type],
            'Origin': [car_origin],
            'Options': [car_option],
            'Type': [car_type],
            'Make': [car_maker],
            'Region': [car_region],
            'Year': [car_year],
            'Engine_Size': [car_engine_size],
            'Mileage': [car_mileage]
        })

    user_data = get_user_input(used_car)

    if st.button("🔮 Prediksi Harga"):
        try:
            user_data = user_data[expected_columns]
            price = round(model.predict(user_data)[0], 2)
            st.success(f"Estimasi Harga Mobil: SAR {price:,.2f}")
            st.toast("✅ Prediksi selesai!")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

# --- BATCH UPLOAD ---
elif mode == "📄 🇺 🇵 🇱 🇴 🇦 🇩   🇫 🇮 🇱 🇪   🇧 🇦 🇹 🇨 🇭":
    st.markdown("""
    <h3 style='display: flex; align-items: center; font-weight: bold;'>
        <img src='https://img.icons8.com/emoji/24/clipboard-emoji.png' style='margin-right: 8px;'/>
        Upload File (Batch)
    </h3>
    <hr style="height: 3px; border: none; background: linear-gradient(to right, red, orange, green, blue, indigo, violet); margin-top: -10px;"/>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 10px; color: black;">
    <strong>File harus berisi kolom berikut:</strong><br>
    <span style="background-color: #f0f0f0; padding: 5px; border-radius: 5px;">
    Gear_Type, Origin, Options, Type, Make, Region, Year, Engine_Size, Mileage
    </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄 Contoh Format CSV"):
        st.code("""Make,Type,Origin,Region,Gear_Type,Options,Year,Engine_Size,Mileage
Toyota,Corolla,Japan,Riyadh,Automatic,Full Option,2015,1.6,85000
Hyundai,Sonata,Korea,Jeddah,Manual,Mid Option,2018,2.0,60000
""", language="csv")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if not all(col in batch_df.columns for col in expected_columns):
                st.error("⚠️ File tidak memiliki semua kolom yang dibutuhkan.")
            else:
                batch_df = batch_df[expected_columns]
                predictions = model.predict(batch_df)
                batch_df["Predicted_Price"] = [round(p, 2) for p in predictions]
                st.success("✅ Prediksi berhasil!")
                st.dataframe(batch_df)
                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Hasil", data=csv, file_name="predicted_prices.csv", mime="text/csv")
                st.toast("✅ File berhasil diproses!")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Created by: Muhammad Umar Abdul Aziz</p>", unsafe_allow_html=True)
