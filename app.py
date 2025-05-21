from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 16px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🚗 USED-CAR PRICE PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Estimate your car's price with Machine Learning!</p>", unsafe_allow_html=True)

# --- IMAGE ---
image_path = Path(__file__).parent / 'used-car-all-brand-sell--613.jpg'
if image_path.exists():
    st.image(str(image_path), use_column_width=True)

st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent / 'data' / 'used_car_cleaned.csv')

used_car = load_data()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / 'used_car_price_model_mae.pkl')

model = load_model()

# --- USER INPUT FUNCTION ---
def get_user_input(df):
    col1, col2 = st.columns(2)

    # Dropdown hanya dari data pelatihan
    make_list = sorted(df['Make'].dropna().unique())
    origin_list = sorted(df['Origin'].dropna().unique())
    region_list = sorted(df['Region'].dropna().unique())
    gear_type_list = sorted(df['Gear_Type'].dropna().unique())
    option_list = sorted(df['Options'].dropna().unique())

    with col1:
        car_maker = st.selectbox("Manufacturer:", make_list)

        # Filter Type berdasarkan pilihan Make
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

    # Urutan kolom disesuaikan dengan ColumnTransformer
    user_data = pd.DataFrame({
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

    return user_data

# --- GET USER INPUT ---
user_data = get_user_input(used_car)

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Price"):
    try:
        # DEBUGGING: cek kolom dan tipe data input
        st.write("User input columns:", user_data.columns.tolist())
        st.write("User input types:", user_data.dtypes)

        # Prediksi harga
        price = round(model.predict(user_data)[0], 2)
        formatted_price = "{:,.2f}".format(price)

        st.markdown(f"""
        <div style='padding:20px; background-color:#d4edda; border:1px solid #c3e6cb; border-radius:10px; text-align:center;'>
            <h2>Estimated Car Price:</h2>
            <p style='font-size:24px; font-weight:bold; color:#155724;'>SAR {formatted_price}</p>
            <p>This is an estimate using a Machine Learning model.</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Created by: Risdan Kristori</p>", unsafe_allow_html=True)
