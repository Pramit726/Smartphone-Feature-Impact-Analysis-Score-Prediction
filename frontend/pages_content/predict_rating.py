import requests
import streamlit as st
from config import API_BASE_URL
from pages_content import metrics

PREDICT_URL = f"{API_BASE_URL}/ratings/predict"
CATEGORIES_URL = f"{API_BASE_URL}/ratings/categories"


def get_categories():
    """Fetches categories from the API and stores them in session state."""
    if "categories" not in st.session_state:
        try:
            response = requests.get(CATEGORIES_URL)
            response.raise_for_status()
            st.session_state.categories = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching categories: {e}")
            st.session_state.categories = None
    return st.session_state.categories


def predict_rating():

    st.title("🔮 Predict Smartphone Rating")

    # Load categories only once
    categories = get_categories()
    if not categories:
        st.stop()

    brand_categories = categories.get("brand_categories", [])
    processor_categories = categories.get("processor_categories", [])
    resolution_categories = categories.get("resolution_categories", [])
    primary_camera_rear_categories = categories.get(
        "primary_camera_rear_categories", []
    )

    # Display categories
    price = st.number_input(
        "💰 Price (in ₹)", value=30000, min_value=3999, max_value=300000
    )
    brand_name = st.selectbox("🏷️ Brand Name", brand_categories)
    has_5g = st.checkbox("📶 Supports 5G")
    has_nfc = st.checkbox("📡 NFC Enabled")
    has_ir_blaster = st.checkbox("🎛️ IR Blaster Available")
    num_cores = st.number_input(
        "💾 Processor Cores", value=8.0, min_value=1.0, max_value=16.0, step=1.0
    )
    processor_speed = st.number_input(
        "⚡ Processor Speed (GHz)", value=2.0, min_value=1.0, max_value=5.0, step=0.1
    )
    processor_brand = st.selectbox("🔧 Processor Brand", processor_categories)
    ram_capacity = st.number_input(
        "🎮 RAM (GB)", value=8.0, min_value=1.0, max_value=24.0, step=1.0
    )
    internal_memory = st.number_input(
        "💽 Internal Storage (GB)",
        value=128.0,
        min_value=16.0,
        max_value=1024.0,
        step=16.0,
    )
    fast_charging_available = st.checkbox("⚡ Fast Charging Available?")
    fast_charging_available = 1 if fast_charging_available else 0

    if fast_charging_available == 1:
        fast_charging = st.number_input(
            "⚡ Fast Charging (Watts)",
            min_value=1.0,
            max_value=240.0,
            step=5.0,
            value=33.0,
        )  # Changed min_value to 1.0
    else:
        fast_charging = None

    resolution = st.selectbox("🖥️ Resolution", resolution_categories)
    primary_camera_rear = st.selectbox(
        "📸 Primary Rear Camera (MP)", primary_camera_rear_categories
    )
    primary_camera_front = st.number_input(
        "🤳 Primary Front Camera (MP)",
        value=16.0,
        min_value=2.0,
        max_value=64.0,
        step=1.0,
    )

    screen_size = st.number_input(
        "📱 Screen Size (inches)", value=6.5, min_value=4.0, max_value=8.0, step=0.1
    )
    refresh_rate = st.number_input(
        "🔄 Refresh Rate (Hz)", value=90, min_value=60, max_value=240, step=1
    )
    num_rear_cameras = st.number_input(
        "📷 Rear Cameras", value=3, min_value=1, max_value=5, step=1
    )
    num_front_cameras = st.number_input(
        "🤳 Front Cameras", value=1, min_value=1, max_value=2, step=1
    )

    extended_memory_available = st.checkbox("💾 Extended Memory Available?")
    extended_memory_available = 1 if extended_memory_available else 0

    if extended_memory_available == 1:
        extended_upto = st.number_input(
            "💾 Extended Memory (GB)",
            min_value=16.0,
            max_value=2048.0,
            step=64.0,
            value=256.0,
        )  # Example values
    else:
        extended_upto = None

    if st.button("🚀 Predict Rating"):
        payload = {
            "price": price,
            "brand_name": brand_name,
            "has_5g": has_5g,
            "has_nfc": has_nfc,
            "has_ir_blaster": has_ir_blaster,
            "num_cores": num_cores,
            "processor_speed": processor_speed,
            "processor_brand": processor_brand,
            "ram_capacity": ram_capacity,
            "internal_memory": internal_memory,
            "fast_charging": fast_charging,
            "screen_size": screen_size,
            "resolution": resolution,
            "refresh_rate": refresh_rate,
            "num_rear_cameras": num_rear_cameras,
            "num_front_cameras": num_front_cameras,
            "primary_camera_rear": primary_camera_rear,
            "primary_camera_front": primary_camera_front,
            "fast_charging_available": fast_charging_available,
            "extended_memory_available": extended_memory_available,
            "extended_upto": extended_upto,
        }

        try:
            response = requests.post(PREDICT_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            st.session_state.last_prediction = payload
            rating = round(float(result["rating"]))

            # Color coding based on rating (1 to 100)
            if rating < 30:
                color = "red"
            elif rating < 60:
                color = "orange"
            elif rating < 80:
                color = "gold"
            else:
                color = "green"

            st.markdown(
                f'<p style="color:{color}; font-size: 24px;">⭐ <b>Predicted Rating: {rating}</b></p>',
                unsafe_allow_html=True,
            )
            st.session_state.last_prediction = payload

            model_metrics()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error during prediction: {e}")
        except KeyError:
            st.error("❌ Error: Rating missing from prediction result.")


def model_metrics():
    metrics.show_metrics()
