import requests
import streamlit as st
from config import API_BASE_URL


def main():
    st.set_page_config(page_title="Home", layout="wide")

    # Check API availability
    try:
        response = requests.get(f"{API_BASE_URL}/home")
        response.raise_for_status()
        st.session_state.api_available = True
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.session_state.api_available = False
        st.error(
            "Oops! It seems our app is temporarily unavailable. Please try again later."
        )
        st.warning("Technical details: " + str(e))
        st.stop()
    except Exception as e:
        st.session_state.api_available = False
        st.error("Uh oh! Something unexpected went wrong. We're looking into it!")
        st.warning("Technical details: " + str(e))
        st.stop()

    # Hero Section
    st.title("📱 Understand & Predict Smartprix Ratings")
    st.subheader(
        "Gain insights into how smartphone features impact specification scores and optimize your designs."
    )

    # Call-to-Action (CTA) Button
    if st.session_state.api_available:
        if st.button("🚀 Start Exploring"):
            st.switch_page("pages/Ratings.py")

    st.markdown("---")

    if st.session_state.api_available:
        st.header("📊 Why Do Smartprix Ratings Matter?")
        for point in data["points"]:
            st.write(f"**{point}**")

        st.info(f"💡 {data['info_message']}")

    st.markdown("---")

    # Navigation Panel
    st.subheader("🔍 Explore Features")
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.api_available:
            if st.button("📊 Explore Ratings"):
                st.switch_page("pages/Ratings.py")

    with col2:
        st.button("📱 Compare Phones (Coming Soon 🚀)", disabled=True)


if __name__ == "__main__":
    main()
