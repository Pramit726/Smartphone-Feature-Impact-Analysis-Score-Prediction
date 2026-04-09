import streamlit as st
from pages_content import interpret, predict_rating, rating_predictor_desc


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Smartprix Rating Predictor", "Predict Rating", "Interpret"]
    )

    if page == "Smartprix Rating Predictor":
        rating_predictor_desc.rating_predictor_desc()

    elif page == "Predict Rating":
        predict_rating.predict_rating()

    elif page == "Interpret":
        interpret.interpret()


if __name__ == "__main__":
    main()
