import base64

import requests
import streamlit as st
import streamlit.components.v1 as components
from config import API_BASE_URL

INTERPRET_URL = f"{API_BASE_URL}/ratings/interpret"


def interpret():
    st.title("🔍 Interpret Prediction")
    if "last_prediction" not in st.session_state:
        st.warning("⚠️ Please make a prediction first.")
    else:
        if st.button("📊 Interpret Last Prediction"):
            with st.spinner("Interpreting prediction..."):
                try:
                    # Check if interpretation results are already in session state
                    if "interpret_result" not in st.session_state:
                        interpret_response = requests.get(
                            INTERPRET_URL, json=st.session_state.last_prediction
                        )
                        interpret_response.raise_for_status()
                        st.session_state.interpret_result = interpret_response.json()

                    interpret_result = st.session_state.interpret_result

                    if "force_plot" in interpret_result:
                        st.subheader("📌 Force Plot")
                        st.write(
                            "See the individual feature contributions that led to the model's predicted rating."
                        )
                        components.html(
                            interpret_result["force_plot"], height=200, scrolling=True
                        )
                    else:
                        st.warning("Force Plot not available.")

                    if "summary_plot" in interpret_result:
                        st.subheader("📊 SHAP Summary Plot")
                        st.write(
                            "Understand feature importance and their impact on the predicted rating on a global level."
                        )
                        img_data = base64.b64decode(interpret_result["summary_plot"])
                        st.image(
                            img_data,
                            caption="SHAP Summary Plot",
                            use_container_width=True,
                        )
                    else:
                        st.warning("SHAP Summary Plot not available.")

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Error during interpretation: {e}")
                    if "interpret_result" in st.session_state:
                        del st.session_state.interpret_result
                except KeyError:
                    st.error(
                        "❌ Error: summary_plot or force_plot missing from interpretation result."
                    )
                    if "interpret_result" in st.session_state:
                        del st.session_state.interpret_result
