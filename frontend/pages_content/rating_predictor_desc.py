import streamlit as st


def rating_predictor_desc():
    try:
        st.title("📱 Smartprix Rating Predictor")
        st.markdown("### Make Smarter Smartphone Choices with AI-Powered Ratings! 🚀")

        st.markdown(
            """
            ## 🎯 **What This Tool Does**
            Ever wondered how smartphone features impact its **Smartprix rating**? Our tool helps you:
            
            ✅ **Predict Ratings** based on specifications like price, camera, RAM, and more.\n
            ✅ **Understand Key Factors** that influence ratings the most.\n
            ✅ **Compare Features** and optimize your smartphone selection.
            """
        )

        st.markdown(
            """
        ## 🏆 **Who Can Benefit?**
        🔹 **Retailers & Brands** – Improve products based on AI-driven insights.\n
        🔹 **Tech Enthusiasts** – Explore how smartphone features affect ratings.
        """
        )

        st.markdown(
            """
        ## 🚀 **How It Works**
        1️⃣ **Enter Smartphone Details** – Input price, camera, RAM, etc.\n
        2️⃣ **Get Instant Prediction** – See the expected Smartprix rating.\n
        3️⃣ **Understand the Insights** – Learn why a phone gets a specific rating. 
        """
        )

        st.markdown(
            """
        ## 🔹 **Why Use This Tool?**
        ✅ **Saves Time** – No need to manually compare multiple phones.\n 
        ✅ **Data-Driven Insights** – Understand what makes a smartphone highly rated.\n 
        ✅ **User-Friendly** – Simple interface for quick predictions. 
        """
        )

        st.success("👀 **Try it now! Predict a smartphone rating today!** 🚀")

    except Exception as e:
        st.error(f"An error occurred while displaying the description: {e}")
