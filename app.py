import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="centered"
)


st.title("👥 Customer Segmentation")
st.write(
    "Enter customer information to identify the customer's segment."
)


# -------------------------
# Input fields
# -------------------------

income = st.number_input(
    "Income",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)

total_spent = st.number_input(
    "Total Amount Spent",
    min_value=0.0,
    value=1200.0,
    step=50.0
)

campaigns = st.number_input(
    "Total Campaigns Accepted",
    min_value=0.0,
    value=2.0,
    step=1.0
)

interaction = st.number_input(
    "Total Interaction",
    min_value=0.0,
    value=20.0,
    step=1.0
)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Customer Segment"):

    payload = {
        "Income": income,
        "Total_amount_spent": total_spent,
        "Total_campaign_accepted": campaigns,
        "Total_interaction": interaction
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            cluster = result["cluster"]

            st.success(
                f"Customer belongs to Cluster {cluster}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the API is running."
        )
    