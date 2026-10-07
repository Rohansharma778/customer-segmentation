import streamlit as st
import requests
import pandas as pd           
import matplotlib.pyplot as plt 
import seaborn as sns

API_URL = "https://customer-segmentation-api-pf7n.onrender.com"

# -------------------------
# Cluster Profiles
# -------------------------

@st.cache_data
def load_background_data():
    return pd.read_csv('pca_clusters_background.csv')

try:
    df_background = load_background_data()
except FileNotFoundError:
    df_background = None

CLUSTER_PROFILES = {
    0: {
        "name": "Cluster 0",
        "description": "Low-Value / Low-Engagement Customers",
        "income": 35437.13,
        "spending": 117.11,
        "campaigns": 0.09,
        "interaction": 14.75,

        "properties": [
            "Average Income: 35,437",
            "Average Spending: 117",
            "Average Campaigns Accepted: 0.09",
            "Average Interaction: 14.75"
        ],
        "recommendation": (
            "Use low-cost promotions, discounts, and simple campaigns. "
            "Focus on understanding why these customers have low spending "
            "and avoid spending excessive marketing budget on this segment."
        )
    },

    1: {
        "name": "Cluster 1",
        "description": "Regular / Highly Engaged Customers",
        "income": 64445.05,
        "spending": 951.66,
        "campaigns": 0.14,
        "interaction": 25.98,

        "properties": [
            "Average Income: 64,445",
            "Average Spending: 952",
            "Average Campaigns Accepted: 0.14",
            "Average Interaction: 25.98"
        ],
        "recommendation": (
            "Focus on personalized offers, product recommendations, "
            "cross-selling, and loyalty programs. These customers interact "
            "frequently but have relatively low campaign acceptance."
        )
    },

    2: {
        "name": "Cluster 2",
        "description": "High-Value / Campaign-Responsive Customers",
        "income": 79080.48,
        "spending": 1561.91,
        "campaigns": 1.80,
        "interaction": 24.39,

        "properties": [
            "Average Income: 79,080",
            "Average Spending: 1,562",
            "Average Campaigns Accepted: 1.80",
            "Average Interaction: 24.39"
        ],
        "recommendation": (
            "Treat this as a high-value customer segment. Use personalized "
            "campaigns, VIP or loyalty programs, premium offers, upselling, "
            "and cross-selling strategies."
        )
    }
}



st.set_page_config(
    page_title="Customer Segmentation",
    layout="centered"
)


st.title("Customer Segmentation")
st.write(
    "Enter customer information to identify the customer's segment."
)
st.sidebar.header("📊 Model Metrics")

# st.sidebar.metric creates beautiful scorecards!
st.sidebar.metric(label="Silhouette Score", value="0.5653")
st.sidebar.metric(label="PCA Variance Explained", value="84.21%")
st.sidebar.metric(label="Optimal Clusters (K)", value="3")

st.sidebar.markdown("---")
st.sidebar.write("**Cluster Sizes (Training Data):**")
st.sidebar.write("High-Value: 250 customers")
st.sidebar.write("Regular: 865 customers")
st.sidebar.write("Low-Value: 1,114 customers")


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
            json=payload,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            cluster = result["cluster"]

            st.success(
                f"Customer belongs to Cluster {cluster}"
            )

            profile = CLUSTER_PROFILES.get(cluster)

            if profile:

                st.subheader(
                    f"{profile['name']} Profile"
                )

                st.write(
                    f"{profile['description']}"
                )
                st.markdown("### Why this segment?")
                if df_background is not None and "pca_x" in result:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # 1. Draw the background clusters (the blue, orange, and green dots)
                    sns.scatterplot(
                        data=df_background,
                        x='PCA1', y='PCA2',
                        hue='Cluster',
                        palette='deep',
                        alpha=0.5,
                        ax=ax
                    )
                    
                    centroids = df_background.groupby('Cluster')[['PCA1', 'PCA2']].mean().values
                    ax.scatter(
                        centroids[:, 0], centroids[:, 1],
                        s=200, c='red', marker='X', edgecolor='black', label='Centroids'
                    )
                    
                    ax.scatter(
                        result["pca_x"], result["pca_y"],
                        s=400, c='gold', marker='*', edgecolor='black',
                        label='New Customer', zorder=5
                    )

                    # Make it look professional
                    ax.set_title("Customer Position in Segment Space (K=3)")
                    ax.set_xlabel("PCA Component 1")
                    ax.set_ylabel("PCA Component 2")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                    st.pyplot(fig)
                else:
                    st.warning("Missing background data. Make sure pca_clusters_background.csv is in the folder!")
                
                
                st.markdown("### Customer vs Segment Average")
                
                # Create a figure with 1 row and 4 columns of mini-charts
                fig_comp, axes_comp = plt.subplots(1, 4, figsize=(12, 4))
                
                # The names of our features
                features = ['Income', 'Spending', 'Campaigns', 'Interaction']
                
                # The customer's actual input values
                customer_values = [income, total_spent, campaigns, interaction]
                
                # The cluster's average values (from our CLUSTER_PROFILES dictionary)
                cluster_averages = [profile['income'], profile['spending'], profile['campaigns'], profile['interaction']]
                
                # Loop through all 4 features and draw a mini bar chart for each
                for i in range(4):
                    axes_comp[i].bar(
                        ['Customer', 'Avg'], 
                        [customer_values[i], cluster_averages[i]], 
                        color=['gold', 'lightgray'], 
                        edgecolor='black'
                    )
                    axes_comp[i].set_title(features[i])
                
                # Adjust layout so the charts don't overlap, then display it!
                plt.tight_layout()
                st.pyplot(fig_comp)
                
                st.markdown("### Recommended Strategy")
                st.info(profile['recommendation'])
                
                with st.expander("cluster profile"):
                    st.write(
                        f"**Customer type:**{profile['description']}"
                    )
                    st.write(
                        f"**Customer Income:**{income:,.2f}"
                    )
                with st.expander("Cluster Characteristics"):
                     for property in profile["properties"]:
                        st.write(f"• {property}")
                
            else:

                st.warning(
                    "Cluster profile not available."
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
