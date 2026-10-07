from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# Load trained pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "customer_segmentation_pipeline.pkl"

model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Customer Segmentation API",
    description="Customer segmentation using K-Means",
    version="1.0.0"
)


class Customer(BaseModel):
    Income: float
    Total_amount_spent: float
    Total_campaign_accepted: float
    Total_interaction: float


@app.get("/")
def home():
    return {
        "message": "Customer Segmentation API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame([{
        "Income": customer.Income,
        "Total_amount_spent": customer.Total_amount_spent,
        "Total_campaign_accepted": customer.Total_campaign_accepted,
        "Total_interaction": customer.Total_interaction
    }])

    # Predict cluster
    cluster = int(model.predict(data)[0])

    # Transform the customer through everything
    # before the final K-Means model
    transformed_data = model[:-1].transform(data)

    # Get distance from the customer to every K-Means centroid
    distances = model[-1].transform(transformed_data)[0]
    pca_x = float(transformed_data[0][0])
    pca_y = float(transformed_data[0][1])
    
    return {
        "cluster": cluster,
        "distances": {
            "0": float(distances[0]),
            "1": float(distances[1]),
            "2": float(distances[2])
        },
        "pca_x": pca_x,  
        "pca_y": pca_y   
    }
