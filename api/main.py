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

    cluster = int(model.predict(data)[0])

    return {
        "cluster": cluster
    }
