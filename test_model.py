import joblib
import pandas as pd

model = joblib.load(
    "model/customer_segmentation_pipeline.pkl"
)

customer = pd.DataFrame([{
    "Income": 60000,
    "Total_amount_spent": 1200,
    "Total_campaign_accepted": 2,
    "Total_interaction": 20
}])

prediction = model.predict(customer)[0]

print("Predicted Cluster:", prediction)
