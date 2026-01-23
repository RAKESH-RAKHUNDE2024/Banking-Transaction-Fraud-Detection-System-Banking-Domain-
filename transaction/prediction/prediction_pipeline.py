import pandas as pd
from transaction.utils import load_object
from transaction.config import MODEL_PATH
from transaction.logger import logger

class PredictionPipeline:
    def __init__(self):
        self.model = load_object(MODEL_PATH)

    def predict(self, input_dict: dict):
        df = pd.DataFrame([input_dict])

        # Same Cleaning
        df.columns = [c.strip().lower().replace(" ", "_").replace("(inr)", "inr") for c in df.columns]

        # Remove Timestamp If Not Included
        if "transaction_id" in df.columns:
            df = df.drop(columns=["transaction_id"])

        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        pred = int(self.model.predict(df)[0])
        proba = float(self.model.predict_proba(df)[0][1])

        logger.info(f"Prediction done. fraud_flag={pred}, probability={proba}")

        return pred, proba
