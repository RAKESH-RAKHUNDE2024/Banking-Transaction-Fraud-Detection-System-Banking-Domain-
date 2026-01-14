import pandas as pd
from sklearn.model_selection import train_test_split
from transaction.logger import logger
from transaction.config import TRAIN_FILE_PATH, TEST_FILE_PATH
from transaction.utils import read_csv

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion...")

        df = read_csv(self.data_path)

        # Clean column name for easy coding
        df.columns = [c.strip().lower().replace(" ", "_").replace("(inr)", "inr") for c in df.columns]

        # Remove transaction_id from ML features but keep in file
        # We'll keep it in dataset, but drop in transformation

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["fraud_flag"]
        )

        train_df.to_csv(TRAIN_FILE_PATH, index=False)
        test_df.to_csv(TEST_FILE_PATH, index=False)

        logger.info(f"Train data saved: {TRAIN_FILE_PATH}")
        logger.info(f"Test data saved: {TEST_FILE_PATH}")

        return TRAIN_FILE_PATH, TEST_FILE_PATH