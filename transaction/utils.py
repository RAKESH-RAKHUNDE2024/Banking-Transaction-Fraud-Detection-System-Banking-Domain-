import pandas as pd
import joblib
from transaction.logger import logger

def read_csv(path: str) -> pd.DataFrame:
    logger.info(f"Reading CSV file: {path}")
    return pd.read_csv(path)

def save_object(path: str, obj) -> None:
    logger.info(f"Saving object to: {path}")
    joblib.dump(obj, path)

def load_object(path: str):
    logger.info(f"Loading object from: {path}")
    return joblib.load(path)