import joblib
import json
import pandas as pd
import os
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: str):
    return joblib.load(filepath)

def save_metrics(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {filepath}")

def save_data(df: pd.DataFrame, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")