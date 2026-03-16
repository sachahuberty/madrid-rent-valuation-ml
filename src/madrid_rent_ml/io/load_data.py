import pandas as pd
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def load_excel(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Loaded dataframe with shape: {df.shape}")
    return df