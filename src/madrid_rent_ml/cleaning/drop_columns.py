import pandas as pd
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def drop_initial_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['Id', 'Number']
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    logger.info(f"Dropping columns: {existing_cols}")
    return df.drop(columns=existing_cols)

def drop_missing_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping rows with missing 'Bedrooms'")
    return df.dropna(subset=['Bedrooms']).copy()