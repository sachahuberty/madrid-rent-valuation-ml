import pandas as pd
import numpy as np
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def add_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding numerical features (SqMt_per_Bedroom, Log_Rent, Log_SqMt, Price_per_sqm)")
    # Spaciousness Index
    df['SqMt_per_Bedroom'] = df['Sq.Mt'] / df['Bedrooms'].replace(0, 1)
    # Log versions
    df['Log_Rent'] = np.log(df['Rent'])
    df['Log_SqMt'] = np.log(df['Sq.Mt'])
    # Price per sqm
    df['Price_per_sqm'] = df['Rent'] / df['Sq.Mt']
    return df