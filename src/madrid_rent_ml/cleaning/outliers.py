import pandas as pd
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def remove_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing top 5 Rent and top 5 Sq.Mt extreme outliers")
    top_rent_indices = df.nlargest(5, 'Rent').index
    top_sqmt_indices = df.nlargest(5, 'Sq.Mt').index
    outlier_indices = set(top_rent_indices).union(set(top_sqmt_indices))
    return df.drop(index=outlier_indices).copy()