import pandas as pd
from sklearn.model_selection import train_test_split
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def perform_split(abt: pd.DataFrame, target: str, test_size: float = 0.5, random_state: int = 42):
    """Note: Original notebook used a 90/10 split to reserve for predictions, then a 50/50 test split."""
    logger.info(f"Splitting data with test size={test_size}")
    
    # 10% kept out entirely as "reserved test"
    abt_model, abt_reserved = train_test_split(abt, test_size=0.1, random_state=random_state)
    logger.info(f"Model dataset: {abt_model.shape[0]} | Reserved unseen: {abt_reserved.shape[0]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        abt_model.drop(columns=[target]), 
        abt_model[target],                
        test_size=test_size,
        random_state=random_state                  
    )
    logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test, abt_reserved