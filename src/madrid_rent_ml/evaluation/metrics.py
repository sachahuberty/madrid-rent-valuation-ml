import numpy as np
import pandas as pd
from sklearn import metrics
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mae = float(metrics.mean_absolute_error(y_true, y_pred))
    mse = float(metrics.mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(100 * abs(y_true - y_pred) / y_true))
    
    logger.info(f"Metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.3f}%")
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }