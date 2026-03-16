import pandas as pd
import statsmodels.api as sm
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def predict_model(model_dict: dict, X: pd.DataFrame) -> pd.Series:
    model = model_dict["model"]
    features = model_dict["features"]
    
    # Subset to the required features
    X_subset = X[features]
    X_const = sm.add_constant(X_subset, has_constant='add')
    
    preds = model.predict(X_const)
    return preds