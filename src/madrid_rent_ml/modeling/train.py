import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.madrid_rent_ml.logging_utils import get_logger
logger = get_logger(__name__)

def remove_vif(X: pd.DataFrame, thresh: float = 30.0) -> pd.DataFrame:
    logger.info(f"Removing VIF > {thresh}")
    X_actual = X.copy()
    num_pred = [c for c in X_actual.select_dtypes(include=[np.number]).columns]
    X_actual = X_actual[num_pred]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while True:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_actual.columns
            vif_data["VIF"] = [variance_inflation_factor(X_actual.values, i) for i in range(X_actual.shape[1])]
            
            max_vif = vif_data["VIF"].max()
            if max_vif > thresh:
                var_to_remove = vif_data.sort_values("VIF", ascending=False).iloc[0]["Variable"]
                logger.info(f"VIF dropping: {var_to_remove} ({max_vif:.2f})")
                X_actual = X_actual.drop(columns=[var_to_remove])
            else:
                break
    return X_actual

def run_rfecv(X: pd.DataFrame, y: pd.Series):
    logger.info("Running RFECV with Linear Regression")
    base_model = LinearRegression()
    n_samples = len(X)
    
    if n_samples < 2:
        logger.warning("Too few samples for RFECV. Returning all features.")
        base_model.fit(X, y)
        return base_model, list(X.columns)

    n_splits = min(5, n_samples)
    rfecv = RFECV(
        estimator=base_model, step=1,
        cv=RepeatedKFold(n_splits=n_splits, n_repeats=3, random_state=42),
        scoring='neg_mean_squared_error'
    )
    pipeline_rfe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', rfecv)
    ])
    pipeline_rfe.fit(X, y)
    selected_features = X.columns[pipeline_rfe.named_steps['feature_selection'].support_]
    return pipeline_rfe, selected_features.tolist()

def backward_elimination(X: pd.DataFrame, y: pd.Series, threshold: float = 0.05):
    logger.info("Running backwards elimination using statsmodels OLS")
    features = list(X.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while len(features) > 0:
            X_with_const = sm.add_constant(X[features])
            model = sm.OLS(y, X_with_const).fit()
            p_values = model.pvalues.drop('const', errors='ignore')
            if len(p_values) == 0:
                break
            max_p_value = p_values.max()
            if max_p_value > threshold:
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
                logger.info(f"OLS dropping (p={max_p_value:.3f}): {excluded_feature}")
            else:
                break
        # Fit final to return
        final_X_const = sm.add_constant(X[features])
        final_model = sm.OLS(y, final_X_const).fit()
    return final_model, features

def train_model(X_train: pd.DataFrame, y_train: pd.Series, vif_t: float=30.0, p_t: float=0.05):
    X_vif = remove_vif(X_train, thresh=vif_t)
    _, selected_features = run_rfecv(X_vif, y_train)
    X_rfe = X_vif[selected_features]
    final_model, final_features = backward_elimination(X_rfe, y_train, threshold=p_t)
    
    logger.info(f"Final training complete. Features kept: {final_features}")
    return {"model": final_model, "features": final_features}