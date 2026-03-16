import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def plot_real_vs_fitted(y_true: pd.Series, y_pred: pd.Series, dataset: str, save_path: str):
    plt.figure(figsize=(15, 6))
    plt.title(f"Real vs. Fitted ({dataset} dataset)", fontsize=14)
    plt.scatter(y_true, y_pred, alpha=0.5) 
    coef = np.polyfit(y_true, y_pred, 1)  
    poly1d_fn = np.poly1d(coef)  
    plt.plot(y_true, poly1d_fn(y_true), color="red", label="Regression line")
    plt.xlabel("Real")
    plt.ylabel("Fitted")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6) 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved Real vs Fitted plot to {save_path}")

def plot_residuals(y_true: pd.Series, y_pred: pd.Series, dataset: str, save_path: str):
    errors = y_true - y_pred
    plt.figure(figsize=(14, 6))
    sns.histplot(errors, kde=True, color="skyblue", edgecolor="white")
    plt.title(f"Histogram of residuals ({dataset} dataset)", fontsize=15, pad=15)
    plt.xlabel("Error")
    plt.ylabel("Frequence")
    plt.axvline(x=0, color='red', linestyle='--', label='Error zero')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved Residuals plot to {save_path}")