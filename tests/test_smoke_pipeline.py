import pytest
import pandas as pd
import numpy as np
import os
from src.madrid_rent_ml.pipeline.steps import load_config, run_data_ingestion, run_cleaning, run_features, run_split, run_training, run_evaluation
from src.madrid_rent_ml.utils.paths import get_root_path
from src.madrid_rent_ml.modeling.predict import predict_model

@pytest.fixture
def mock_config():
    return {
        "project": "Test Project",
        "random_seed": 42,
        "paths": {
            "raw_data": "data/raw/dummy.xlsx",
            "cleaned_data": "artifacts/cleaned_data_test.csv",
            "model": "artifacts/model_test.pkl",
            "metrics": "artifacts/metrics_test.json"
        },
        "pipeline": {
            "target_column": "log_rent",
            "vif_threshold": 30.0,
            "p_value_threshold": 0.05,
            "test_size": 0.2
        },
        "features": {
            "clustering_k": 2
        }
    }

@pytest.fixture
def mock_df():
    data = {
        'Rent': [1000, 1500, 2000, 1200, 800, 3000, 2500, 1100, 1600, 900],
        'Sq.Mt': [50, 80, 120, 60, 40, 150, 130, 55, 85, 45],
        'Bedrooms': [1, 2, 3, 2, 0, 4, 3, 1, 2, 1], # 0 represents Studio
        'Address': ['Estudio 1', 'Piso A', 'Piso B', 'Piso C', 'Piso D', 'Piso E', 'Piso F', 'Piso G', 'Piso H', 'Chalet 1'],
        'Area': ['Centro', 'Salamanca', 'Centro', 'Chamberí', 'Retiro', 'Chamartín', 'Tetuán', 'Latina', 'Carabanchel', 'Moncloa'],
        'Cottage': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'Floor': [1, 3, 5, 2, -1, 4, 2, 1, 3, np.nan],
        'Elevator': [1, 1, 1, np.nan, 0, 1, 1, 0, 1, np.nan],
        'Outer': [1, 1, np.nan, 1, 0, 1, 1, 1, 0, np.nan],
        'District': ['Centro', 'Salamanca', 'Centro', 'Chamberí', 'Retiro', 'Chamartín', 'Tetuán', 'Latina', 'Carabanchel', 'Moncloa']
    }
    return pd.DataFrame(data)

def test_pipeline_smoke(mock_df, mock_config, tmp_path, monkeypatch):
    # Mock IO
    monkeypatch.setattr("src.madrid_rent_ml.pipeline.steps.load_excel", lambda x: mock_df.copy())
    
    # Run
    df_clean = run_cleaning(mock_df)
    assert not df_clean.empty, "Cleaning failed"
    
    abt = run_features(df_clean, mock_config)
    assert "log_rent" in abt.columns, "Features failed"
    
    X_train, X_test, y_train, y_test, abt_res = run_split(abt, mock_config)
    assert len(X_train) > 0, "Split failed"
    
    # Pass arbitrary thresh for very tiny dataset to prevent throwing away all cols
    model_dict = run_training(X_train, y_train, {"pipeline": {"vif_threshold": 1000.0, "p_value_threshold": 0.99}})
    assert model_dict["model"] is not None
    
    # We don't verify plots since they save to hardcoded paths for this test unless we monkeypatch them
    y_pred = predict_model(model_dict, X_test)
    assert len(y_pred) == len(y_test)
