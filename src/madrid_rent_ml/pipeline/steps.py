import yaml
from src.madrid_rent_ml.logging_utils import get_logger
from src.madrid_rent_ml.io.load_data import load_excel
from src.madrid_rent_ml.io.save_artifacts import save_model, save_metrics, load_model
from src.madrid_rent_ml.cleaning.drop_columns import drop_initial_columns, drop_missing_bedrooms
from src.madrid_rent_ml.cleaning.missing_values import clean_missing_values
from src.madrid_rent_ml.cleaning.outliers import remove_extreme_outliers
from src.madrid_rent_ml.features.build_features import engineer_features
from src.madrid_rent_ml.split.make_split import perform_split
from src.madrid_rent_ml.modeling.train import train_model
from src.madrid_rent_ml.modeling.predict import predict_model
from src.madrid_rent_ml.evaluation.metrics import calculate_metrics
from src.madrid_rent_ml.evaluation.plots import plot_real_vs_fitted, plot_residuals

logger = get_logger(__name__)

def sanitize_config(element):
    """Recursively strip trailing newlines and cast to int/float if feasible."""
    if isinstance(element, dict):
        return {k: sanitize_config(v) for k, v in element.items()}
    elif isinstance(element, list):
        return [sanitize_config(v) for v in element]
    elif isinstance(element, str):
        cleaned = element.strip().replace('\\n', '')
        if cleaned.isdigit():
            return int(cleaned)
        try:
            return float(cleaned)
        except ValueError:
            return cleaned
    return element

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        return sanitize_config(raw_config)

def run_data_ingestion(config: dict):
    df = load_excel(config["paths"]["raw_data"])
    return df

def run_cleaning(df):
    df = drop_initial_columns(df)
    df = clean_missing_values(df)
    df = drop_missing_bedrooms(df)
    df = remove_extreme_outliers(df)
    return df

def run_features(df, config: dict):
    k = config["features"]["clustering_k"]
    abt = engineer_features(df, clustering_k=k)
    return abt

def run_split(abt, config: dict):
    target = config["pipeline"]["target_column"]
    test_size = config["pipeline"]["test_size"]
    seed = config["random_seed"]
    return perform_split(abt, target, test_size, seed)

def run_training(X_train, y_train, config: dict):
    vif = config["pipeline"]["vif_threshold"]
    p_val = config["pipeline"]["p_value_threshold"]
    model_dict = train_model(X_train, y_train, vif, p_val)
    return model_dict

def run_evaluation(model_dict, X_test, y_test, config: dict):
    y_pred = predict_model(model_dict, X_test)
    metrics = calculate_metrics(y_test, y_pred)
    plot_real_vs_fitted(y_test, y_pred, "test", "artifacts/plots/real_vs_fitted_test.png")
    plot_residuals(y_test, y_pred, "test", "artifacts/plots/residuals_test.png")
    return metrics