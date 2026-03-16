import argparse
from src.madrid_rent_ml.pipeline.steps import (
    load_config, run_data_ingestion, run_cleaning, 
    run_features, run_split, run_training, run_evaluation
)
from src.madrid_rent_ml.io.save_artifacts import save_model, save_metrics, save_data
from src.madrid_rent_ml.utils.random_seed import set_seed
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def run(config_path: str):
    logger.info("Initializing Madrid Rent ML Pipeline...")
    config = load_config(config_path)
    set_seed(config["random_seed"])
    
    df = run_data_ingestion(config)
    df_clean = run_cleaning(df)
    abt = run_features(df_clean, config)
    save_data(abt, config["paths"]["cleaned_data"])
    
    X_train, X_test, y_train, y_test, abt_reserved = run_split(abt, config)
    
    model_dict = run_training(X_train, y_train, config)
    save_model(model_dict, config["paths"]["model"])
    
    metrics = run_evaluation(model_dict, X_test, y_test, config)
    save_metrics(metrics, config["paths"]["metrics"])
    
    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLOps Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)