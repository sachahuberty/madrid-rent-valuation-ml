import argparse
from src.madrid_rent_ml.pipeline.run_pipeline import run as run_pipeline_e2e
from src.madrid_rent_ml.pipeline.steps import load_config, run_data_ingestion, run_cleaning, run_features
from src.madrid_rent_ml.io.save_artifacts import load_model, save_data
from src.madrid_rent_ml.modeling.predict import predict_model

def run_train(config_path):
    print("Running training only... (Usually implies loading pre-cleaned data and training)")
    # Not fully isolated in this snippet for brevity, but pipeline handles logic
    run_pipeline_e2e(config_path)

def run_predict(config_path, input_file):
    print(f"Running inference on {input_file} using {config_path}")
    df = run_data_ingestion({"paths": {"raw_data": input_file}}) 
    config = load_config(config_path)
    df_clean = run_cleaning(df)
    abt = run_features(df_clean, config)

    model_dict = load_model(config["paths"]["model"])
    preds = predict_model(model_dict, abt)
    abt["prediction"] = preds
    output_path = "artifacts/predictions.csv"
    save_data(abt, output_path)
    print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Madrid Rent Valuation MLOps CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Pipeline
    parser_pipeline = subparsers.add_parser("pipeline")
    parser_pipeline.add_argument("--config", required=True)

    # Train
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--config", required=True)

    # Predict
    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("--config", required=True)
    parser_predict.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.command == "pipeline":
        run_pipeline_e2e(args.config)
    elif args.command == "train":
        run_pipeline_e2e(args.config)
    elif args.command == "predict":
        run_predict(args.config, args.input)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()