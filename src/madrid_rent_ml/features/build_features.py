import pandas as pd
from src.madrid_rent_ml.features.numerical import add_numerical_features
from src.madrid_rent_ml.features.geospatial import resolve_area_names, calculate_distances
from src.madrid_rent_ml.features.clustering import assign_mega_districts, build_clustering_and_filter, create_abt
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def engineer_features(df: pd.DataFrame, clustering_k: int = 4) -> pd.DataFrame:
    logger.info("Starting feature engineering pipe")
    df = add_numerical_features(df)
    df = resolve_area_names(df)
    
    # We will assume geopy is cached or can be calculated dynamically
    df = calculate_distances(df)
    
    df = assign_mega_districts(df)
    df_core = build_clustering_and_filter(df, k=clustering_k)
    
    abt = create_abt(df_core)
    return abt