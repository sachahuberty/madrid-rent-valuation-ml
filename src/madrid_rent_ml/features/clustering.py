import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def assign_mega_districts(df: pd.DataFrame) -> pd.DataFrame:
    district_mapping = {
        'Salamanca': 'Premium Core', 'Retiro': 'Premium Core', 'Chamberí': 'Premium Core', 'Centro': 'Premium Core',
        'Chamartín': 'Affluent North', 'Moncloa': 'Affluent North', 'Hortaleza': 'Affluent North', 'Fuencarral': 'Affluent North',
        'Arganzuela': 'Middle Ring', 'Tetuán': 'Middle Ring', 'Ciudad Lineal': 'Middle Ring', 'San Blas': 'Middle Ring',
        'Carabanchel': 'Budget South', 'Latina': 'Budget South', 'Puente de Vallecas': 'Budget South', 'Usera': 'Budget South', 'Moratalaz': 'Budget South',
        'Villa de Vallecas': 'Deep Periphery', 'Vicálvaro': 'Deep Periphery', 'Villaverde': 'Deep Periphery', 'Barajas': 'Deep Periphery'
    }
    df['Mega_District'] = df['District'].map(district_mapping)
    return df

def build_clustering_and_filter(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    logger.info(f"Running Segmentation 2.0 with KMeans (k={k})")
    k = int(k)
    if k < 1:
        raise ValueError(f"n_clusters (k) must be an integer >= 1. Got: {k}")
        
    df_dummies = pd.get_dummies(df['Mega_District'], prefix='Zone').astype(int)
    continuous_features = ['Rent', 'Sq.Mt', 'Distance_to_Center_km', 'Bedrooms']
    
    df_seg = pd.concat([df[continuous_features], df_dummies], axis=1).dropna()
    
    scaler_2 = StandardScaler()
    scaled_data_2 = scaler_2.fit_transform(df_seg)
    
    kmeans_2 = KMeans(n_clusters=k, random_state=42, n_init=10)
    df.loc[df_seg.index, 'Cluster_2.0'] = kmeans_2.fit_predict(scaled_data_2)
    
    # Identify Premium Core cluster
    cluster_zone = df.groupby("Cluster_2.0")["Mega_District"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    cluster_sizes = df["Cluster_2.0"].value_counts()
    premium_candidates = cluster_zone[cluster_zone == "Premium Core"].index.tolist()
    core_cluster_id = max(premium_candidates, key=lambda c: cluster_sizes.loc[c])
    
    logger.info(f"Filtering to Premium Core Segment (Cluster {core_cluster_id})")
    df_core = df[df['Cluster_2.0'] == core_cluster_id].copy()
    return df_core

def create_abt(df_core: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating Analytical Base Table (ABT)...")
    abt = df_core.copy()
    abt["log_rent"] = np.log(abt["Rent"])
    abt["log_sqmt"] = np.log(abt["Sq.Mt"])
    
    den = abt["Bedrooms"].replace(0, np.nan)
    abt["SqMt_per_Bedroom"] = (abt["Sq.Mt"] / den).fillna(abt["Sq.Mt"])
    
    amenity_cols = [c for c in ["Elevator","Outer","Terrace","Parking","Furnished","Penthouse"] if c in abt.columns]
    abt["amenities_count"] = abt[amenity_cols].sum(axis=1) if amenity_cols else 0
    
    if "District" in abt.columns:
        district_dum = pd.get_dummies(abt["District"], prefix="District", drop_first=True)
        abt = pd.concat([abt.drop(columns=["District"]), district_dum], axis=1)
        
    cols_to_drop = ['Price_per_sqm', 'Log_Rent', 'Address', 'Cluster', 'Cluster_2.0', 'Mega_District', 'Area', 'Rent', 'Sq.Mt', 'Log_SqMt']
    existing_cols = [c for c in cols_to_drop if c in abt.columns]
    abt = abt.drop(columns=existing_cols)
    abt = abt.dropna()
    return abt