import pandas as pd
import numpy as np
import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def resolve_area_names(df: pd.DataFrame) -> pd.DataFrame:
    area_corrections = {
        'Niño JesÁºs': 'Niño Jesús',
        'chalet independiente en Nueva España': 'Nueva España',
        'en Nuevos Ministerios-Ríos Rosas': 'Ríos Rosas', 
        'HellÁ­n': 'Hellín',
        'ZofÁ­o': 'Zofío',
        'San FermÁ­n': 'San Fermín',
        'Valdebernardo - Valderribas': 'Valdebernardo' 
    }
    df['Area'] = df['Area'].replace(area_corrections)
    return df

def calculate_distances(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating geopy distances dynamically... This might take a few minutes if first run.")
    sol_coords = (40.416729, -3.703339) 
    geolocator = Nominatim(user_agent="madrid_real_estate_student_project")
    
    unique_areas = df['Area'].dropna().unique()
    area_coordinates = {}
    
    for area in unique_areas:
        try:
            search_query = f"{area}, Madrid, Spain"
            location = geolocator.geocode(search_query, timeout=10)
            if location:
                area_coordinates[area] = (location.latitude, location.longitude)
            else:
                area_coordinates[area] = (np.nan, np.nan)
            time.sleep(1) # ratelimit
        except Exception as e:
            area_coordinates[area] = (np.nan, np.nan)
    
    def calc_dist(coords):
        if pd.isna(coords[0]): return np.nan
        return geodesic(coords, sol_coords).kilometers
        
    df['Area_Coords'] = df['Area'].map(area_coordinates)
    df['Distance_to_Center_km'] = df['Area_Coords'].apply(calc_dist)
    df = df.drop(columns=['Area_Coords'])
    return df