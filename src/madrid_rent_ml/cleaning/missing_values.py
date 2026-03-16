import pandas as pd
import numpy as np
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def impute_studios(df: pd.DataFrame) -> pd.DataFrame:
    is_studio = df['Address'].str.contains('Estudio', case=False, na=False)
    df.loc[is_studio & df['Bedrooms'].isna(), 'Bedrooms'] = 0
    return df

def impute_hardcoded_areas(df: pd.DataFrame) -> pd.DataFrame:
    area_fixes = {
        'Piso en Bellas Vistas': 'Bellas Vistas',
        'Piso en Ventilla-Almenara': 'Ventilla-Almenara',
        'Piso en Cuzco-Castillejos': 'Cuzco-Castillejos',
        'Piso en Cuatro Caminos': 'Cuatro Caminos'
    }
    for address_text, correct_area in area_fixes.items():
        mask = df['Area'].isna() & (df['Address'] == address_text)
        df.loc[mask, 'Area'] = correct_area
    return df

def impute_cottage_features(df: pd.DataFrame) -> pd.DataFrame:
    is_cottage = df['Cottage'] == 1
    df.loc[is_cottage, 'Floor'] = df.loc[is_cottage, 'Floor'].fillna(0)
    df.loc[is_cottage, 'Elevator'] = df.loc[is_cottage, 'Elevator'].fillna(0)
    df.loc[is_cottage, 'Outer'] = df.loc[is_cottage, 'Outer'].fillna(1)
    return df

def hierarchical_mode_fills(df: pd.DataFrame) -> pd.DataFrame:
    def group_mode_fill(df, col, by):
        df[col] = df.groupby(by)[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )
        return df
    
    # 1. Floor
    df['Floor'] = df.groupby(['District', 'Cottage'])['Floor'].transform(lambda x: x.fillna(x.median()))
    df['Floor'] = df['Floor'].fillna(df['Floor'].median())
    
    # 2. Elevator
    df['Floor_bin'] = pd.cut(df['Floor'], bins=[-np.inf, 0, 2, 5, np.inf], labels=['<=0', '1-2', '3-5', '6+'])
    df.loc[df['Floor'] <= 0, 'Elevator'] = df.loc[df['Floor'] <= 0, 'Elevator'].fillna(0)
    df = group_mode_fill(df, 'Elevator', ['District', 'Floor_bin'])
    df = group_mode_fill(df, 'Elevator', ['District'])
    df['Elevator'] = df['Elevator'].fillna(0).astype(int)
    
    # 3. Outer
    df = group_mode_fill(df, 'Outer', ['Area'])
    df = group_mode_fill(df, 'Outer', ['District'])
    df['Outer'] = df['Outer'].fillna(1).astype(int)
    
    # 4. Refine Floor
    df['Floor'] = df.groupby(['District', 'Elevator'])['Floor'].transform(lambda x: x.fillna(x.median()))
    df['Floor'] = df['Floor'].fillna(df['Floor'].median())
    df = df.drop(columns=['Floor_bin'])
    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Executing missing value imputations...")
    df = impute_studios(df)
    df = impute_hardcoded_areas(df)
    df = impute_cottage_features(df)
    df = hierarchical_mode_fills(df)
    return df