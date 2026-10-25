import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AIRPORT_ZONES = {1, 132, 138}  # 1: Newark (EWR), 132: JFK Airport, 138: LaGuardia Airport (LGA)

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Genera variables temporales, ciclicas y espaciales."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        pickup = pd.to_datetime(X['tpep_pickup_datetime'])
        
        # Features temporales
        X['pickup_hour'] = pickup.dt.hour
        X['pickup_dayofweek'] = pickup.dt.dayofweek
        X['pickup_month'] = pickup.dt.month
        X['is_weekend'] = (pickup.dt.dayofweek >= 5).astype(int)
        
        # Hora punta urbana en NYC
        is_weekday = X['is_weekend'] == 0
        is_rush = ((X['pickup_hour'].between(7, 9)) | (X['pickup_hour'].between(16, 19))) & is_weekday
        X['is_rush_hour'] = is_rush.astype(int)
        
        # Transformaciones cíclicas de la hora (seno y coseno)
        X['sin_hour'] = np.sin(2 * np.pi * X['pickup_hour'] / 24.0)
        X['cos_hour'] = np.cos(2 * np.pi * X['pickup_hour'] / 24.0)
        
        # Indicador de viaje con origen o destino en aeropuerto
        X['is_airport'] = (
            X['PULocationID'].isin(AIRPORT_ZONES) | 
            X['DOLocationID'].isin(AIRPORT_ZONES)
        ).astype(int)
        
        return X

def get_preprocessor(categorical_features, numeric_features):
    """Retorna un ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
