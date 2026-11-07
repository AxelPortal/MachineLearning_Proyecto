import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AIRPORT_ZONES = {1, 132, 138}  # 1: Newark (EWR), 132: JFK Airport, 138: LaGuardia Airport (LGA)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Genera variables temporales, cíclicas y espaciales sin riesgo de Data Leakage,
    utilizando directamente tipos de datos compactos (int8, uint16, float32) para optimizar la memoria RAM.
    """

    def __init__(self, drop_datetime: bool = True):
        self.drop_datetime = drop_datetime

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        pickup = pd.to_datetime(X["tpep_pickup_datetime"])

        # Features temporales en int8 (1 byte por fila en vez de 8 bytes)
        X["pickup_hour"] = pickup.dt.hour.astype("int8")
        X["pickup_dayofweek"] = pickup.dt.dayofweek.astype("int8")
        X["pickup_month"] = pickup.dt.month.astype("int8")
        X["is_weekend"] = (X["pickup_dayofweek"] >= 5).astype("int8")

        # Hora punta urbana en NYC (Lunes a Viernes de 7-9 AM y 16-19 PM) en int8
        is_weekday = X["is_weekend"] == 0
        is_rush = ((X["pickup_hour"].between(7, 9)) | (X["pickup_hour"].between(16, 19))) & is_weekday
        X["is_rush_hour"] = is_rush.astype("int8")

        # Transformaciones cíclicas de la hora (seno y coseno) en float32 (4 bytes en vez de 8 bytes)
        X["sin_hour"] = np.sin(2.0 * np.pi * X["pickup_hour"] / 24.0).astype("float32")
        X["cos_hour"] = np.cos(2.0 * np.pi * X["pickup_hour"] / 24.0).astype("float32")

        # Indicador de viaje con origen o destino en aeropuerto en int8
        X["is_airport"] = (
            X["PULocationID"].isin(AIRPORT_ZONES) | X["DOLocationID"].isin(AIRPORT_ZONES)
        ).astype("int8")

        # Eliminar el timestamp crudo una vez extraídas las componentes temporales para ahorrar ~125 MB de RAM
        if self.drop_datetime and "tpep_pickup_datetime" in X.columns:
            X.drop(columns=["tpep_pickup_datetime"], inplace=True)

        return X


def get_preprocessor(categorical_features, numeric_features, binary_features=None):
    """
    Retorna un ColumnTransformer optimizado en memoria (sparse CSR + float32)
    listo para encadenar en un Pipeline de scikit-learn.
    """
    transformers = [
        ("num", StandardScaler(), numeric_features),
    ]
    if binary_features:
        transformers.append(("bin", "passthrough", binary_features))
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), categorical_features)
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")
