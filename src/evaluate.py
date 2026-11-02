import json
import os
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def compute_mae(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Calcula el Error Absoluto Medio (MAE) en minutos."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Calcula la Raíz del Error Cuadrático Medio (RMSE) en minutos."""
    return float(root_mean_squared_error(y_true, y_pred))


def compute_r2(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Calcula el coeficiente de determinación (R²)."""
    return float(r2_score(y_true, y_pred))


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: Optional[str] = None
) -> Dict[str, Union[str, float]]:
    """
    Calcula de forma consistente el conjunto oficial de métricas de regresión del proyecto:
    - MAE (Mean Absolute Error, principal)
    - RMSE (Root Mean Squared Error, secundaria)
    - R² (Coeficiente de determinación)
    
    Retorna un diccionario con las métricas redondeadas a 4 decimales.
    """
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)

    result = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }
    if model_name:
        result = {"Modelo": model_name, **result}
    return result


def metrics_to_dataframe(metrics_list: List[Dict[str, Union[str, float]]]) -> pd.DataFrame:
    """Convierte una lista de diccionarios de métricas en un DataFrame ordenado para tablas comparativas."""
    return pd.DataFrame(metrics_list)


def save_metrics_json(metrics_data: Union[Dict, List[Dict]], filepath: str = "outputs/metrics.json") -> None:
    """Guarda o actualiza las métricas en un archivo JSON estructurado."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)
