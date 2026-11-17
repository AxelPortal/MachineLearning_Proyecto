import json
import os
import sys
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluate import evaluate_predictions, save_metrics_json
from src.features import ODPairTransformer

RANDOM_STATE = 42

DATA_PATH = os.path.join(ROOT, "data", "processed", "yellow_tripdata_processed.parquet")
METRICS_PATH = os.path.join(ROOT, "outputs", "metrics.json")
MODEL_DIR = os.path.join(ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

CATEGORICAL = [
    "VendorID",
    "PULocationID",
    "DOLocationID",
]

NUMERIC = [
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "sin_hour",
    "cos_hour",
    "is_airport",
]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "trip_duration"


def load_and_split_data(
    data_path: str = DATA_PATH,
) -> Dict[str, Any]:
    """
    Carga el dataset procesado de 6 meses y realiza la partición temporal oficial:
    - train: Enero a Abril (meses 1-4) para búsqueda de hiperparámetros
    - val: Mayo (mes 5) para validación y selección sin data leakage
    - train_val: Enero a Mayo (meses 1-5, ~83.41%) para entrenamiento definitivo
    - test: Junio (mes 6, ~16.59%) para evaluación final
    """
    cols = FEATURES + [TARGET, "pickup_month"]
    df = pd.read_parquet(data_path, columns=cols)

    train_mask = df["pickup_month"].isin([1, 2, 3, 4])
    val_mask = df["pickup_month"] == 5
    train_val_mask = df["pickup_month"].isin([1, 2, 3, 4, 5])
    test_mask = df["pickup_month"] == 6

    return {
        "X_train": df.loc[train_mask, FEATURES],
        "y_train": df.loc[train_mask, TARGET],
        "X_val": df.loc[val_mask, FEATURES],
        "y_val": df.loc[val_mask, TARGET],
        "X_train_val": df.loc[train_val_mask, FEATURES],
        "y_train_val": df.loc[train_val_mask, TARGET],
        "X_test": df.loc[test_mask, FEATURES],
        "y_test": df.loc[test_mask, TARGET],
    }


def build_rf_pipeline(max_samples: int | float | None = None) -> Pipeline:
    """Construye el Pipeline de Random Forest con OneHotEncoder disperso en float32."""
    preprocessor_rf = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor_rf),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    max_samples=max_samples,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_hgb_pipeline() -> Pipeline:
    """
    Construye el Pipeline de HistGradientBoosting con:
    - ODPairTransformer para capturar la interacción exacta Origen-Destino.
    - TargetEncoder en float32/64 compacto para evitar matrices One-Hot densas en RAM.
    - loss='absolute_error' (L1) para optimizar directamente MAE y ser robusto ante outliers.
    """
    cat_with_od = CATEGORICAL + ["od_pair"]
    preprocessor_hgb = ColumnTransformer(
        [
            (
                "cat",
                TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=RANDOM_STATE),
                cat_with_od,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline(
        [
            ("od_features", ODPairTransformer()),
            ("preprocessor", preprocessor_hgb),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    random_state=RANDOM_STATE,
                    early_stopping=False,
                ),
            ),
        ]
    )


def tune_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_sample_size: int = 80_000,
    val_sample_size: int = 30_000,
    n_iter: int = 4,
) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """
    Ejecuta búsqueda aleatoria con PredefinedSplit (Ene-Abr vs May) sobre muestras
    y compara los mejores candidatos en Mayo (sin leakage).
    """
    X_train_sample = X_train.sample(n=min(train_sample_size, len(X_train)), random_state=RANDOM_STATE)
    y_train_sample = y_train.loc[X_train_sample.index]

    X_val_sample = X_val.sample(n=min(val_sample_size, len(X_val)), random_state=RANDOM_STATE)
    y_val_sample = y_val.loc[X_val_sample.index]

    X_search = pd.concat([X_train_sample, X_val_sample])
    y_search = pd.concat([y_train_sample, y_val_sample])
    fold = np.concatenate(
        [
            np.full(len(X_train_sample), -1),
            np.zeros(len(X_val_sample)),
        ]
    )
    temporal_split = PredefinedSplit(fold)

    params_rf = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.7],
    }

    search_rf = RandomizedSearchCV(
        build_rf_pipeline(),
        params_rf,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=temporal_split,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
        verbose=1,
    )
    search_rf.fit(X_search, y_search)

    params_hgb = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_iter": [150, 250],
        "model__max_leaf_nodes": [31, 63, 127],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    search_hgb = RandomizedSearchCV(
        build_hgb_pipeline(),
        params_hgb,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=temporal_split,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
        verbose=1,
    )
    search_hgb.fit(X_search, y_search)

    rf_candidate = clone(build_rf_pipeline()).set_params(**search_rf.best_params_)
    rf_candidate.fit(X_train_sample, y_train_sample)
    mae_rf_val = float(mean_absolute_error(y_val_sample, rf_candidate.predict(X_val_sample)))

    hgb_candidate = clone(build_hgb_pipeline()).set_params(**search_hgb.best_params_)
    hgb_candidate.fit(X_train_sample, y_train_sample)
    mae_hgb_val = float(mean_absolute_error(y_val_sample, hgb_candidate.predict(X_val_sample)))

    best_name = "Random Forest" if mae_rf_val <= mae_hgb_val else "HistGradientBoosting"
    best_params = {
        "Random Forest": search_rf.best_params_,
        "HistGradientBoosting": search_hgb.best_params_,
    }
    val_maes = {
        "Random Forest": mae_rf_val,
        "HistGradientBoosting": mae_hgb_val,
    }
    return best_name, best_params, val_maes


def update_metrics_file(new_entries: list[Dict[str, Any]], metrics_path: str = METRICS_PATH) -> None:
    """Actualiza el JSON de métricas preservando los baselines existentes."""
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = []

    if not isinstance(metrics, list):
        metrics = []

    managed_names = {entry.get("Modelo") for entry in new_entries} | {
        "Random Forest",
        "HistGradientBoosting",
        "Random Forest Optimizado",
        "HistGradientBoosting Optimizado",
        "HistGradientBoosting L1 + OD",
    }
    metrics = [m for m in metrics if m.get("Modelo") not in managed_names]
    metrics.extend(new_entries)
    save_metrics_json(metrics, metrics_path)


def run_training_pipeline(
    data_path: str = DATA_PATH,
    metrics_path: str = METRICS_PATH,
    model_path: str = BEST_MODEL_PATH,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo de entrenamiento, evaluación en Junio y persistencia."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    print("\n===== 1. CARGA Y PARTICIÓN TEMPORAL (6 MESES) =====")
    splits = load_and_split_data(data_path)
    print(f"Ene-Abr (tuning train): {len(splits['X_train']):,}")
    print(f"Mayo    (validación):   {len(splits['X_val']):,}")
    print(f"Ene-May (train_val):    {len(splits['X_train_val']):,}")
    print(f"Junio   (test final):   {len(splits['X_test']):,}")

    print("\n===== 2. BÚSQUEDA DE HIPERPARÁMETROS Y COMPARACIÓN EN MAYO =====")
    best_name, best_params, val_maes = tune_and_compare(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
    )
    print("MAE validación (Mayo) - Random Forest:", round(val_maes["Random Forest"], 4))
    print("MAE validación (Mayo) - HistGradientBoosting (L1 + OD):", round(val_maes["HistGradientBoosting"], 4))
    print("Modelo seleccionado:", best_name)

    print("\n===== 3. ENTRENAMIENTO DEFINITIVO EN ENE-MAY Y EVALUACIÓN EN JUNIO =====")
    rf_final = build_rf_pipeline(max_samples=min(150_000, len(splits["X_train_val"]))).set_params(
        **best_params["Random Forest"]
    )
    rf_final.fit(splits["X_train_val"], splits["y_train_val"])
    rf_metrics = evaluate_predictions(
        splits["y_test"],
        rf_final.predict(splits["X_test"]),
        "Random Forest Optimizado",
    )

    hgb_final = build_hgb_pipeline().set_params(**best_params["HistGradientBoosting"])
    hgb_final.fit(splits["X_train_val"], splits["y_train_val"])
    hgb_metrics = evaluate_predictions(
        splits["y_test"],
        hgb_final.predict(splits["X_test"]),
        "HistGradientBoosting Optimizado (L1 + OD)",
    )

    best_model = rf_final if best_name == "Random Forest" else hgb_final
    joblib.dump(best_model, model_path)
    print(f"Modelo ganador ({best_name}) guardado en: {model_path}")

    update_metrics_file([hgb_metrics, rf_metrics], metrics_path)
    print("Métricas actualizadas en:", metrics_path)

    return {
        "best_model_name": best_name,
        "val_maes": val_maes,
        "test_metrics": [hgb_metrics, rf_metrics],
        "model_path": model_path,
    }


if __name__ == "__main__":
    run_training_pipeline()
