
import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------
# 1. CONFIGURACIÓN DEL PROYECTO
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.chdir(ROOT)

from src.evaluate import evaluate_predictions, save_metrics_json

RANDOM_STATE = 42

DATA_PATH = "data/processed/yellow_tripdata_processed.parquet"
METRICS_PATH = "outputs/metrics.json"
MODEL_DIR = "models"

os.makedirs("outputs", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. CARGA Y PARTICIÓN TEMPORAL
# --------------------------------------------------

print("\n===== CARGA DE DATOS =====")

df = pd.read_parquet(DATA_PATH)

df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"]
)

df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

train = df[df["pickup_month"] == 1].copy()
val = df[df["pickup_month"] == 2].copy()
test = df[df["pickup_month"] == 3].copy()

print("Enero:", len(train))
print("Febrero:", len(val))
print("Marzo:", len(test))

FEATURES = [
    "VendorID",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "sin_hour",
    "cos_hour",
    "is_airport"
]

TARGET = "trip_duration"

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = val[FEATURES]
y_val = val[TARGET]

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

X_test = test[FEATURES]
y_test = test[TARGET]

# --------------------------------------------------
# 3. MUESTRAS PARA OPTIMIZACIÓN
# --------------------------------------------------

X_train_sample = X_train.sample(
    n=min(40000, len(X_train)),
    random_state=RANDOM_STATE
)

y_train_sample = y_train.loc[X_train_sample.index]

X_val_sample = X_val.sample(
    n=min(20000, len(X_val)),
    random_state=RANDOM_STATE
)

y_val_sample = y_val.loc[X_val_sample.index]

X_search = pd.concat([X_train_sample, X_val_sample])
y_search = pd.concat([y_train_sample, y_val_sample])

fold = np.concatenate([
    np.full(len(X_train_sample), -1),
    np.zeros(len(X_val_sample))
])

temporal_split = PredefinedSplit(fold)

# Muestra común para comparar los modelos
X_val_common = X_val_sample
y_val_common = y_val_sample

# --------------------------------------------------
# 4. PREPROCESAMIENTO
# --------------------------------------------------

CATEGORICAL = [
    "VendorID",
    "PULocationID",
    "DOLocationID"
]

NUMERIC = [
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "sin_hour",
    "cos_hour",
    "is_airport"
]

# --------------------------------------------------
# 5. RANDOM FOREST
# --------------------------------------------------

print("\n===== RANDOM FOREST =====")

preprocessor_rf = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", "passthrough", NUMERIC)
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_rf),
    ("model", RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

params_rf = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.7]
}

search_rf = RandomizedSearchCV(
    rf_pipeline,
    params_rf,
    n_iter=6,
    scoring="neg_mean_absolute_error",
    cv=temporal_split,
    random_state=RANDOM_STATE,
    n_jobs=1,
    refit=False,
    verbose=1
)

search_rf.fit(X_search, y_search)

print("Mejores parámetros RF:")
print(search_rf.best_params_)

# --------------------------------------------------
# 6. HISTGRADIENTBOOSTING
# --------------------------------------------------

print("\n===== HISTGRADIENTBOOSTING =====")

preprocessor_hgb = ColumnTransformer([
    ("cat", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32
    ), CATEGORICAL),
    ("num", "passthrough", NUMERIC)
])

hgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_hgb),
    ("model", HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=False
    ))
])

params_hgb = {
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_iter": [100, 200],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0, 0.1, 1]
}

search_hgb = RandomizedSearchCV(
    hgb_pipeline,
    params_hgb,
    n_iter=6,
    scoring="neg_mean_absolute_error",
    cv=temporal_split,
    random_state=RANDOM_STATE,
    n_jobs=1,
    refit=False,
    verbose=1
)

search_hgb.fit(X_search, y_search)

print("Mejores parámetros HGB:")
print(search_hgb.best_params_)

# --------------------------------------------------
# 7. COMPARACIÓN EN VALIDACIÓN
# --------------------------------------------------

print("\n===== COMPARACIÓN EN FEBRERO =====")

from sklearn.base import clone

# Crear modelos con los mejores hiperparámetros.
# Se ajustan únicamente con enero para evaluar febrero sin leakage.

rf_candidate = clone(rf_pipeline).set_params(
    **search_rf.best_params_
)
rf_candidate.fit(X_train_sample, y_train_sample)

hgb_candidate = clone(hgb_pipeline).set_params(
    **search_hgb.best_params_
)
hgb_candidate.fit(X_train_sample, y_train_sample)

pred_rf_val = rf_candidate.predict(X_val_common)
pred_hgb_val = hgb_candidate.predict(X_val_common)

mae_rf_val = mean_absolute_error(y_val_common, pred_rf_val)
mae_hgb_val = mean_absolute_error(y_val_common, pred_hgb_val)

print("MAE Random Forest:", round(mae_rf_val, 4))
print("MAE HistGradientBoosting:", round(mae_hgb_val, 4))

# Selección basada exclusivamente en febrero
if mae_rf_val <= mae_hgb_val:
    best_name = "Random Forest"
    best_model = rf_candidate
else:
    best_name = "HistGradientBoosting"
    best_model = hgb_candidate

print("\nModelo seleccionado:", best_name)

# --------------------------------------------------
# 8. ENTRENAMIENTO DEFINITIVO
# --------------------------------------------------

print("\n===== ENTRENAMIENTO DEFINITIVO =====")

best_model.fit(X_train_val, y_train_val)

# --------------------------------------------------
# 9. EVALUACIÓN FINAL EN MARZO
# --------------------------------------------------

print("\n===== EVALUACIÓN FINAL EN MARZO =====")

pred_test = best_model.predict(X_test)

final_metrics = evaluate_predictions(
    y_test,
    pred_test,
    best_name + " Optimizado"
)

print(final_metrics)

# --------------------------------------------------
# 10. GUARDAR MODELO
# --------------------------------------------------

model_path = os.path.join(MODEL_DIR, "best_model.joblib")

joblib.dump(best_model, model_path)

print("\nModelo guardado en:", model_path)

# --------------------------------------------------
# 11. ACTUALIZAR MÉTRICAS
# --------------------------------------------------

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
else:
    metrics = []

if not isinstance(metrics, list):
    metrics = []

metrics = [
    m for m in metrics
    if m.get("Modelo") not in [
        "Random Forest Optimizado",
        "HistGradientBoosting Optimizado",
        "Random Forest",
        "HistGradientBoosting"
    ]
]

metrics.append(final_metrics)

save_metrics_json(metrics, METRICS_PATH)

print("\n===== PIPELINE FINALIZADO =====")
print("Métricas guardadas en:", METRICS_PATH)
