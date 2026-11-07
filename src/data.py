import gc
import glob
import os
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Columnas estrictamente necesarias para calcular target, filtrar anomalías y extraer features sin leakage
REQUIRED_RAW_COLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
]

# Columnas con riesgo de Data Leakage (generadas durante o al finalizar el viaje)
LEAKAGE_COLS = [
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
    "RatecodeID",
    "payment_type",
    "store_and_fwd_flag",
]


def get_data_dir(subdir: str = "raw") -> str:
    """Localiza la carpeta de datos independientemente de si se ejecuta desde root o notebooks/."""
    candidates = [
        os.path.join("data", subdir),
        os.path.join("..", "data", subdir),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", subdir),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join("data", subdir)


def get_monthly_files(data_dir: Optional[str] = None, months: Iterable[int] = (1, 2, 3, 4, 5, 6)) -> List[str]:
    """Retorna la lista ordenada de archivos Parquet para los meses seleccionados (por defecto: Enero a Junio)."""
    if data_dir is None:
        data_dir = get_data_dir("raw")
    selected_files = []
    for m in months:
        pattern = os.path.join(data_dir, f"*2026-{m:02d}.parquet")
        matched = sorted(glob.glob(pattern))
        selected_files.extend(matched)
    if not selected_files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet para los meses {list(months)} en {data_dir}")
    return selected_files


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el consumo de memoria RAM convirtiendo variables numéricas a los tipos de menor tamaño en bytes:
    - int64 -> int8 / uint16 / int32 según el rango de cada variable.
    - float64 -> float32.
    """
    int8_cols = [
        "VendorID",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "is_airport",
    ]
    uint16_cols = ["PULocationID", "DOLocationID"]

    for col in int8_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].isna().any():
                df[col] = df[col].astype("int8")
            else:
                df[col] = df[col].astype("float32")

    for col in uint16_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].isna().any():
                df[col] = df[col].astype("uint16")
            else:
                df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        if col not in int8_cols and col not in uint16_cols:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype("int8")
            elif c_min >= 0 and c_max <= 65535:
                df[col] = df[col].astype("uint16")
            else:
                df[col] = df[col].astype("int32")

    return df


def load_raw_data(
    data_dir: Optional[str] = None,
    months: Iterable[int] = (1, 2, 3, 4, 5, 6),
    columns: Optional[List[str]] = None,
    sample_fraction: Optional[float] = None,
    random_state: int = 42,
    downcast: bool = True,
) -> pd.DataFrame:
    """
    Carga y concatena los archivos Parquet de los meses indicados (por defecto: 6 meses completos, Ene-Jun).
    Aplica downcasting por lote mensual para evitar picos de memoria RAM.
    """
    files = get_monthly_files(data_dir=data_dir, months=months)
    dfs = []
    for f in files:
        df_month = pd.read_parquet(f, columns=columns)
        if sample_fraction is not None and 0 < sample_fraction < 1.0:
            df_month = df_month.sample(frac=sample_fraction, random_state=random_state)
        if downcast:
            df_month = downcast_dtypes(df_month)
        dfs.append(df_month)
    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return df_all


def clean_data(df: pd.DataFrame, valid_months: Iterable[int] = (1, 2, 3, 4, 5, 6)) -> pd.DataFrame:
    """
    Limpia registros inválidos, calcula el target trip_duration en minutos, elimina variables con
    riesgo de Data Leakage y optimiza los tipos de datos (downcasting).
    """
    # 1. Calcular duración del viaje (target) en minutos en float32
    duration_min = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    ).astype("float32")

    # 2. Filtros de calidad, consistencia física y coherencia temporal
    pickup_year = df["tpep_pickup_datetime"].dt.year
    pickup_month = df["tpep_pickup_datetime"].dt.month

    mask = (
        (duration_min >= 1.0)
        & (duration_min <= 120.0)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
        & (df["PULocationID"].between(1, 265))
        & (df["DOLocationID"].between(1, 265))
        & (pickup_year == 2026)
        & (pickup_month.isin(list(valid_months)))
    )

    # 3. Seleccionar únicamente las columnas seguras (sin Data Leakage) y filas válidas
    safe_cols = [c for c in df.columns if c not in LEAKAGE_COLS]
    df_clean = df.loc[mask, safe_cols].copy()
    df_clean["trip_duration"] = duration_min.loc[mask].values

    # 4. Aplicar downcasting de tipos para minimizar uso de RAM
    df_clean = downcast_dtypes(df_clean)
    return df_clean


def load_and_clean_monthly(
    data_dir: Optional[str] = None,
    months: Iterable[int] = (1, 2, 3, 4, 5, 6),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y limpia los meses completos (100% de los datos) de forma secuencial mes a mes,
    combinando proyección de columnas y downcasting para mantener el consumo de RAM al mínimo.
    Retorna (df_clean, df_reporte_mensual).
    """
    files = get_monthly_files(data_dir=data_dir, months=months)
    cleaned_chunks = []
    report_rows = []

    for f in files:
        meta = pq.read_metadata(f)
        n_raw = meta.num_rows
        # Estimación de memoria si se cargaran las 20 columnas en tipos por defecto (int64/float64)
        # 20 columnas * 8 bytes * n_raw
        est_raw_mem_mb = (n_raw * 20 * 8) / (1024 ** 2)

        df_month_raw = pd.read_parquet(f, columns=REQUIRED_RAW_COLS)
        df_month_clean = clean_data(df_month_raw, valid_months=months)
        del df_month_raw
        gc.collect()

        n_clean = len(df_month_clean)
        clean_mem_mb = df_month_clean.memory_usage(deep=True).sum() / (1024 ** 2)

        report_rows.append(
            {
                "Archivo": os.path.basename(f),
                "Filas_Crudas": n_raw,
                "Filas_Limpias": n_clean,
                "Descartados_%": round((1 - n_clean / n_raw) * 100, 2),
                "RAM_Sin_Optimizar_MB": round(est_raw_mem_mb, 2),
                "RAM_Optimizada_MB": round(clean_mem_mb, 2),
            }
        )
        cleaned_chunks.append(df_month_clean)

    df_all_clean = pd.concat(cleaned_chunks, ignore_index=True)
    del cleaned_chunks
    gc.collect()

    df_report = pd.DataFrame(report_rows)
    return df_all_clean, df_report
