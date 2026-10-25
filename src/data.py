import pandas as pd
import glob
import os

def get_data_dir(subdir="raw"):
    """Localiza la carpeta de datos independientemente de si se ejecuta desde root o notebooks/."""
    candidates = [
        os.path.join("data", subdir),
        os.path.join("..", "data", subdir),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", subdir)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join("data", subdir)

def load_raw_data(data_dir=None, sample_fraction=None, random_state=42):
    """Carga y concatena los archivos Parquet de raw. Permite submuestreo opcional."""
    if data_dir is None:
        data_dir = get_data_dir("raw")
        
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos parquet en {data_dir}")
        
    dfs = []
    for f in files:
        df_month = pd.read_parquet(f)
        if sample_fraction is not None and 0 < sample_fraction < 1.0:
            df_month = df_month.sample(frac=sample_fraction, random_state=random_state)
        dfs.append(df_month)
        
    return pd.concat(dfs, ignore_index=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia registros inválidos y calcula el target trip_duration en minutos sin Data Leakage."""
    df = df.copy()
    
    # 1. Calcular duración del viaje (target) en minutos
    duration_min = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    df['trip_duration'] = duration_min
    
    # 2. Filtros de calidad y consistencia
    mask = (
        (df['trip_duration'] >= 1.0) & (df['trip_duration'] <= 120.0) &
        (df['passenger_count'] >= 1) & (df['passenger_count'] <= 6) &
        (df['PULocationID'].between(1, 265)) &
        (df['DOLocationID'].between(1, 265))
    )
    df_clean = df[mask].copy()
    
    # 3. Eliminar columnas con riesgo de data leakage
    cols_to_drop = [
        'tpep_dropoff_datetime', 'trip_distance', 'fare_amount', 'extra', 
        'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 
        'total_amount', 'congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee',
        'RatecodeID', 'payment_type', 'store_and_fwd_flag'
    ]
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    return df_clean
