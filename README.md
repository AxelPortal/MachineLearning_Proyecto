# Proyecto de Machine Learning: Predicción de Duración de Viajes en Taxi (NYC)

Este repositorio contiene el desarrollo del proyecto de Machine Learning para predecir la duración de los viajes de los taxis amarillos (*Yellow Taxis*) de la ciudad de Nueva York antes de que inicien, utilizando registros históricos oficiales de la TLC (Taxi & Limousine Commission) correspondientes a **6 meses completos (Enero a Junio de 2026 — 22,836,530 viajes crudos y 16,409,232 viajes limpios válidos)**.

---

## 1. Pregunta Predictiva y Variable Objetivo

* **Pregunta predictiva:** ¿Es posible predecir la duración de un viaje en taxi en la ciudad de Nueva York antes de que este inicie, utilizando un histórico de datos de 6 meses completos de 2026 y basándose en la ubicación de origen, destino, día, hora del servicio y número de pasajeros?
* **Tipo de problema:** Regresión (valor numérico continuo).
* **Variable objetivo (*Target*):** `trip_duration` (medida en minutos), calculada como la diferencia entre la hora de llegada (`tpep_dropoff_datetime`) y la hora de inicio (`tpep_pickup_datetime`).
* **Unidad de análisis:** Cada fila representa un viaje individual completado.

---

## 2. Estructura del Proyecto

```text
MachineLearning_Proyecto/
├── data/
│   ├── raw/                                 # Archivos Parquet originales (Enero - Junio 2026)
│   │   ├── yellow_tripdata_2026-01.parquet
│   │   ├── yellow_tripdata_2026-02.parquet
│   │   ├── yellow_tripdata_2026-03.parquet
│   │   ├── yellow_tripdata_2026-04.parquet
│   │   ├── yellow_tripdata_2026-05.parquet
│   │   └── yellow_tripdata_2026-06.parquet
│   ├── processed/                           # Dataset limpio y optimizado (16.4M filas, zstd)
│   │   └── yellow_tripdata_processed.parquet
│   └── README.md                            # Diccionario y descripción de datos
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb         # Análisis Exploratorio de Datos (EDA)
│   ├── 02_limpieza_features.ipynb           # Limpieza, Downcasting e Ingeniería de Features
│   ├── 03_modelos.ipynb                     # Partición Temporal (~80/20), Baselines y Modelos
│   └── 04_analisis_errores.ipynb            # Análisis de errores por segmento e interpretabilidad
├── src/
│   ├── data.py                              # Carga secuencial, proyección de columnas y downcasting
│   ├── features.py                          # Extracción de features en int8/float32 y preprocesador
│   ├── evaluate.py                          # Cálculo consistente de MAE, RMSE y R²
│   └── train.py                             # Script de entrenamiento reproducible
├── outputs/
│   └── metrics.json                         # Registro estructurado de métricas de evaluación
├── reports/
│   ├── figures/                             # Gráficos y visualizaciones exportadas
│   └── informe_final.md                     # Informe técnico final del proyecto
├── proposal.md                              # Propuesta técnica del proyecto
├── requirements.txt                         # Dependencias del proyecto
└── README.md                                # Documentación principal
```

---

## 3. Optimización de Memoria RAM (*Downcasting* y Proyección de Columnas)

Para procesar el **100% de los 6 meses (22.84 millones de registros crudos $\rightarrow$ 16.41 millones de registros limpios)** sin submuestreo en equipos de 16 GB de RAM, se implementaron tres técnicas en `src/data.py` y `src/features.py`:

1. **Proyección de columnas en lectura (`columns=REQUIRED_RAW_COLS`):** Se leen desde Parquet únicamente las 6 columnas requeridas antes del inicio del viaje, descartando en disco las 14 columnas de *Data Leakage*.
2. **Procesamiento secuencial por lote mensual:** Cada mes se filtra y limpia de forma independiente antes de concatenarse en memoria.
3. **Reducción de tipos numéricos (*Downcasting*):**
   * `int64` (8 bytes) $\rightarrow$ `int8` (1 byte) para `VendorID`, `passenger_count`, `pickup_hour`, `pickup_dayofweek`, `pickup_month`, `is_weekend`, `is_rush_hour`, `is_airport` (**ahorro del 87.5%**).
   * `int64` (8 bytes) $\rightarrow$ `uint16` (2 bytes) para `PULocationID` y `DOLocationID` (**ahorro del 75%**).
   * `float64` (8 bytes) $\rightarrow$ `float32` (4 bytes) para `trip_duration`, `sin_hour`, `cos_hour` (**ahorro del 50%**).
   * **Impacto total:** Los 16,409,232 viajes con sus 13 columnas ocupan apenas **375.58 MB en RAM** (frente a >3,480 MB sin optimizar, logrando un **ahorro total superior al 89%**) y **58.51 MB en disco** con compresión `zstd`.

---

## 4. Estrategia de Partición Temporal por Meses (~80/20) y Baselines

Respetando la naturaleza temporal del fenómeno y evitando cualquier fuga de información futura (*look-ahead bias*), la partición se realiza estrictamente por meses completos:

* **Entrenamiento (Enero a Mayo de 2026 — 5 meses):** `13,686,384` viajes (**83.41%**).
* **Evaluación / Testing (Junio de 2026 — 1 mes):** `2,722,848` viajes (**16.59%**).

### Resultados de la Línea Base (Evaluados en Junio 2026):

| Modelo | MAE (min) | RMSE (min) | $R^2$ | Mejora MAE vs. Baseline 0 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 0: Mediana Global (`13.07` min)** | 9.1992 | 14.2975 | -0.0852 | 0.00% |
| **Baseline 1: Dummy por Par Origen-Destino** | **4.9237** | **7.7802** | **0.6787** | **+46.48%** |
| **Baseline 2: Regresión Ridge Regularizada** | 7.3279 | 10.2034 | 0.4473 | +20.34% |

---

## 5. Requisitos e Instrucciones de Reproducción

```bash
pip install -r requirements.txt
```

Para regenerar el flujo completo:
1. Ejecutar `notebooks/01_exploracion_inicial.ipynb` para revisar el análisis exploratorio de los 6 meses.
2. Ejecutar `notebooks/02_limpieza_features.ipynb` para procesar los 6 meses con *downcasting* y generar `data/processed/yellow_tripdata_processed.parquet`.
3. Ejecutar `notebooks/03_modelos.ipynb` para reproducir la partición temporal (Ene–May vs. Jun), los baselines y el entrenamiento de modelos.
