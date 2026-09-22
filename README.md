# Proyecto de Machine Learning: Predicción de Duración de Viajes en Taxi (NYC)

Este repositorio contiene el desarrollo del proyecto de Machine Learning para predecir la duración de los viajes de los taxis amarillos (*Yellow Taxis*) de la ciudad de Nueva York antes de que inicien, utilizando registros históricos oficiales de la TLC (Taxi & Limousine Commission).

---

## 1. Pregunta Predictiva y Variable Objetivo

* **Pregunta predictiva:** ¿Es posible predecir la duración de un viaje en taxi en la ciudad de Nueva York antes de que este inicie, utilizando un histórico de datos y basándose en la ubicación de origen, destino, día y hora del servicio?
* **Tipo de problema:** Regresión (valor numérico continuo).
* **Variable objetivo (*Target*):** `trip_duration` (medida en minutos), calculada como la diferencia entre la hora de llegada (`tpep_dropoff_datetime`) y la hora de inicio (`tpep_pickup_datetime`).
* **Unidad de análisis:** Cada fila representa un viaje individual completado.

---

## 2. Estructura del Proyecto

```text
MachineLearning_Proyecto/
├── data/                                # Archivos de datos en formato Parquet
│   ├── yellow_tripdata_2026-01.parquet
│   ├── yellow_tripdata_2026-02.parquet
│   ├── yellow_tripdata_2026-03.parquet
├── notebooks/                           # Jupyter Notebooks de exploración y modelado
│   └── 01_exploracion_inicial.ipynb     # Análisis Exploratorio de Datos (EDA)
├── src/                                 # Scripts y módulos auxiliares
├── proposal.md                          # Propuesta inicial del proyecto
└── README.md                            # Documentación del proyecto
```

---

## 3. Requisitos e Instalación

### Requisitos previos
* **Python 3.10** o superior.
* Gestor de paquetes `pip` o entorno Conda.

### Instalación de dependencias
Para ejecutar la exploración y procesar los archivos Parquet, instala las librerías necesarias ejecutando en tu terminal:

```bash
pip install pandas pyarrow fastparquet numpy matplotlib seaborn jupyter scikit-learn
```

> **Nota:** `pyarrow` o `fastparquet` son necesarios como motor interno de `pandas` para la lectura de archivos `.parquet`.

---

## 4. Instrucciones para Reproducir la Exploración

Sigue estos pasos para reproducir exactamente todo el análisis exploratorio:

### Paso 1: Clonar o abrir el proyecto
Abre una terminal en la carpeta raíz del proyecto (`MachineLearning_Proyecto`).

### Paso 2: Verificar la ubicación de los datos
Asegúrate de que los archivos `.parquet` se encuentren dentro de la carpeta `data/`. Los datos pueden ser descargados desde la página oficial de la [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):
* `data/yellow_tripdata_2026-01.parquet`
* `data/yellow_tripdata_2026-02.parquet`
* `data/yellow_tripdata_2026-03.parquet`

### Paso 3: Iniciar el entorno de Jupyter
Ejecuta en tu terminal:
```bash
jupyter notebook
```
o abre la carpeta del proyecto directamente en **Visual Studio Code** con la extensión de Jupyter instalada.

### Paso 4: Ejecutar el Notebook
1. Navega a la carpeta `notebooks/` y abre el archivo [`01_exploracion_inicial.ipynb`](notebooks/01_exploracion_inicial.ipynb).
2. Selecciona el kernel de Python correspondiente.
3. Ejecuta todas las celdas secuencialmente o presiona **"Run All"** / **"Ejecutar todo"**.

---

## 5. Contenido del Análisis Exploratorio (EDA)

El notebook `01_exploracion_inicial.ipynb` ejecuta de forma automatizada las siguientes secciones sobre los **3 meses de datos (+11 millones de filas)**:

1. **Carga de Datos:** Lectura eficiente de los 3 meses iniciales con gestión dinámica de rutas.
2. **Dimensiones y Tipos de Datos:** Inspección de filas, columnas y tipos de datos mediante `df.info()` y `df.describe()`.
3. **Diagnóstico de Valores Faltantes:** Tabla cuantitativa y gráfico de barras del porcentaje de nulos por variable (analizando campos como `passenger_count`, `RatecodeID`, recargos por congestión, etc.).
4. **Variable Objetivo (`trip_duration`):** Cálculo del tiempo de viaje en minutos y cálculo de estadísticas descriptivas con percentiles (1%, 5%, 25%, 50%, 75%, 95%, 99%).
5. **5 Visualizaciones Clave del Fenómeno:**
   * **Distribución:** Histograma y Boxplot de la duración del viaje (rango 1 a 120 min).
   * **Patrón Horario:** Evolución de la duración media y mediana a lo largo de las 24 horas del día (identificación de horas punta y congestión vehicular).
   * **Comportamiento Semanal:** Duración promedio de lunes a domingo.
   * **Distancia vs. Duración:** Gráfico de densidad Hexbin en 2D (con escala logarítmica) para visualizar la relación espacial sin saturación de memoria.
   * **Matriz de Correlación:** Mapa de calor (*Heatmap*) de correlación de Pearson entre variables continuas.
6. **Auditoría de Outliers y Registros Sospechosos:**
   * Conteo y porcentaje de viajes con duración $\le 0$ o $> 3$ horas.
   * Viajes con distancia cero y duración positiva (viajes estáticos/errores de taxímetro).
   * Distancias extremas ($> 100$ millas) y conteos irregulares de pasajeros ($0$ o $> 6$).
   * Comparación visual de la variable objetivo con y sin outliers.
7. **Modelo Baseline:**
   * División temporal de los datos (Entrenamiento: Enero y Febrero | Evaluación: Marzo).
   * Cálculo de una predicción estática basada en la mediana de `trip_duration`.
   * Evaluación del modelo con métricas MAE (*Mean Absolute Error*) y RMSE (*Root Mean Squared Error*).