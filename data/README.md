# Directorio de Datos (`data/`)

## 1. Datos Crudos (`data/raw/`)
Contiene los archivos oficiales de viajes en taxis amarillos de Nueva York (*NYC Yellow Taxi Trip Records*) publicados por la **NYC Taxi & Limousine Commission (TLC)** en formato `.parquet`.

Para el proyecto se emplean los **6 meses completos del primer semestre de 2026 (Enero a Junio)** sin submuestreo:

| Archivo | Mes | Registros Crudos | Registros Limpios Válidos |
| :--- | :--- | ---: | ---: |
| `yellow_tripdata_2026-01.parquet` | Enero 2026 | 3,724,889 | 2,537,460 |
| `yellow_tripdata_2026-02.parquet` | Febrero 2026 | 3,399,866 | 2,289,402 |
| `yellow_tripdata_2026-03.parquet` | Marzo 2026 | 3,952,451 | 2,903,272 |
| `yellow_tripdata_2026-04.parquet` | Abril 2026 | 3,831,240 | 2,929,587 |
| `yellow_tripdata_2026-05.parquet` | Mayo 2026 | 4,090,836 | 3,026,647 |
| `yellow_tripdata_2026-06.parquet` | Junio 2026 | 3,837,248 | 2,722,864 |
| **Total Semestre (6 Meses)** | **Ene – Jun 2026** | **22,836,530** | **16,409,232** |

## 2. Datos Procesados (`data/processed/`)
* **`yellow_tripdata_processed.parquet`** (`58.51 MB` en disco con compresión `zstd`, `375.58 MB` en RAM):
  Contiene los **16,409,232 viajes válidos**, libres de *Data Leakage* y con reducción de tipos de datos (*downcasting* a `int8`, `uint16` y `float32`).

### Esquema de Columnas y Tipos Optimizados:
* `VendorID` (`int8`): Proveedor del taxímetro (1 byte).
* `passenger_count` (`int8`): Número de pasajeros entre 1 y 6 (1 byte).
* `PULocationID` (`uint16`): Zona TLC de origen entre 1 y 265 (2 bytes).
* `DOLocationID` (`uint16`): Zona TLC de destino entre 1 y 265 (2 bytes).
* `trip_duration` (`float32`): **Variable objetivo** en minutos, filtrada entre 1 y 120 min (4 bytes).
* `pickup_hour` (`int8`): Hora de inicio del viaje de 0 a 23 (1 byte).
* `pickup_dayofweek` (`int8`): Día de la semana de 0=Lunes a 6=Domingo (1 byte).
* `pickup_month` (`int8`): Mes del viaje de 1 a 6, utilizado para el split temporal Ene–May vs. Jun (1 byte).
* `is_weekend` (`int8`): Indicador binario de fin de semana (1 byte).
* `is_rush_hour` (`int8`): Indicador binario de hora punta laboral 7–9 AM y 16–19 PM (1 byte).
* `sin_hour` (`float32`): Componente seno de la hora del día (4 bytes).
* `cos_hour` (`float32`): Componente coseno de la hora del día (4 bytes).
* `is_airport` (`int8`): Indicador binario de origen o destino en aeropuertos JFK, LGA o EWR (1 byte).
