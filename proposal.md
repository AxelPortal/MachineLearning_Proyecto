## 1. Título del Proyecto

Análisis Predictivo y Estimación de la Duración de Viajes de Taxis en Nueva York

## 2. Integrantes

| N° | Nombre Completo |
| :---: | :--- |
| 1 | Axel Roberth Portal Ruiz |
| 2 | Mariel Azucena Reyes Neyra |
| 3 | Dayron Saiyuk Cueva Loayza |
| 4 | Patricia Alexandra Luyo Barrios |


## 3. Dataset elegido

Para ese proyecto se utilizaron los registros oficiales de la TLC (NYC Taxi & Limousine Commission). Se elegieron los datasets con los registros de los viajes de taxis amarillos en Nueva York (Yellow Taxi Trip Records), de enero a marzo de 2026. 

Los datasets y el diccionario de datos se encuentran en el siguiente enlace:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## 4. Pregunta predictiva 

¿Es posible predecir la duración de un viaje en taxi en la ciudad de Nueva York antes de que este inicie, utilizando un histórico de datos de 3 meses de 2026 y basándose en la ubicación de origen, destino, día, hora del servicio y número de pasajeros?

## 5. Variable objetivo

Target: trip_duration (medida en minutos). Esta se calcula como `tpep_dropoff_datetime-tpep_pickup_datetime`

Tipo de problema: Regresión (valor numérico continuo).
  
## 6. Unidad de predicción

Cada viaje individual, es decir, cada fila o registro único dentro del dataset masivo de 3 meses representa un trayecto independiente realizado por un taxi.

## 7. Variables disponibles antes de la predicción
   
   Con el objetivo de formular un problema predictivo realista, emplearemos variables que se conocen en el instante en que el pasajero aborda el taxi y solicita el servicio.
   
   **Siguiendo el diccionario de datos, nuestras variables disponibles son**:

   - **`tpep_pickup_datetime`**: La fecha y hora exacta en que se enciende el taxímetro para iniciar el servicio.
   - **`PULocationID`**: La zona de taxi (TLC Taxi Zone) de origen donde se aborda la unidad.
   - **`DOLocationID`**: La zona de taxi de destino. Asumimos que esta variable está disponible antes de la predicción, ya que el pasajero suele indicar hacia dónde va antes de que el vehículo empiece a moverse.
   - **`VendorID`**: El proveedor del sistema de tecnología (ej. Creative Mobile Technologies o Curb Mobility).
   - **`passenger_count`**: El número de pasajeros en el vehículo al inicio del viaje.
## 8. Riesgos de leakage

Evitamos que el modelo reciba información futura, lo cual representaría una fuga de datos ya que el modelo estaría trabajando con variables que no existen inicialmente. 

**Riesgos identificados:**

- El uso de la variable **`tpep_dropoff_datetime`** (hora en que se apaga el taxímetro) permitiría al algoritmo deducir la duración exacta con una simple resta matemática, lo cual anula el objetivo del modelo predictivo. De la misma manera, la variable **`trip_distance`** representa la distancia real transcurrida, la cual está directamente relacionada con la variable a predecir.
- Las diversas variables de facturación y recargos, como **`fare_amount`**, **`tolls_amount`**, **`tip_amount`**, **`total_amount`**, **`extra`**, **`mta_tax`**, **`airport_fee`**, **`improvement_surcharge`**, **`congestion_surcharge`** y **`cbd_congestion_fee`**, representan un alto riesgo, ya que se calculan, aplican o totalizan durante o al finalizar el viaje.
- El **`RatecodeID`** indica el código de la tarifa final que entra en vigor al término del viaje. De igual forma, variables como **`payment_type`** (forma de pago) y **`store_and_fwd_flag`** se registran tras la finalización del servicio.

**Primeras medidas de control:**
Como medida de prevención, se descartarán del dataset todas las variables que suponen riesgo de fuga de datos. Esta limpieza se realizará antes de cualquier exploración profunda o partición de datos para garantizar que el modelo base no se contamine.

## 9. Métricas de evaluación

El desempeño de los modelos se evaluará utilizando métricas de regresión que permitan
medir el error entre la duración real del viaje y la duración estimada por el modelo.

La métrica principal será el **Mean Absolute Error (MAE)**, ya que permite interpretar
directamente el error promedio de las predicciones en minutos. Por ejemplo, un MAE de
5 minutos indica que, en promedio, las predicciones se encuentran a aproximadamente
5 minutos del valor real.

Como métrica secundaria se utilizará el **Root Mean Squared Error (RMSE)**, que
penaliza en mayor medida los errores de mayor magnitud. Esta métrica permitirá
identificar si existen predicciones que presentan errores considerablemente superiores
al comportamiento habitual.

La comparación de los modelos se realizará utilizando las mismas métricas y el mismo
conjunto de evaluación. Además, los resultados serán comparados con el modelo
baseline definido previamente, con el objetivo de determinar si los modelos de
Machine Learning logran mejorar la referencia inicial.

Las métricas consideradas serán:

- **MAE (Mean Absolute Error):** error absoluto promedio de las predicciones,
  expresado en minutos.
- **RMSE (Root Mean Squared Error):** raíz del error cuadrático medio, expresado
  en minutos y con mayor sensibilidad frente a errores grandes.

Un menor valor de MAE y RMSE representará un menor error de predicción de la
duración del viaje.

## 10. Estrategia de validación

La validación del proyecto considerará la naturaleza temporal de los registros de
viajes de taxi. Debido a que el objetivo es predecir la duración de un viaje a partir
de información disponible antes de su inicio, se evitará utilizar información futura
durante el entrenamiento del modelo.

Para la evaluación se utilizará una separación temporal de los datos. Los registros
correspondientes a los primeros meses disponibles serán utilizados para el
entrenamiento, mientras que el último mes será reservado para evaluar el desempeño
de los modelos sobre datos que no fueron utilizados durante su entrenamiento.

En este proyecto se considerarán los registros de **enero y febrero de 2026 como
conjunto de entrenamiento**, mientras que los registros de **marzo de 2026 serán
utilizados como conjunto de evaluación**.

Esta estrategia permitirá simular de mejor manera un escenario real de predicción,
en el cual el modelo utiliza información histórica para realizar predicciones sobre
viajes posteriores.

Además, la separación temporal permitirá reducir el riesgo de **data leakage**,
evitando que información correspondiente al periodo de evaluación sea utilizada para
entrenar o ajustar el modelo.

El desempeño obtenido sobre el conjunto de marzo será comparado mediante las métricas
MAE y RMSE, utilizando el modelo baseline como referencia.


## 11. Modelo baseline

Como modelo baseline simple se utilizó un predictor constante basado en la **mediana de la variable objetivo `trip_duration`**, expresada en minutos. El objetivo de este baseline es establecer una referencia inicial de desempeño que permita posteriormente evaluar si los modelos de Machine Learning desarrollados logran mejorar la predicción de la duración de los viajes.

Para mantener la naturaleza temporal del problema, se realizó una separación de los datos utilizando los meses disponibles. Los registros correspondientes a **enero y febrero de 2026** fueron utilizados como conjunto de entrenamiento, mientras que los registros correspondientes a **marzo de 2026** fueron reservados como conjunto de evaluación. De esta manera, la información utilizada para calcular el valor del baseline proviene únicamente de periodos anteriores al conjunto de evaluación.

La mediana de `trip_duration` calculada sobre el conjunto de entrenamiento fue de **13.93 minutos**. Este valor se utilizó como predicción constante para todos los viajes pertenecientes al conjunto de evaluación. Se seleccionó la mediana como baseline debido a que la variable objetivo presenta una distribución con valores extremos, como se observó durante el análisis exploratorio. Por ello, la mediana constituye una referencia menos sensible a valores atípicos que una predicción basada en el promedio.

El conjunto de entrenamiento estuvo compuesto por **6,958,926 registros**, mientras que el conjunto de evaluación estuvo compuesto por **3,858,758 registros**. Para medir el desempeño del baseline se utilizaron dos métricas de regresión: **MAE (Mean Absolute Error)** como métrica principal y **RMSE (Root Mean Squared Error)** como métrica secundaria.

Los resultados obtenidos fueron los siguientes:

| Métrica | Resultado |
|---|---:|
| Registros de entrenamiento | 6,958,926 |
| Registros de evaluación | 3,858,758 |
| Mediana utilizada | 13.93 minutos |
| MAE | 9.08 minutos |
| RMSE | 14.36 minutos |

El baseline obtuvo un **MAE de 9.08 minutos**, lo que significa que, en promedio, la diferencia absoluta entre la duración real de los viajes y la predicción realizada por este modelo simple fue de aproximadamente 9.08 minutos. Por otro lado, el **RMSE de 14.36 minutos** presenta un valor superior al MAE debido a que esta métrica otorga mayor peso a los errores de mayor magnitud.

Estos resultados constituyen la **línea base del proyecto** y servirán como punto de comparación para los modelos de Machine Learning que se desarrollarán posteriormente. De esta manera, el desempeño de cada modelo podrá evaluarse no solo de forma individual, sino también respecto a una estrategia sencilla que utiliza únicamente el comportamiento central de la variable objetivo.

Finalmente, este baseline permite establecer una referencia reproducible y fácil de interpretar antes de incorporar variables predictoras y modelos más complejos. La mejora respecto a esta referencia será analizada posteriormente utilizando el mismo conjunto de evaluación y las mismas métricas, con el fin de realizar una comparación consistente entre las diferentes estrategias de modelamiento.

## 12. Riesgos técnicos

El proyecto presenta diversos riesgos técnicos relacionados principalmente con el
volumen de datos, la calidad de la información y el procesamiento necesario para
el desarrollo de los modelos de Machine Learning.

Uno de los principales riesgos es el **alto volumen de datos**, debido a la
cantidad de registros correspondientes a los tres meses seleccionados. El
procesamiento de un conjunto de datos de este tamaño puede generar un consumo
elevado de memoria y aumentar los tiempos de ejecución de las operaciones de
limpieza, transformación, exploración y entrenamiento.

Otro riesgo corresponde a la **presencia de valores faltantes y registros
atípicos**. Las variables del dataset pueden contener datos incompletos,
duraciones de viaje inusuales o valores extremos que podrían afectar el
entrenamiento y el desempeño de los modelos. Por ello, será necesario realizar
una revisión de la calidad de los datos y establecer criterios de tratamiento
antes del modelamiento.

También existe un riesgo de **data leakage**, debido a que algunas variables
del dataset pueden contener información generada durante o después del viaje.
Estas variables no estarían disponibles en el momento en que se desea realizar
la predicción. Por ello, se deberá verificar que las variables utilizadas como
predictoras representen únicamente información disponible antes del inicio del
viaje.

Otro riesgo está relacionado con la **alta cardinalidad de las variables de
ubicación**, como las zonas de origen y destino. Una transformación inadecuada
de estas variables podría incrementar considerablemente la cantidad de
características utilizadas por los modelos y aumentar el costo computacional.

Asimismo, existe un riesgo de **reproducibilidad**, debido a que cambios en las
versiones de las librerías, parámetros o procesos de preparación de los datos
podrían generar resultados diferentes. Para reducir este riesgo, se documentará
el proceso utilizado y se mantendrán consistentes las configuraciones del
proyecto.

Finalmente, se considera un riesgo de **escalabilidad**. Si posteriormente se
incrementa el periodo de análisis o se incorporan nuevos registros, aumentarán
los requerimientos de almacenamiento, memoria y tiempo de procesamiento. Por
ello, se buscará mantener un procesamiento eficiente y utilizar únicamente las
variables necesarias para cada etapa del proyecto.


## 13. Plan de trabajo

El desarrollo del proyecto se estructurará de manera progresiva. A continuación, se detalla lo trabajado durante la presente semana para la entrega previa y el cronograma planificado para las semanas restantes hasta la entrega final en la semana 16.

### Lo trabajado esta semana (Semana 7 - Entrega Previa)

Durante esta semana, el equipo se enfocó en comprender la viabilidad del proyecto, formular correctamente el problema predictivo y realizar la exploración inicial de los datos (notebook `01_exploracion_inicial.ipynb`).

La elaboración de los puntos del documento `proposal.md` se dividió de la siguiente manera:

* **Dayron:** Estructuración de la pregunta predictiva (punto 4), definición de la variable objetivo (punto 5) y delimitación de la unidad de predicción (punto 6).
* **Mariel:** Análisis de las variables disponibles antes de la predicción (punto 7), identificación de riesgos de leakage (punto 8) y redacción del plan de trabajo (punto 13).
* **Axel:** Selección y justificación de la métrica principal y secundaria (punto 9), y diseño del plan de validación (punto 10).
* **Patricia:** Formulación del modelo baseline (punto 11) y análisis de los riesgos técnicos del proyecto (punto 12).

### Plan de trabajo semanas restantes (Semanas 8 a 16 - Entrega Final)

Para cumplir con lo requerido en entrega final, las próximas tareas se han distribuido en las siguientes fases:

| Fase / Semanas | Actividades y Entregables | Requisito de la Rúbrica | Responsables |
| :--- | :--- | :--- | :--- |
| **Semanas 8 a 9:**<br>Preprocesamiento | Limpieza de datos, eliminación de leakage y creación de variables. | Pipeline reproducible y control de leakage. | Mariel y Axel |
| **Semanas 10 a 11:**<br>Validación y Baseline | División temporal y construcción del modelo baseline. | Validación correcta y baseline ejecutado. | Dayron y Patricia |
| **Semanas 12 a 13:**<br>Modelado Avanzado | Entrenamiento y comparación de 3 familias de modelos. | Modelado y mejora frente al baseline. | Axel y Dayron |
| **Semanas 14 a 15:**<br>Evaluación | Comparación de métricas y análisis de errores. | Interpretabilidad y análisis de errores. | Mariel y Patricia |
| **Semana 16:**<br>Documentación | Informe final, limpieza del código y presentación. | Reproducibilidad e informe final. | Todo el equipo |
