4. Pregunta predictiva

¿Es posible predecir la duración de un viaje en taxi en la ciudad de Nueva York antes de que este inicie, utilizando un histórico de 3 a 12 meses de datos, basándose en la ubicación de origen, destino, día y hora del servicio?

5. Variable objetivo

Target: trip_duration (medida en minutos).

Tipo de problema: Regresión (valor numérico continuo).
  
6. Unidad de predicción

Cada viaje individual, es decir, cada fila o registro único dentro del dataset masivo de 3 a 12 meses representa un trayecto independiente realizado por un taxi.

7. Variables disponibles antes de la predicción
   
   Con el objetivo de formular un problema predictivo realista, emplearemos variables que se conocen en el instante en que el pasajero aborda el taxi y solicita el servicio.
   
   **Siguiendo el diccionario de datos, nuestras variables disponibles son**:

   - **`tpep_pickup_datetime`**: La fecha y hora exacta en que se enciende el taxímetro para iniciar el servicio.
   - **`PULocationID`**: La zona de taxi (TLC Taxi Zone) de origen donde se aborda la unidad.
   - **`DOLocationID`**: La zona de taxi de destino. Asumimos que esta variable está disponible antes de la predicción, ya que el pasajero suele indicar hacia dónde va antes de que el vehículo empiece a moverse.
   - **`VendorID`**: El proveedor del sistema de tecnología (ej. Creative Mobile Technologies o Curb Mobility).
   - **`passenger_count`**: El número de pasajeros en el vehículo al inicio del viaje.
8. Riesgos de leakage

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

Como modelo baseline se utilizará un predictor constante basado en la mediana de
la variable objetivo `trip_duration`. Este modelo permitirá establecer una
referencia sencilla del desempeño que se puede obtener sin utilizar un modelo
predictivo complejo.

La mediana de `trip_duration` será calculada únicamente a partir de los datos
del conjunto de entrenamiento. Posteriormente, este mismo valor será utilizado
como predicción para todos los viajes pertenecientes al conjunto de evaluación.

Se utilizará la mediana debido a que la duración de los viajes puede presentar
valores atípicos que podrían afectar una referencia basada en el promedio. De
esta manera, se obtiene una referencia más robusta frente a observaciones
extremas.

El desempeño del baseline será evaluado utilizando MAE como métrica principal y
RMSE como métrica secundaria. Estos resultados servirán como punto de
comparación para determinar si los modelos de Machine Learning desarrollados
posteriormente logran reducir el error de predicción respecto a esta referencia
simple.

El cálculo del baseline respetará la separación definida para la validación,
evitando utilizar información del conjunto de evaluación para determinar el
valor utilizado como predicción.

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

El desarrollo del proyecto se realizará de manera progresiva, comenzando con la
preparación y exploración de los datos y avanzando posteriormente hacia la
construcción, evaluación y comparación de los modelos de Machine Learning.

Las actividades planificadas son las siguientes:

1. **Finalización de la exploración y preparación de los datos:** revisar la
   calidad de los datos, valores faltantes, registros atípicos y variables
   disponibles para la predicción.

2. **Definición de variables y validación:** seleccionar las variables que
   estarán disponibles antes del inicio del viaje, controlar posibles casos de
   data leakage y establecer el esquema de validación.

3. **Implementación del modelo baseline:** establecer la mediana de
   `trip_duration` como referencia y evaluar su desempeño mediante las métricas
   definidas.

4. **Preparación de los datos para Machine Learning:** realizar las
   transformaciones necesarias sobre las variables seleccionadas y preparar los
   conjuntos de entrenamiento y evaluación.

5. **Entrenamiento de modelos:** desarrollar y entrenar los modelos de
   Machine Learning seleccionados para el problema de regresión.

6. **Evaluación y comparación:** comparar el desempeño de los modelos utilizando
   MAE y RMSE, tomando el baseline como referencia.

7. **Análisis de resultados:** identificar las variables y modelos que presenten
   un comportamiento relevante, analizar los errores obtenidos y revisar las
   limitaciones del enfoque utilizado.

8. **Documentación y presentación final:** consolidar los resultados, documentar
   el proceso realizado en el repositorio de GitHub y preparar las conclusiones
   y presentación del proyecto.
