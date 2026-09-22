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
