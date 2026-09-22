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
   - **`passenger_count`**: El número de pasajeros en el vehículo al iniciar el viaje.
9. Riesgos de leakage
   Evitamos que el modelo reciba información futura, lo cual representaría una fuga de datos ya que el modelo estaría trabajando con variables que no existen inicialmente.
   **Los riesgos que identificamos fueron:**
   - El uso de la variable **`tpep_dropoff_datetime`** (hora en que se apaga el taxímetro) permitiría al algoritmo deducir la duración exacta con una simple resta matemática, lo cual anula el objetivo del modelo predictivo. De la misma manera, la variable **`trip_distance`** representa la distancia real transcurrida, la cual es la variable a predecir.
   - Las diversas variables de facturación, como **`fare_amount`**, **`tolls_amount`**, **`tip_amount`** y **`total_amount`**, representan un alto riesgo, ya que se calculan o totalizan durante o al finalizar el viaje.
   - El **`RatecodeID`** representa la tarifa final que se aplica al culminar el viaje. De igual forma, variables como **`payment_type`** (forma de pago) y **`store_and_fwd_flag`** se registran tras la finalización del servicio.
   **Primeras medidas de control:**
   - Como medida de prevención, se descartarán del dataset todas las variables que suponen riesgo de fuga de datos. Esta limpieza se realizará antes de cualquier exploración profunda o partición de datos para garantizar que el modelo base no se contamine.
