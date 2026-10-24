# Informe Final: Predicción de Duración de Viajes en Taxis de NYC

## 1. Resumen ejecutivo
*Breve síntesis del problema abordado, la metodología aplicada, el mejor modelo obtenido (comparado contra el baseline) y los hallazgos principales de interpretabilidad y errores.*

## 2. Descripción del problema
*Delimitación de la tarea predictiva supervisada (regresión para estimar la duración de los viajes en taxi), justificación de su relevancia operativa y definición clara de la variable objetivo y la unidad de predicción.*

## 3. Descripción del dataset
*Detalle de la fuente de datos (NYC Taxi & Limousine Commission), periodo seleccionado (3 a 12 meses), volumen inicial de registros, estructura de las tablas, licencias y las decisiones de muestreo adoptadas.*

## 4. Limpieza y preparación de datos
*Documentación del pipeline de preprocesamiento, tratamiento de valores faltantes, filtrado de registros anómalos u outliers geográficos/temporales, y las medidas implementadas para prevenir el data leakage.*

## 5. Análisis exploratorio
*Síntesis de los patrones visuales y estadísticos clave descubiertos en los datos (distribución de la variable objetivo, correlaciones espaciales y temporales, y comportamiento de las características principales).*

## 6. Estrategia de partición y validación
*Descripción de la estrategia de división de los datos (train/validation/test), justificación del esquema de validación para evitar fugas de información temporal o espacial, y selección de las métricas principal (ej. RMSE / MAE) y secundarias.*

## 7. Baselines
*Definición y justificación de los modelos baseline implementados (ej. media, mediana o regresión lineal), junto con sus resultados cuantitativos iniciales para establecer un punto de comparación honesto.*

## 8. Modelos entrenados
*Comparativa de al menos tres familias de modelos evaluadas (ej. modelos lineales regularizados, árboles de decisión, gradient boosting como XGBoost/LightGBM) y la evolución de su rendimiento frente al baseline.*

## 9. Búsqueda de hiperparámetros
*Detalle de la metodología utilizada para la optimización de hiperparámetros (búsqueda en grilla o aleatoria), rangos evaluados y los mejores parámetros seleccionados para el modelo final.*

## 10. Evaluación final
*Desempeño cuantitativo definitivo del modelo seleccionado evaluado estrictamente sobre el conjunto de prueba (test set), incluyendo métricas de error absolutas y relativas.*

## 11. Interpretación
*Análisis de la importancia de variables y explicación global/local del modelo utilizando valores SHAP (incluyendo gráficos como summary plots y dependence plots para hora punta, distancias y condiciones climáticas).*
* ![Summary SHAP](figures/shap_summary.png)
* ![Dependence SHAP](figures/shap_dependence_distancia.png)

## 12. Análisis de errores
*Evaluación detallada del comportamiento del modelo en subgrupos críticos (ej. trayectos hacia aeropuertos como JFK o LaGuardia, viajes de madrugada, y outliers de distancia o duración larga).*
* ![Error por Subgrupos](figures/error_por_subgrupos.png)

## 13. Riesgos éticos, sesgos y limitaciones
*Discusión crítica sobre posibles sesgos algorítmicos (ej. distorsiones en tarifas o tiempos por zonas geográficas), riesgos éticos asociados al uso de datos de movilidad urbana, y limitaciones técnicas del modelo ante eventos imprevistos.*

## 14. Trabajo futuro
*Propuestas de mejora a mediano y largo plazo (incorporación de datos meteorológicos externos en tiempo real, modelado espacial avanzado mediante grafos o redes neuronales, y despliegue en producción).*

## 15. Instrucciones de reproducción
*Guía paso a paso para clonar el repositorio, configurar el entorno virtual, instalar las dependencias desde el archivo `requirements.txt`, fijar las semillas y ejecutar los scripts o notebooks en orden para regenerar todos los resultados.*
