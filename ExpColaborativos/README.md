# Experimento Colaborativo 16 — Workflow Junior (Grupo B)

Este directorio contiene los notebooks de ejecución, orquestación y análisis desarrollados para el **Experimento 16** sobre el **Workflow Junior - Grupo B** (`z621_WorkFlow_01_junior_grupoB.ipynb`).

---

## Objetivo del Experimento 16

El propósito central del experimento es evaluar **si conviene o no quitar los campos clave (`numero_de_cliente` como ID y `foto_mes` como indicador temporal/snapshot) al momento de entrenar el modelo**.

Para responder esta pregunta, se definieron **4 variantes** evaluadas a lo largo de **10 semillas** distintas (40 corridas en total por tanda):

1. **`baseline` (código 1):** Entrena manteniendo todos los campos habituales (incluyendo `numero_de_cliente` y `foto_mes`).
2. **`without_id` (código 2):** Quita el campo `numero_de_cliente` del entrenamiento.
3. **`without_month_snapshot` (código 3):** Quita el campo `foto_mes` del entrenamiento.
4. **`without_both` (código 4):** Quita ambos campos clave (`numero_de_cliente` y `foto_mes`).

---

## Arquitectura: Orquestador y Worker

Para poder correr todas las pruebas de forma consecutiva, desatendida y tolerante a cortes, armamos una arquitectura básica compuesta por un **Orquestador** (en Python usando `papermill`) y un **Worker** (el workflow en R parametrizado):

- **Orquestador (`*orchestrator*.ipynb`):**
  - Se encarga de recorrer la grilla de combinaciones entre las 4 variantes y las 10 semillas.
  - Le inyecta al worker los parámetros de cada corrida: el código de la variante (`k_variant_code`), la semilla (`k_seed_val`) y el índice de la semilla (`k_seed_idx`).
  - Antes de lanzar cada ejecución, verifica si ya existe el archivo `PARAM.yml` en la carpeta de salida del experimento; si la corrida ya terminó previamente, la saltea automáticamente para poder retomar ejecuciones sin perder progreso.
- **Worker (`*worker*.ipynb`):**
  - Toma los parámetros enviados por el orquestador en su primera celda (`# Tag: parameters`), configura dinámicamente el identificador del experimento y la variante activa, y ejecuta de punta a punta el flujo del modelo (preprocesamiento, Grid Search de hiperparámetros, entrenamiento final y cálculo de la ganancia suavizada).

*(Los archivos `orchestrator.ipynb` y `worker.ipynb` son los prototipos iniciales que armamos para probar esta mecánica de inyección de parámetros).*

---

## El Centro del Cambio: Modificación de `campos_buenos`

El corazón técnico del Experimento 16 vive en el worker, justo en el bloque donde se define el vector `campos_buenos` (las columnas predictoras que efectivamente se usan para construir los datasets de LightGBM: `dtrain`, `dvalidate`, el entrenamiento final y la predicción en `dfuture`).

A las columnas que siempre se excluyen del entrenamiento (`clase_ternaria`, `clase01` y `azar`), se les agregan dinámicamente `numero_de_cliente` y/o `foto_mes` según la variante recibida por parámetro:

```r
cols_to_exclude <- c("clase_ternaria", "clase01", "azar")

if (PARAM$variant$code %in% c(Variant$WithoutId$code, Variant$WithoutBoth$code)) {
  cols_to_exclude <- c(cols_to_exclude, "numero_de_cliente")
}

if (PARAM$variant$code %in% c(Variant$WithoutMonthSnapshot$code, Variant$WithoutBoth$code)) {
  cols_to_exclude <- c(cols_to_exclude, "foto_mes")
}

# Columnas finales con las que se entrena el modelo
campos_buenos <- copy(setdiff(colnames(dataset), cols_to_exclude))
```

---

## Evolución de las Corridas: Versión Inicial vs. Versión 3 (`v3`)

En el directorio van a encontrar dos juegos de archivos para el orquestador, el worker y el análisis: los que **no tienen sufijo de versión** y los que terminan en **`_v3`**.

### 1. Primera corrida (archivos sin versión)
- **Archivos:** `z621_WorkFlow_01_junior_grupoB_orchestrator.ipynb`, `z621_WorkFlow_01_junior_grupoB_worker.ipynb` y `exp_16_analysis.ipynb`.
- **Grilla utilizada:**
  - `num_leaves`: `c(64, 128, 256, 384, 512)`
  - `min_data_in_leaf`: `c(64, 256, 512, 1024, 2048)`
  - `feature_fraction`: `c(0.5, 0.8)`
- **Qué pasó:** Al analizar los resultados de esta primera tanda, vimos que en varias corridas los hiperparámetros óptimos elegidos habían quedado **en el borde** de la grilla (por ejemplo, tocando el mínimo de `num_leaves = 64` o el máximo de `min_data_in_leaf = 2048`). Esto indicaba que el espacio de búsqueda estaba restringiendo al modelo.

### 2. Corrida definitiva — Versión 3 (`_v3`)
- **Archivos:** `z621_WorkFlow_01_junior_grupoB_orchestrator_v3.ipynb`, `z621_WorkFlow_01_junior_grupoB_worker_v3.ipynb` y `exp_16_analysis_v3.ipynb`.
- **Grilla ajustada:** Movimos y ampliamos el rango de los hiperparámetros para evitar que saturen en los extremos:
  - `num_leaves`: `c(16, 32, 64, 128, 256, 512)`
  - `min_data_in_leaf`: `c(256, 512, 1024, 2048, 4096, 8192)`
  - `feature_fraction`: `c(0.5)`
- **Resultado:** Al volver a correr todo el experimento con esta nueva configuración, verificamos que los hiperparámetros ganadores quedaron contenidos dentro del rango interior de la grilla. **Esta versión (`v3`) fue la que se utilizó para sacar las conclusiones finales del experimento.**

---

## Análisis de Resultados

Los notebooks `exp_16_analysis.ipynb` y `exp_16_analysis_v3.ipynb` concentran el post-procesamiento y evaluación estadística de las corridas en tres etapas:

1. **Lectura y consolidación:** Recorren las carpetas de salida de cada corrida, leen los archivos `PARAM.yml` y arman una tabla comparativa (semillas en filas y variantes en columnas) tomando como métrica la ganancia máxima suavizada (`ganancia_suavizada_max`).
2. **Test de Wilcoxon apareado:** Comparan cada variante candidata (`without_id`, `without_month_snapshot` y `without_both`) contra el `baseline` mediante un test de Wilcoxon apareado (`wilcox.test(..., paired = TRUE)`), reportando cuántas semillas gana cada variante, la media y mediana de las diferencias de ganancia y el *p-valor* para determinar si la diferencia es estadísticamente significativa.
3. **Control de bordes de hiperparámetros:** Extraen `mejores_hiperparametros` de cada `PARAM.yml` y chequean cuántas corridas terminaron eligiendo valores en los extremos (mínimo o máximo) de la grilla configurada.

---

## Resumen de Archivos del Directorio

| Archivo | Descripción |
| :--- | :--- |
| `z621_WorkFlow_01_junior_grupoB.ipynb` | Notebook original de referencia del Workflow Junior - Grupo B. |
| `orchestrator.ipynb` / `worker.ipynb` | Prototipos básicos de la arquitectura Orquestador-Worker con `papermill`. |
| `z621_WorkFlow_01_junior_grupoB_orchestrator.ipynb` | Orquestador de la primera corrida del Experimento 16. |
| `z621_WorkFlow_01_junior_grupoB_worker.ipynb` | Worker de la primera corrida (donde algunos hiperparámetros quedaron en el borde). |
| `exp_16_analysis.ipynb` | Análisis estadístico y chequeo de bordes de la primera corrida. |
| `z621_WorkFlow_01_junior_grupoB_orchestrator_v3.ipynb` | Orquestador de la corrida definitiva (`v3`). |
| `z621_WorkFlow_01_junior_grupoB_worker_v3.ipynb` | Worker con el rango de hiperparámetros corregido (`v3`). |
| `exp_16_analysis_v3.ipynb` | **Análisis definitivo (`v3`)** utilizado para las conclusiones del experimento. |
| `z610_*`, `z611_*`, `z620_*`, `z635_*` | Otros workflows base de la cátedra (Gerencial, Junior Grupo A y Senior). |

