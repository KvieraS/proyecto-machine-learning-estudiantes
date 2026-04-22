# Proyecto Machine Learning - Rendimiento académico de estudiantes

## Objetivo

Desarrollar un proyecto completo de **Machine Learning** a partir del dataset `dataset_estudiantes.csv`, cubriendo:

- análisis exploratorio de datos (EDA)
- preprocesamiento
- entrenamiento y validación de un modelo de **regresión lineal**
- entrenamiento y validación de un modelo de **regresión logística**

La variable objetivo de regresión es **`nota_final`** y la variable objetivo de clasificación es **`aprobado`**.

---

## Dataset

El conjunto de datos contiene información sobre hábitos de estudio, asistencia, sueño y características personales de estudiantes.

### Variables predictoras

- `horas_estudio_semanal`
- `nota_anterior`
- `tasa_asistencia`
- `horas_sueno`
- `edad`
- `nivel_dificultad`
- `tiene_tutor`
- `horario_estudio_preferido`
- `estilo_aprendizaje`

### Variables objetivo

- **Regresión:** `nota_final`
- **Clasificación:** `aprobado`

---

## Estructura del proyecto

```text
Proyecto-MachineLearning/
│
├── data/
│   └── raw/
│       └── dataset_estudiantes.csv
│
├── outputs/
│   ├── 01_distribucion_nota_final.png
│   ├── 02_distribucion_aprobado.png
│   ├── 03_heatmap_correlaciones.png
│   ├── 04_scatter_horas_nota.png
│   ├── 05_boxplot_asistencia_aprobado.png
│   ├── 06_boxplot_dificultad_nota.png
│   ├── metrics_resumen.json
│   └── resumen_modelos.json
│
├── src/
│   └── proyecto_ml_estudiantes.py
│
└── README.md
```

---

## Carga y revisión inicial

Se cargó el archivo CSV con `pandas` y se revisaron los siguientes aspectos:

- dimensiones del dataset
- tipos de datos
- valores nulos
- duplicados
- estadísticas descriptivas básicas

### Tamaño del dataset

- **Filas:** 1000
- **Columnas:** 11

### Valores nulos detectados

- `horas_estudio_semanal`: 0
- `nota_anterior`: 0
- `tasa_asistencia`: 0
- `horas_sueno`: 150
- `edad`: 0
- `nivel_dificultad`: 0
- `tiene_tutor`: 0
- `horario_estudio_preferido`: 100
- `estilo_aprendizaje`: 50
- `nota_final`: 0
- `aprobado`: 0

### Duplicados

- **Filas duplicadas:** 0

---

## Limpieza y preprocesamiento

### 1. Tratamiento de nulos

Se detectaron valores faltantes en:

- `horas_sueno`: **150**
- `horario_estudio_preferido`: **100**
- `estilo_aprendizaje`: **50**

Para resolverlo:

- en variables numéricas se imputó la **mediana**
- en variables categóricas se imputó la **moda**

### 2. Codificación de variables categóricas

Las columnas categóricas se transformaron con **OneHotEncoder** para que pudieran ser utilizadas por los modelos.

### 3. Escalado

Las variables numéricas se escalaron con **StandardScaler**.

### 4. Fuga de información

Había un punto crítico en este proyecto:

- para clasificación, **no se puede usar `nota_final` como predictor**, porque `aprobado` se define directamente a partir de esa variable
- para regresión, **no se debe usar `aprobado` como predictor**, porque deriva de la nota final

Esto se eliminó en ambos modelos para evitar **data leakage**.

---

## Análisis exploratorio

Durante el EDA se revisaron distribuciones, correlaciones y relaciones entre variables.

### Hallazgos principales

- `horas_estudio_semanal` tiene una relación positiva clara con `nota_final`
- `nota_anterior` también muestra una asociación importante con la nota final
- `tasa_asistencia` influye positivamente en el rendimiento
- `edad` apenas muestra relación lineal con la nota final
- la variable `aprobado` está desbalanceada:
  - **Aprobados:** 89.8%
  - **No aprobados:** 10.2%

### Correlaciones numéricas con `nota_final`

```text
nota_final               1.000
aprobado                 0.579
horas_estudio_semanal    0.514
nota_anterior            0.470
tasa_asistencia          0.317
horas_sueno              0.075
edad                    -0.012
```

---

## Modelo de regresión lineal

### Objetivo

Predecir la variable continua **`nota_final`**.

### Configuración

- train/test split: 80/20
- preprocesado con pipeline
- modelo: `LinearRegression`
- validación adicional con **cross-validation de 5 folds**

### Resultados en test

- **MAE:** 5.8161
- **RMSE:** 7.2281
- **R²:** 0.3613

### Resultados medios en validación cruzada

- **MAE CV:** 6.0462
- **RMSE CV:** 7.5871
- **R² CV:** 0.3647

### Interpretación

El modelo explica alrededor del **36% de la variabilidad** de la nota final. No es un ajuste excelente, pero sí suficiente para mostrar una relación real entre varias variables académicas y el rendimiento del estudiante.

Las variables con mayor peso en el modelo fueron principalmente:

- horas de estudio semanal
- nota anterior
- tasa de asistencia
- nivel de dificultad

---

## Modelo de regresión logística

### Objetivo

Clasificar si un estudiante **aprueba o no**.

### Configuración

- train/test split: 80/20 con estratificación
- preprocesado con pipeline
- modelo: `LogisticRegression`
- `class_weight="balanced"` por desbalance de clases
- validación adicional con **cross-validation de 5 folds**

### Resultados en test

- **Accuracy:** 0.7000
- **Balanced Accuracy:** 0.6778
- **ROC AUC:** 0.8122
- **Matriz de confusión:** `[[13, 7], [53, 127]]`

### Resultados medios en validación cruzada

- **Accuracy CV:** 0.7290
- **Precision CV:** 0.9510
- **Recall CV:** 0.7361
- **F1 CV:** 0.8295
- **ROC AUC CV:** 0.7864

### Interpretación

Aunque el dataset presenta una clase mayoritaria muy dominante, el uso de pesos balanceados permite que el modelo no se limite a predecir siempre la clase mayoritaria.

Esto es importante porque:

- una accuracy alta por sí sola puede ser engañosa
- en datasets desbalanceados conviene revisar también **balanced accuracy**, **recall**, **F1** y **ROC AUC**

---

## Conclusiones

1. El dataset tiene una estructura adecuada para aplicar tanto regresión como clasificación.
2. El preprocesamiento es una parte clave del proyecto por la presencia de nulos y variables categóricas.
3. Existe riesgo real de **fuga de información**, por lo que fue necesario excluir variables derivadas del objetivo.
4. En regresión, las variables más influyentes fueron las relacionadas con el esfuerzo y el historial académico.
5. En clasificación, el problema está condicionado por el desbalance de clases, por lo que no conviene evaluar el modelo solo con accuracy.
6. El proyecto cumple con los requisitos mínimos del enunciado: EDA, preprocesamiento, regresión y clasificación.

---

## Tecnologías utilizadas

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Cómo ejecutar el proyecto

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python src/proyecto_ml_estudiantes.py
```

El script generará:

- gráficos del análisis exploratorio en la carpeta `outputs`
- un resumen de métricas en formato JSON

---

## Posibles mejoras

- probar otros modelos de regresión y clasificación
- ajustar hiperparámetros
- aplicar selección de variables
- tratar el desbalance con otras técnicas
- comparar resultados con árboles, random forest o gradient boosting

---

## Autor

Proyecto desarrollado por **Kevin Jesús Santoveña Viera** como práctica de Machine Learning.

