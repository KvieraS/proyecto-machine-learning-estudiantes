import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# RUTAS DEL PROYECTO
# -----------------------------------------------------------------------------
# BASE_DIR apunta a la raíz del proyecto para que el script funcione
# correctamente aunque se ejecute desde distintas carpetas.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset_estudiantes.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def cargar_datos(path: Path) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Parámetros
    ----------
    path : Path
        Ruta del archivo CSV que contiene los datos originales.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la información del dataset.
    """
    return pd.read_csv(path)


def resumen_inicial(df: pd.DataFrame) -> None:
    """
    Muestra un resumen inicial del dataset.

    Este bloque sirve como primera inspección del conjunto de datos para:
    - revisar tipos de datos,
    - detectar valores nulos,
    - comprobar duplicados,
    - observar las primeras filas,
    - obtener estadísticas descriptivas básicas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original cargado desde el CSV.
    """
    print("\n--- INFORMACIÓN GENERAL ---")
    print(df.info())

    print("\n--- PRIMERAS FILAS ---")
    print(df.head())

    print("\n--- VALORES NULOS ---")
    print(df.isna().sum())

    print("\n--- DUPLICADOS ---")
    print(df.duplicated().sum())

    print("\n--- DESCRIPTIVO NUMÉRICO ---")
    print(df.describe().round(2))


def analisis_exploratorio(df: pd.DataFrame) -> None:
    """
    Genera y guarda gráficos básicos del análisis exploratorio de datos (EDA).

    Los gráficos producidos permiten estudiar:
    - la distribución de la variable objetivo continua,
    - el equilibrio de clases en la variable binaria,
    - la correlación entre variables numéricas,
    - relaciones entre variables predictoras y objetivos.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame completo con las variables predictoras y objetivo.
    """
    # 1) Histograma de la variable objetivo continua.
    # Ayuda a ver si la nota final está centrada, sesgada o muy dispersa.
    plt.figure(figsize=(8, 5))
    plt.hist(df["nota_final"].dropna(), bins=25)
    plt.title("Distribución de la nota final")
    plt.xlabel("Nota final")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "01_distribucion_nota_final.png")
    plt.close()

    # 2) Distribución de la variable binaria aprobado.
    # Permite comprobar si hay desbalanceo entre clases.
    plt.figure(figsize=(6, 4))
    conteo_aprobado = df["aprobado"].value_counts().sort_index()
    plt.bar(conteo_aprobado.index.astype(str), conteo_aprobado.values)
    plt.title("Distribución de la variable aprobado")
    plt.xlabel("Aprobado (0 = No, 1 = Sí)")
    plt.ylabel("Número de estudiantes")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_distribucion_aprobado.png")
    plt.close()

    # 3) Mapa de calor de correlaciones numéricas.
    # Es útil para detectar relaciones lineales entre variables.
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include=np.number).corr()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Se anotan los coeficientes dentro del mapa para facilitar la lectura.
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Mapa de calor de correlaciones")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "03_heatmap_correlaciones.png")
    plt.close()

    # 4) Dispersión entre horas de estudio y nota final.
    # Sirve para ver si existe una tendencia positiva entre estudio y rendimiento.
    plt.figure(figsize=(8, 5))
    plt.scatter(df["horas_estudio_semanal"], df["nota_final"], alpha=0.7)
    plt.title("Horas de estudio semanal vs nota final")
    plt.xlabel("Horas de estudio semanal")
    plt.ylabel("Nota final")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "04_scatter_horas_nota.png")
    plt.close()

    # 5) Boxplot de asistencia según aprobado.
    # Ayuda a comparar si la asistencia cambia entre aprobados y suspensos.
    plt.figure(figsize=(8, 5))
    grupos_asistencia = [
        grupo["tasa_asistencia"].dropna().values for _, grupo in df.groupby("aprobado")
    ]
    plt.boxplot(
        grupos_asistencia,
        tick_labels=[str(k) for k in sorted(df["aprobado"].dropna().unique())],
    )
    plt.title("Tasa de asistencia según aprobado")
    plt.xlabel("Aprobado")
    plt.ylabel("Tasa de asistencia")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "05_boxplot_asistencia_aprobado.png")
    plt.close()

    # 6) Boxplot de nota final según nivel de dificultad.
    # Permite comparar el rendimiento entre distintos perfiles de dificultad.
    plt.figure(figsize=(8, 5))
    orden = sorted(df["nivel_dificultad"].dropna().unique())
    grupos_nota = [
        df.loc[df["nivel_dificultad"] == nivel, "nota_final"].dropna().values
        for nivel in orden
    ]
    plt.boxplot(grupos_nota, tick_labels=orden)
    plt.title("Nota final según nivel de dificultad")
    plt.xlabel("Nivel de dificultad")
    plt.ylabel("Nota final")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "06_boxplot_dificultad_nota.png")
    plt.close()


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye el preprocesador para variables numéricas y categóricas.

    Estrategia aplicada:
    - variables numéricas:
        * imputación por mediana para rellenar nulos,
        * estandarización para homogeneizar escalas.
    - variables categóricas:
        * imputación por valor más frecuente,
        * codificación One Hot Encoding para convertir categorías en columnas binarias.

    Parámetros
    ----------
    X : pd.DataFrame
        Conjunto de variables predictoras.

    Retorna
    -------
    ColumnTransformer
        Transformador preparado para integrarse dentro de un Pipeline de sklearn.
    """
    # Se identifican automáticamente las columnas numéricas y categóricas.
    columnas_numericas = X.select_dtypes(exclude="object").columns.tolist()
    columnas_categoricas = X.select_dtypes(include="object").columns.tolist()

    # Pipeline para variables numéricas.
    # La mediana es robusta frente a valores extremos y el escalado ayuda
    # especialmente en modelos como la regresión logística.
    pipeline_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline para variables categóricas.
    # OneHotEncoder evita imponer un orden artificial entre categorías.
    pipeline_categorico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer aplica el tratamiento correcto a cada grupo de columnas.
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, columnas_numericas),
            ("cat", pipeline_categorico, columnas_categoricas),
        ]
    )

    return preprocesador


def modelo_regresion(df: pd.DataFrame) -> dict:
    """
    Entrena y evalúa un modelo de regresión lineal.

    Objetivo
    --------
    Predecir la variable continua `nota_final`.

    Decisión importante
    -------------------
    Se elimina `aprobado` de las variables predictoras para evitar fuga de información,
    ya que `aprobado` está directamente derivada de `nota_final`.

    Evaluación
    ----------
    - conjunto de test: MAE, RMSE y R²
    - validación cruzada: medias de MAE, RMSE y R²

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original del proyecto.

    Retorna
    -------
    dict
        Diccionario con métricas de test y validación cruzada.
    """
    # X contiene solo predictores válidos para la tarea de regresión.
    X = df.drop(columns=["nota_final", "aprobado"])
    y = df["nota_final"]

    preprocesador = construir_preprocesador(X)

    # Pipeline completo: preprocesamiento + modelo.
    # De esta forma evitamos fugas de datos y mantenemos el flujo limpio.
    modelo = Pipeline(
        steps=[
            ("prep", preprocesador),
            ("model", LinearRegression()),
        ]
    )

    # División train/test para evaluar el rendimiento en datos no vistos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    # Métricas principales de regresión.
    resultados_test = {
        "mae": mean_absolute_error(y_test, predicciones),
        "rmse": np.sqrt(mean_squared_error(y_test, predicciones)),
        "r2": r2_score(y_test, predicciones),
    }

    # Validación cruzada para estimar la estabilidad del modelo.
    resultados_cv = cross_validate(
        modelo,
        X,
        y,
        cv=5,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
    )

    # sklearn devuelve los errores en negativo en algunos scorings.
    resumen_cv = {
        "mae_cv_mean": (-resultados_cv["test_neg_mean_absolute_error"]).mean(),
        "rmse_cv_mean": (-resultados_cv["test_neg_root_mean_squared_error"]).mean(),
        "r2_cv_mean": resultados_cv["test_r2"].mean(),
    }

    return {
        "test": resultados_test,
        "cv": resumen_cv,
    }


def modelo_clasificacion(df: pd.DataFrame) -> dict:
    """
    Entrena y evalúa un modelo de regresión logística.

    Objetivo
    --------
    Predecir la variable binaria `aprobado`.

    Decisión importante
    -------------------
    Se elimina `nota_final` de las variables predictoras para evitar fuga de información,
    ya que `aprobado` depende directamente de si la nota final es mayor o igual que 60.

    Evaluación
    ----------
    - conjunto de test: accuracy, balanced accuracy, ROC AUC,
      matriz de confusión y classification report
    - validación cruzada: accuracy, precision, recall, f1 y ROC AUC

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original del proyecto.

    Retorna
    -------
    dict
        Diccionario con métricas de test y validación cruzada.
    """
    # X contiene únicamente las variables permitidas para clasificación.
    X = df.drop(columns=["aprobado", "nota_final"])
    y = df["aprobado"]

    preprocesador = construir_preprocesador(X)

    # Se usa class_weight="balanced" para compensar el desbalanceo de clases.
    modelo = Pipeline(
        steps=[
            ("prep", preprocesador),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    # La partición estratificada mantiene la proporción de aprobados/suspensos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    # Métricas clave para clasificación.
    resultados_test = {
        "accuracy": accuracy_score(y_test, predicciones),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicciones),
        "roc_auc": roc_auc_score(y_test, probabilidades),
        "confusion_matrix": confusion_matrix(y_test, predicciones).tolist(),
        "classification_report": classification_report(y_test, predicciones, digits=4),
    }

    # Validación cruzada del modelo de clasificación.
    resultados_cv = cross_validate(
        modelo,
        X,
        y,
        cv=5,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    resumen_cv = {
        "accuracy_cv_mean": resultados_cv["test_accuracy"].mean(),
        "precision_cv_mean": resultados_cv["test_precision"].mean(),
        "recall_cv_mean": resultados_cv["test_recall"].mean(),
        "f1_cv_mean": resultados_cv["test_f1"].mean(),
        "roc_auc_cv_mean": resultados_cv["test_roc_auc"].mean(),
    }

    return {
        "test": resultados_test,
        "cv": resumen_cv,
    }


def guardar_resumen(df: pd.DataFrame, resultados_reg: dict, resultados_clf: dict) -> None:
    """
    Guarda un resumen del proyecto en formato JSON.

    El archivo incluye:
    - dimensiones del dataset,
    - valores nulos por columna,
    - filas duplicadas,
    - distribución de la clase objetivo de clasificación,
    - métricas finales de regresión y clasificación.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del proyecto.
    resultados_reg : dict
        Resultados del modelo de regresión.
    resultados_clf : dict
        Resultados del modelo de clasificación.
    """
    resumen = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "aprobado_distribution": df["aprobado"].value_counts(normalize=True).round(4).to_dict(),
        "regresion": resultados_reg,
        "clasificacion": resultados_clf,
    }

    with open(OUTPUTS_DIR / "resumen_modelos.json", "w", encoding="utf-8") as file:
        json.dump(resumen, file, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Ejecuta el flujo completo del proyecto.

    Orden de ejecución:
    1. carga de datos,
    2. resumen inicial,
    3. análisis exploratorio,
    4. entrenamiento del modelo de regresión,
    5. entrenamiento del modelo de clasificación,
    6. impresión de métricas,
    7. guardado del resumen final.
    """
    df = cargar_datos(DATA_PATH)

    resumen_inicial(df)
    analisis_exploratorio(df)

    resultados_reg = modelo_regresion(df)
    resultados_clf = modelo_clasificacion(df)

    print("\n--- RESULTADOS REGRESIÓN ---")
    print(resultados_reg)

    print("\n--- RESULTADOS CLASIFICACIÓN ---")
    print(resultados_clf)

    guardar_resumen(df, resultados_reg, resultados_clf)


if __name__ == "__main__":
    main()
