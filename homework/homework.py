import os
import pandas as pd
import gzip
import json
import pickle

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def cargar_datos(ruta_entrenamiento: str, ruta_prueba: str):
    """Carga los archivos CSV comprimidos del conjunto de entrenamiento y prueba."""
    datos_entrenamiento = pd.read_csv(ruta_entrenamiento)
    datos_prueba = pd.read_csv(ruta_prueba)
    print("Datasets cargados correctamente")
    return datos_entrenamiento, datos_prueba


def limpiar_datos(datos_entrenamiento: pd.DataFrame, datos_prueba: pd.DataFrame):
    """Limpia los datos según las reglas del enunciado."""
    # Renombrar la columna objetivo
    datos_entrenamiento.rename(columns={"default payment next month": "default"}, inplace=True)
    datos_prueba.rename(columns={"default payment next month": "default"}, inplace=True)

    # Eliminar columna ID si existe
    for conjunto in [datos_entrenamiento, datos_prueba]:
        if "ID" in conjunto.columns:
            conjunto.drop(columns=["ID"], inplace=True)

    # Eliminar registros con valores no válidos
    datos_entrenamiento = datos_entrenamiento.loc[
        (datos_entrenamiento["MARRIAGE"] != 0) & (datos_entrenamiento["EDUCATION"] != 0)
    ]
    datos_prueba = datos_prueba.loc[
        (datos_prueba["MARRIAGE"] != 0) & (datos_prueba["EDUCATION"] != 0)
    ]

    # Agrupar EDUCATION > 4 como “others” (4)
    datos_entrenamiento["EDUCATION"] = datos_entrenamiento["EDUCATION"].apply(
        lambda valor: 4 if valor > 4 else valor
    )
    datos_prueba["EDUCATION"] = datos_prueba["EDUCATION"].apply(
        lambda valor: 4 if valor > 4 else valor
    )

    datos_entrenamiento.dropna(inplace=True)
    datos_prueba.dropna(inplace=True)

    # Dividir en X e y
    caracteristicas_entrenamiento = datos_entrenamiento.drop(columns="default")
    etiquetas_entrenamiento = datos_entrenamiento["default"]
    caracteristicas_prueba = datos_prueba.drop(columns="default")
    etiquetas_prueba = datos_prueba["default"]

    print("Datos limpiados correctamente")
    return caracteristicas_entrenamiento, etiquetas_entrenamiento, caracteristicas_prueba, etiquetas_prueba


def crear_pipeline(caracteristicas_entrenamiento):
    """Crea un pipeline con OneHotEncoder, PCA, StandardScaler, SelectKBest y SVC."""

    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = list(set(caracteristicas_entrenamiento.columns).difference(columnas_categoricas))

    transformador_columnas = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", StandardScaler(with_mean=True, with_std=True), columnas_numericas),
        ],
        remainder="passthrough",
    )

    flujo_procesamiento = Pipeline(
        steps=[
            ("transformador", transformador_columnas),
            ("pca", PCA()),
            ("seleccion_k", SelectKBest(score_func=f_classif)),
            ("clasificador", SVC(kernel="rbf", random_state=42, max_iter=-1)),
        ]
    )

    print("Pipeline creado correctamente")
    return flujo_procesamiento


def ejecutar_grid_search(flujo_procesamiento, caracteristicas_entrenamiento, etiquetas_entrenamiento):
    """Ejecuta GridSearchCV con validación cruzada estratificada (10 folds)."""

    parametros = {
        "pca__n_components": [20, caracteristicas_entrenamiento.shape[1] - 2],
        "seleccion_k__k": [12],
        "clasificador__C": [0.8],
        "clasificador__kernel": ["rbf"],
        "clasificador__gamma": [0.099],
    }

    busqueda = GridSearchCV(
        estimator=flujo_procesamiento,
        param_grid=parametros,
        cv=10,
        scoring="balanced_accuracy",
    )

    busqueda.fit(caracteristicas_entrenamiento, etiquetas_entrenamiento)
    return busqueda


def guardar_modelo(modelo_entrenado):
    """Guarda el modelo entrenado comprimido como .pkl.gz"""
    os.makedirs("files/models", exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo_entrenado, archivo)
    print("Modelo guardado correctamente")


def calcular_metricas(modelo_entrenado, caracteristicas_entrenamiento, etiquetas_entrenamiento, caracteristicas_prueba, etiquetas_prueba):
    """Calcula métricas y matrices de confusión para train y test."""
    lista_metricas = []

    for x_datos, y_datos, tipo_conjunto in [
        (caracteristicas_entrenamiento, etiquetas_entrenamiento, "train"),
        (caracteristicas_prueba, etiquetas_prueba, "test"),
    ]:
        predicciones = modelo_entrenado.predict(x_datos)

        precision_val = precision_score(y_datos, predicciones, average="binary")
        balanced_acc_val = balanced_accuracy_score(y_datos, predicciones)
        recall_val = recall_score(y_datos, predicciones, average="binary")
        f1_val = f1_score(y_datos, predicciones, average="binary")

        lista_metricas.append(
            {
                "type": "metrics",
                "dataset": tipo_conjunto,
                "precision": precision_val,
                "balanced_accuracy": balanced_acc_val,
                "recall": recall_val,
                "f1_score": f1_val,
            }
        )

    for x_datos, y_datos, tipo_conjunto in [
        (caracteristicas_entrenamiento, etiquetas_entrenamiento, "train"),
        (caracteristicas_prueba, etiquetas_prueba, "test"),
    ]:
        predicciones = modelo_entrenado.predict(x_datos)
        matriz_conf = confusion_matrix(y_datos, predicciones)

        lista_metricas.append(
            {
                "type": "cm_matrix",
                "dataset": tipo_conjunto,
                "true_0": {"predicted_0": int(matriz_conf[0, 0]), "predicted_1": int(matriz_conf[0, 1])},
                "true_1": {"predicted_0": int(matriz_conf[1, 0]), "predicted_1": int(matriz_conf[1, 1])},
            }
        )

    return lista_metricas


def guardar_metricas(lista_metricas):
    """Guarda las métricas y matrices de confusión en un archivo JSON."""
    ruta_salida = "files/output"
    os.makedirs(ruta_salida, exist_ok=True)

    with open(os.path.join(ruta_salida, "metrics.json"), "w") as archivo_json:
        for metrica in lista_metricas:
            archivo_json.write(json.dumps(metrica, ensure_ascii=False))
            archivo_json.write("\n")


def main():
    datos_entrenamiento, datos_prueba = cargar_datos(
        "files/input/train_data.csv.zip", "files/input/test_data.csv.zip"
    )

    x_entrenamiento, y_entrenamiento, x_prueba, y_prueba = limpiar_datos(
        datos_entrenamiento, datos_prueba
    )

    flujo_procesamiento = crear_pipeline(x_entrenamiento)
    modelo_final = ejecutar_grid_search(flujo_procesamiento, x_entrenamiento, y_entrenamiento)
    guardar_modelo(modelo_final)

    resultados = calcular_metricas(modelo_final, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)
    guardar_metricas(resultados)


if __name__ == "__main__":
    main()
