import pandas as pd

from src.train import (
    DATA_PATH,
    MODEL_PATH,
    NEW_MODEL_PATH,
    entrenar_modelo,
)


def test_dataset_tiene_columnas_requeridas():
    datos = pd.read_csv(DATA_PATH)

    assert list(datos.columns) == ["dia", "ventas"]

    # Ahora el dataset tiene 20 registros
    assert len(datos) == 20


def test_entrenamiento_crea_modelos_blue_green():
    modelo = entrenar_modelo()

    # Verificar que exista el modelo BLUE
    assert MODEL_PATH.exists()

    # Verificar que exista el modelo GREEN
    assert NEW_MODEL_PATH.exists()

    # Probar que el modelo pueda generar una predicción
    entrada = pd.DataFrame({"dia": [21]})

    prediccion = modelo.predict(entrada)[0]

    assert prediccion > 0