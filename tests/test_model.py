import pandas as pd

from src.train import DATA_PATH, MODEL_PATH, entrenar_modelo


def test_dataset_tiene_columnas_requeridas():
    datos = pd.read_csv(DATA_PATH)
    assert list(datos.columns) == ["dia", "ventas"]
    assert len(datos) == 12


def test_entrenamiento_crea_modelo():
    modelo = entrenar_modelo()
    assert MODEL_PATH.exists()

    entrada = pd.DataFrame({"dia": [15]})
    prediccion = modelo.predict(entrada)[0]
    assert prediccion > 0
