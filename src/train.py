from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ventas.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "modelo.pkl"


def entrenar_modelo() -> LinearRegression:
    """Entrena un modelo de regresión lineal con las ventas históricas."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    datos = pd.read_csv(DATA_PATH)
    X = datos[["dia"]]
    y = datos["ventas"]

    modelo = LinearRegression()
    modelo.fit(X, y)
    joblib.dump(modelo, MODEL_PATH)
    return modelo


if __name__ == "__main__":
    entrenar_modelo()
    print(f"Modelo guardado en: {MODEL_PATH}")
