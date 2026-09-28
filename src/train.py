from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


# ==========================
# RUTAS DEL PROYECTO
# ==========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "ventas.csv"

MODEL_DIR = BASE_DIR / "models"

# Modelo actual de producción - BLUE
MODEL_PATH = MODEL_DIR / "modelo.pkl"

# Nueva versión candidata - GREEN
NEW_MODEL_PATH = MODEL_DIR / "modelo_nuevo.pkl"


# ==========================
# ENTRENAMIENTO
# ==========================

def entrenar_modelo() -> LinearRegression:
    """
    Entrena el modelo de regresión lineal y genera
    las versiones BLUE y GREEN.
    """

    # 1. Asegurar que exista la carpeta models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Cargar datos
    datos = pd.read_csv(DATA_PATH)

    X = datos[["dia"]]
    y = datos["ventas"]

    # 3. Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X, y)

    # 4. Guardar versión BLUE
    joblib.dump(modelo, MODEL_PATH)

    # 5. Guardar versión GREEN
    joblib.dump(modelo, NEW_MODEL_PATH)

    return modelo


# ==========================
# EJECUCIÓN DIRECTA
# ==========================

if __name__ == "__main__":

    entrenar_modelo()

    print(
        f"Modelo principal (BLUE) guardado en: {MODEL_PATH}"
    )

    print(
        f"Nuevo modelo (GREEN) guardado en: {NEW_MODEL_PATH}"
    )