from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"

modelo = joblib.load(MODEL_PATH)

dia = pd.DataFrame({"dia": [12]})
prediccion = modelo.predict(dia)

print(f"Predicción para el día 12: {float(prediccion[0]):.2f}")
