from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"

app = FastAPI(
    title="API de Predicción de Ventas",
    description="Servicio de IA para estimar ventas a partir del número de día.",
    version="1.0.0",
)


def cargar_modelo():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "No se encontró models/modelo.pkl. Ejecute primero: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


modelo = cargar_modelo()


class Entrada(BaseModel):
    dia: int = Field(gt=0, description="Número de día para realizar la predicción")


@app.get("/")
def inicio():
    return {"mensaje": "Servidor IA activo", "estado": "activo"}


@app.get("/predict")
def predict_get(dia: int):
    if dia <= 0:
        raise HTTPException(status_code=400, detail="El día debe ser mayor que 0")
    entrada = pd.DataFrame({"dia": [dia]})
    resultado = modelo.predict(entrada)
    return {"dia": dia, "prediccion": float(resultado[0])}


@app.post("/predict")
def predict_post(data: Entrada):
    entrada = pd.DataFrame({"dia": [data.dia]})
    resultado = modelo.predict(entrada)
    return {"dia": data.dia, "prediccion": float(resultado[0])}
