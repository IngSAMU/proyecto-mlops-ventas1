
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import time
from datetime import datetime
import os

from app.rag import (
    generar_respuesta_natural,
    obtener_maxima_venta,
    obtener_promedio_ventas
)

os.makedirs("logs", exist_ok=True)
# ===================================================
# CONFIGURACIÓN DE LOGS
# ===================================================

pred_logger = logging.getLogger("predicciones")
pred_logger.setLevel(logging.INFO)

pred_handler = logging.FileHandler(
    "logs/predicciones.log"
)

pred_formatter = logging.Formatter(
    "%(message)s"
)

pred_handler.setFormatter(pred_formatter)

if not pred_logger.handlers:
    pred_logger.addHandler(pred_handler)


# ---------------------------------------------------

deploy_logger = logging.getLogger("deployment")
deploy_logger.setLevel(logging.INFO)

deploy_handler = logging.FileHandler(
    "logs/deployment.log"
)

deploy_formatter = logging.Formatter(
    "%(message)s"
)

deploy_handler.setFormatter(deploy_formatter)

if not deploy_logger.handlers:
    deploy_logger.addHandler(deploy_handler)
app = FastAPI()


# ==========================
# CARGA DE MODELOS
# ==========================

modelo_blue = joblib.load("models/modelo.pkl")
modelo_green = joblib.load("models/modelo_nuevo.pkl")

# Modelo que está atendiendo actualmente
ACTIVE_MODEL = "BLUE"

# Contador básico de solicitudes
TOTAL_REQUESTS = 0


# ==========================
# CLASE DE ENTRADA
# ==========================

class Entrada(BaseModel):
    dia: int


# ==========================
# ESTADO DEL SERVICIO
# ==========================

@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "modelo_activo": ACTIVE_MODEL
    }
# ===================================================
# RAG - CONSULTA DE VENTAS
# ===================================================

@app.get("/consulta/{dia}")
def consulta(dia: int):
    respuesta = generar_respuesta_natural(dia)

    return {
        "respuesta": respuesta
    }


# ===================================================
# RAG - MAYOR VENTA
# ===================================================

@app.get("/max-ventas")
def max_ventas():
    resultado = obtener_maxima_venta()

    return {
        "mensaje": f"El día con mayores ventas fue {resultado['dia']}",
        "ventas": resultado["ventas"]
    }


# ===================================================
# RAG - PROMEDIO DE VENTAS
# ===================================================

@app.get("/promedio-ventas")
def promedio_ventas():
    promedio = obtener_promedio_ventas()

    return {
        "promedio": promedio
    }

# ==========================
# PREDICCION
# ==========================

@app.post("/predict")
def predict(datos: Entrada):

    global TOTAL_REQUESTS

    TOTAL_REQUESTS += 1

    # Medir inicio de la predicción
    tiempo_inicio = time.time()

    # Seleccionar modelo activo
    if ACTIVE_MODEL == "BLUE":
        resultado = modelo_blue.predict([[datos.dia]])
    else:
        resultado = modelo_green.predict([[datos.dia]])

    # Calcular latencia
    tiempo_fin = time.time()
    latencia = tiempo_fin - tiempo_inicio

    prediccion = float(resultado[0])

    # Registrar predicción en el log
    pred_logger.info(
        f"{datetime.now()} | "
        f"Modelo={ACTIVE_MODEL} | "
        f"Dia={datos.dia} | "
        f"Prediccion={prediccion:.2f} | "
        f"Latencia={latencia:.6f}"
    )

    return {
        "modelo_utilizado": ACTIVE_MODEL,
        "dia": datos.dia,
        "prediccion": prediccion,
        "latencia_segundos": round(latencia, 6)
    }


# ==========================
# SWITCH BLUE-GREEN
# ==========================

@app.put("/switch/{color}")
def switch_model(color: str):

    global ACTIVE_MODEL

    color = color.upper()

    if color not in ["BLUE", "GREEN"]:
        return {
            "error": "Debe elegir BLUE o GREEN"
        }

    ACTIVE_MODEL = color

    return {
        "mensaje": f"Producción ahora utiliza {ACTIVE_MODEL}"
    }


# ==========================
# RECARGAR MODELOS
# ==========================

@app.put("/reload-model")
def reload_model():

    global modelo_blue
    global modelo_green

    modelo_blue = joblib.load(
        "models/modelo.pkl"
    )

    modelo_green = joblib.load(
        "models/modelo_nuevo.pkl"
    )

    return {
        "mensaje": "Modelos recargados correctamente"
    }


# ==========================
# RESET
# ==========================

# ===================================================
# MÉTRICAS BÁSICAS
# ===================================================

@app.get("/metrics")
def metrics():

    return {
        "estado": "ok",
        "modelo_activo": ACTIVE_MODEL,
        "total_predicciones": TOTAL_REQUESTS
    }
# ===================================================
# RESET DEL SERVICIO
# ===================================================

@app.delete("/reset")
def reset_service():

    return {
        "mensaje":
        "Recursos reiniciados correctamente"
    }