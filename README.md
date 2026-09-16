# proyecto-mlops-ventas1

## Sistema Inteligente de Predicción de Ventas

Proyecto académico de MLOps que implementa el flujo solicitado por el instructor: datos, entrenamiento de Machine Learning, pruebas automáticas, GitHub Actions, API REST con FastAPI y contenerización con Docker.

## Arquitectura

```text
Datos CSV
   ↓
Entrenamiento (scikit-learn)
   ↓
Modelo modelo.pkl
   ↓
FastAPI
   ↓
Endpoint /predict
   ↓
Docker
   ↓
GitHub Actions CI
```

## Estructura

```text
proyecto-mlops-ventas1/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── app/
│   ├── __init__.py
│   └── api.py
├── data/
│   └── ventas.csv
├── models/
│   └── modelo.pkl        # se genera automáticamente
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## 1. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Entrenar el modelo

```bash
python src/train.py
```

Salida esperada:

```text
Modelo guardado en: .../models/modelo.pkl
```

Con los datos entregados por el instructor, la regresión lineal obtiene aproximadamente:

- Día 12: 312.12
- Día 15: 370.54
- Día 20: 467.92

## 3. Probar la predicción local

```bash
python src/predict.py
```

## 4. Ejecutar pruebas

```bash
pytest -q
```

## 5. Ejecutar la API FastAPI

Primero debe existir el modelo:

```bash
python src/train.py
uvicorn app.api:app --reload
```

Abrir en el navegador:

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

### GET

```text
http://localhost:8000/predict?dia=15
```

### POST

En Swagger, usar `POST /predict` con:

```json
{
  "dia": 20
}
```

## 6. Docker

Construir la imagen:

```bash
docker build -t api-ventas .
```

Ejecutar el contenedor:

```bash
docker run --rm -p 8000:8000 api-ventas
```

Abrir:

```text
http://localhost:8000/docs
```

## 7. Pipeline CI

Cada `git push` a `main` ejecuta automáticamente:

1. Descarga del repositorio.
2. Instalación de Python 3.11.
3. Instalación de dependencias.
4. Ejecución de pruebas con pytest.
5. Entrenamiento del modelo.
6. Prueba de predicción.
7. Construcción de la imagen Docker.
8. Publicación de `modelo.pkl` como artefacto de GitHub Actions.

## 8. Estrategias de despliegue revisadas

- **Blue-Green:** mantiene dos entornos; permite cambiar a la nueva versión con mínima interrupción.
- **Canary:** libera el modelo nuevo a un porcentaje reducido de usuarios y amplía gradualmente si funciona correctamente.
- **Shadow:** envía tráfico al modelo nuevo para comparar su comportamiento sin mostrar su respuesta al usuario final.

## Evidencias recomendadas para la entrega

Tomar capturas de:

1. Estructura del repositorio en GitHub.
2. GitHub Actions con `Pipeline ML` en verde.
3. Terminal con `python src/train.py` y creación de `modelo.pkl`.
4. Swagger en `http://localhost:8000/docs`.
5. Ejecución de `GET /predict?dia=15`.
6. Ejecución del `POST /predict` con día 20.
7. `docker build -t api-ventas .` finalizado correctamente.
8. `docker run -p 8000:8000 api-ventas` ejecutándose.
