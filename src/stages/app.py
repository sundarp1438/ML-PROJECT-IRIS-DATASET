import uvicorn
import pandas as pd
import joblib
import yaml
import json
import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

# Load Configuration
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

# Paths from Config
model_path = config.get("train", {}).get("model_path", "models/model.joblib")
metrics_path = config.get("evaluate", {}).get("metrics_path", "reports/metrics.json")

# Load Trained Model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file not found: {model_path}")
model = joblib.load(model_path)

# Load Evaluation Metrics
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
else:
    metrics = {"message": "Metrics not found. Train the model first."}

# Initialize FastAPI
app = FastAPI(title="Iris ML API", description="API for Iris Classification using FastAPI and MLflow")

# Request Body Schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# MLflow Experiment
mlflow.set_experiment("Iris API Requests")

@app.get("/")
def home():
    """Root endpoint"""
    return {"message": "Welcome to the Iris Classification API!"}

@app.get("/metrics")
def get_metrics():
    """Returns model evaluation metrics"""
    return metrics

@app.post("/predict")
def predict(iris: IrisInput):
    """Predict the class of Iris flower based on input features"""
    input_data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    prediction = model.predict(input_data)
    predicted_class = int(prediction[0])

    # MLflow Logging
    with mlflow.start_run():
        mlflow.log_params(iris.dict())
        mlflow.log_metric("prediction", predicted_class)
    
    return {
        "predicted_class": predicted_class,
        "message": f"Model predicts class {predicted_class} for the given input."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
