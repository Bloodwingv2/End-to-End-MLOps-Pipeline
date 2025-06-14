from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Disease Prediction ML API",
    description="Production-ready ML API for predicting Diabetes, Heart Disease, and Parkinson's Disease",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request validation
class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: int = Field(..., ge=0, le=200, description="Glucose level (mg/dL)")
    blood_pressure: int = Field(..., ge=0, le=150, description="Blood pressure (mm Hg)")
    skin_thickness: int = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    insulin: int = Field(..., ge=0, le=900, description="Insulin level (IU/mL)")
    bmi: float = Field(..., ge=0.0, le=50.0, description="Body Mass Index")
    dpf: float = Field(..., ge=0.0, le=2.5, description="Diabetes Pedigree Function")
    age: int = Field(..., ge=1, le=120, description="Age")

class HeartInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: Female, 1: Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure")
    chol: int = Field(..., ge=100, le=600, description="Cholesterol (mg/dL)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dL")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG")
    thalach: int = Field(..., ge=50, le=220, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=5.0, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia")

class ParkinsonsInput(BaseModel):
    fo: float = Field(..., ge=50.0, le=300.0, description="MDVP:Fo(Hz)")
    fhi: float = Field(..., ge=50.0, le=600.0, description="MDVP:Fhi(Hz)")
    flo: float = Field(..., ge=50.0, le=300.0, description="MDVP:Flo(Hz)")
    jitter: float = Field(..., ge=0.0, le=0.1, description="MDVP:Jitter(%)")
    jitter_abs: float = Field(..., ge=0.0, le=0.1, description="MDVP:Jitter(Abs)")
    rap: float = Field(..., ge=0.0, le=1.0, description="MDVP:RAP")
    ppq: float = Field(..., ge=0.0, le=1.0, description="MDVP:PPQ")
    ddp: float = Field(..., ge=0.0, le=1.0, description="Jitter:DDP")
    shimmer: float = Field(..., ge=0.0, le=0.5, description="MDVP:Shimmer")
    shimmer_db: float = Field(..., ge=0.0, le=0.5, description="MDVP:Shimmer(dB)")
    apq3: float = Field(..., ge=0.0, le=1.0, description="Shimmer:APQ3")
    apq5: float = Field(..., ge=0.0, le=1.0, description="Shimmer:APQ5")
    apq: float = Field(..., ge=0.0, le=1.0, description="MDVP:APQ")
    dda: float = Field(..., ge=0.0, le=1.0, description="Shimmer:DDA")
    nhr: float = Field(..., ge=0.0, le=1.0, description="NHR")
    hnr: float = Field(..., ge=0.0, le=40.0, description="HNR")
    rpde: float = Field(..., ge=0.0, le=1.0, description="RPDE")
    dfa: float = Field(..., ge=0.0, le=1.0, description="DFA")
    spread1: float = Field(..., ge=-10.0, le=0.0, description="spread1")
    spread2: float = Field(..., ge=0.0, le=1.0, description="spread2")
    d2: float = Field(..., ge=0.0, le=3.0, description="D2")
    ppe: float = Field(..., ge=0.0, le=1.0, description="PPE")

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    model_metrics: Dict[str, float]

# Global variables for models and scalers
models = {}
scalers = {}
metrics = {}

def load_models_and_scalers():
    """Load all models, scalers, and metrics at startup"""
    try:
        # Load models
        models['diabetes'] = joblib.load('Trained_Models/Diabetes_Models/Best_model_diabetes.pkl')
        models['heart'] = joblib.load('Trained_Models/Heart_Models/Best_model_Heart.pkl')
        models['parkinsons'] = joblib.load('Trained_Models/Parkinsons_Models/best_model_parkinsons.pkl')
        
        # Load scalers
        scalers['diabetes'] = joblib.load('Trained_Models/Scalers/diabetes_scaler.pkl')
        scalers['heart'] = joblib.load('Trained_Models/Scalers/heart_scaler.pkl')
        scalers['parkinsons'] = joblib.load('Trained_Models/Scalers/parkinsons_scaler.pkl')
        
        # Load metrics
        with open('Trained_Models/Diabetes_Models/Best_model_diabetes_metrics.json', 'r') as f:
            metrics['diabetes'] = json.load(f)
        with open('Trained_Models/Heart_Models/Best_model_heart_metrics.json', 'r') as f:
            metrics['heart'] = json.load(f)
        with open('Trained_Models/Parkinsons_Models/Best_model_parkinsons_metrics.json', 'r') as f:
            metrics['parkinsons'] = json.load(f)
            
        print("✅ All models, scalers, and metrics loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load models at startup
@app.on_event("startup")
async def startup_event():
    success = load_models_and_scalers()
    if not success:
        raise Exception("Failed to load models and scalers")

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Disease Prediction ML API", 
        "status": "healthy",
        "version": "1.0.0",
        "available_endpoints": ["/predict/diabetes", "/predict/heart", "/predict/parkinsons"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models) == 3}

# Get model metrics
@app.get("/metrics/{disease}")
async def get_metrics(disease: str):
    if disease not in metrics:
        raise HTTPException(status_code=404, detail=f"Metrics for {disease} not found")
    return metrics[disease]

# Prediction endpoints
@app.post("/predict/diabetes", response_model=PredictionResponse)
async def predict_diabetes(input_data: DiabetesInput):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([[
            input_data.pregnancies, input_data.glucose, input_data.blood_pressure,
            input_data.skin_thickness, input_data.insulin, input_data.bmi,
            input_data.dpf, input_data.age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])
        
        # Scale and predict
        scaled_data = scalers['diabetes'].transform(df)
        prediction = models['diabetes'].predict(scaled_data)[0]
        
        # Get prediction probability for confidence
        try:
            prob = models['diabetes'].predict_proba(scaled_data)[0]
            confidence = float(max(prob))
        except:
            confidence = 0.0
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Diabetic" if prediction == 1 else "Non-Diabetic",
            confidence=confidence,
            model_metrics=metrics['diabetes']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/heart", response_model=PredictionResponse)
async def predict_heart(input_data: HeartInput):
    try:
        df = pd.DataFrame([[
            input_data.age, input_data.sex, input_data.cp, input_data.trestbps,
            input_data.chol, input_data.fbs, input_data.restecg, input_data.thalach,
            input_data.exang, input_data.oldpeak, input_data.slope, input_data.ca, input_data.thal
        ]], columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])
        
        scaled_data = scalers['heart'].transform(df)
        prediction = models['heart'].predict(scaled_data)[0]
        
        try:
            prob = models['heart'].predict_proba(scaled_data)[0]
            confidence = float(max(prob))
        except:
            confidence = 0.0
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            confidence=confidence,
            model_metrics=metrics['heart']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/parkinsons", response_model=PredictionResponse)
async def predict_parkinsons(input_data: ParkinsonsInput):
    try:
        df = pd.DataFrame([[
            input_data.fo, input_data.fhi, input_data.flo, input_data.jitter,
            input_data.jitter_abs, input_data.rap, input_data.ppq, input_data.ddp,
            input_data.shimmer, input_data.shimmer_db, input_data.apq3, input_data.apq5,
            input_data.apq, input_data.dda, input_data.nhr, input_data.hnr,
            input_data.rpde, input_data.dfa, input_data.spread1, input_data.spread2,
            input_data.d2, input_data.ppe
        ]], columns=[
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ])
        
        scaled_data = scalers['parkinsons'].transform(df)
        prediction = models['parkinsons'].predict(scaled_data)[0]
        
        try:
            prob = models['parkinsons'].predict_proba(scaled_data)[0]
            confidence = float(max(prob))
        except:
            confidence = 0.0
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Parkinson's Detected" if prediction == 1 else "No Parkinson's",
            confidence=confidence,
            model_metrics=metrics['parkinsons']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint (bonus feature)
@app.post("/predict/batch/{disease}")
async def batch_predict(disease: str, data: list):
    if disease not in models:
        raise HTTPException(status_code=404, detail=f"Model for {disease} not found")
    
    try:
        predictions = []
        for item in data:
            # Process each item based on disease type
            # This would need to be implemented based on your specific needs
            pass
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)