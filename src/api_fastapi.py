"""
Titanic Survival Prediction - FastAPI REST API
Author: Senior ML Engineer
Phase 13: Deployment Layer - Option 2

This is a production-grade REST API for Titanic survival prediction.

Installation:
    pip install fastapi uvicorn pydantic

Run:
    uvicorn src.api_fastapi:app --reload
    
Access:
    - API docs: http://localhost:8000/docs
    - Redoc: http://localhost:8000/redoc
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import joblib
import pandas as pd
import os
from datetime import datetime


# ============================================================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ============================================================================

class PassengerInput(BaseModel):
    """
    Input schema for passenger data.
    
    Pydantic automatically validates inputs and generates API documentation.
    """
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="Gender (male or female)")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, le=10, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, le=10, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare in British Pounds")
    embarked: str = Field(..., description="Port of embarkation (S, C, or Q)")
    
    @validator('sex')
    def validate_sex(cls, v):
        """Validate gender input."""
        v = v.lower()
        if v not in ['male', 'female']:
            raise ValueError('Sex must be "male" or "female"')
        return v
    
    @validator('embarked')
    def validate_embarked(cls, v):
        """Validate embarkation port."""
        v = v.upper()
        if v not in ['S', 'C', 'Q']:
            raise ValueError('Embarked must be "S" (Southampton), "C" (Cherbourg), or "Q" (Queenstown)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "pclass": 1,
                "sex": "female",
                "age": 25,
                "sibsp": 0,
                "parch": 0,
                "fare": 100.0,
                "embarked": "S"
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction results."""
    survived: int = Field(..., description="Prediction: 0 = Did not survive, 1 = Survived")
    survival_probability: float = Field(..., description="Probability of survival (0-1)")
    death_probability: float = Field(..., description="Probability of death (0-1)")
    confidence: str = Field(..., description="Confidence level (Low/Moderate/High/Very High)")
    message: str = Field(..., description="Human-readable prediction message")
    passenger_profile: dict = Field(..., description="Summary of passenger characteristics")
    
    class Config:
        schema_extra = {
            "example": {
                "survived": 1,
                "survival_probability": 0.87,
                "death_probability": 0.13,
                "confidence": "Very High",
                "message": "Passenger would have SURVIVED",
                "passenger_profile": {
                    "class": 1,
                    "gender": "female",
                    "age": 25,
                    "family_size": 1,
                    "is_alone": True
                }
            }
        }


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="""
    🚢 **Production-Ready ML API for Titanic Survival Prediction**
    
    This API predicts whether a Titanic passenger would have survived based on
    demographic and travel information.
    
    **Features:**
    - Input validation using Pydantic
    - Comprehensive error handling
    - Automatic API documentation
    - Production-ready architecture
    
    **Model:** Random Forest Classifier (82%+ accuracy)
    
    **Author:** Senior ML Engineer
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# MODEL LOADING (ON STARTUP)
# ============================================================================

model = None
model_path = None


@app.on_event("startup")
async def load_model():
    """Load the ML model when the API starts."""
    global model, model_path
    
    # Try multiple paths
    possible_paths = [
        'models/titanic_production_pipeline.pkl',
        '../models/titanic_production_pipeline.pkl',
        'models/titanic_survival_model.pkl',
        '../models/titanic_survival_model.pkl'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                model_path = path
                print(f"✅ Model loaded successfully from: {path}")
                return
            except Exception as e:
                print(f"❌ Failed to load model from {path}: {e}")
    
    print("⚠️  WARNING: No model file found. API will not work properly.")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict"
    }


@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and model availability.
    """
    return HealthCheck(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_survival(passenger: PassengerInput):
    """
    Predict Titanic passenger survival.
    
    **Input:** Passenger demographic and travel information
    
    **Output:** Survival prediction with probability and confidence level
    
    **Example Request:**
    ```json
    {
        "pclass": 1,
        "sex": "female",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 100.0,
        "embarked": "S"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "survived": 1,
        "survival_probability": 0.87,
        "death_probability": 0.13,
        "confidence": "Very High",
        "message": "Passenger would have SURVIVED"
    }
    ```
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Calculate derived features
        family_size = passenger.sibsp + passenger.parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Create DataFrame
        passenger_df = pd.DataFrame([{
            'pclass': passenger.pclass,
            'sex': passenger.sex.lower(),
            'age': passenger.age,
            'sibsp': passenger.sibsp,
            'parch': passenger.parch,
            'fare': passenger.fare,
            'embarked': passenger.embarked.upper(),
            'family_size': family_size,
            'is_alone': is_alone
        }])
        
        # Make prediction
        try:
            prediction = int(model.predict(passenger_df)[0])
            probabilities = model.predict_proba(passenger_df)[0]
        except Exception:
            # Try without engineered features if model doesn't have them
            passenger_df = pd.DataFrame([{
                'pclass': passenger.pclass,
                'sex': passenger.sex.lower(),
                'age': passenger.age,
                'sibsp': passenger.sibsp,
                'parch': passenger.parch,
                'fare': passenger.fare,
                'embarked': passenger.embarked.upper()
            }])
            prediction = int(model.predict(passenger_df)[0])
            probabilities = model.predict_proba(passenger_df)[0]
        
        survival_prob = float(probabilities[1])
        death_prob = float(probabilities[0])
        
        # Determine confidence level
        max_prob = max(survival_prob, death_prob)
        if max_prob >= 0.80:
            confidence = "Very High"
        elif max_prob >= 0.65:
            confidence = "High"
        elif max_prob >= 0.55:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        # Create response
        return PredictionOutput(
            survived=prediction,
            survival_probability=survival_prob,
            death_probability=death_prob,
            confidence=confidence,
            message=f"Passenger would have {'SURVIVED' if prediction == 1 else 'NOT SURVIVED'}",
            passenger_profile={
                "class": passenger.pclass,
                "gender": passenger.sex,
                "age": passenger.age,
                "family_size": family_size,
                "is_alone": is_alone == 1,
                "fare": passenger.fare,
                "embarked": passenger.embarked
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_survival_batch(passengers: list[PassengerInput]):
    """
    Predict survival for multiple passengers.
    
    **Input:** List of passenger data
    
    **Output:** List of predictions
    
    **Example:**
    ```json
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 25,
            "sibsp": 0,
            "parch": 0,
            "fare": 100.0,
            "embarked": "S"
        },
        {
            "pclass": 3,
            "sex": "male",
            "age": 30,
            "sibsp": 1,
            "parch": 2,
            "fare": 15.0,
            "embarked": "Q"
        }
    ]
    ```
    """
    results = []
    
    for passenger in passengers:
        result = await predict_survival(passenger)
        results.append(result)
    
    return results


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "model_path": model_path,
        "features_required": [
            "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"
        ],
        "output": "Binary classification (0 = Did not survive, 1 = Survived)",
        "accuracy": "82%+",
        "algorithm": "Random Forest Classifier (Tuned)"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /docs"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    return {
        "error": "Internal server error",
        "message": "Something went wrong. Please try again later."
    }


# ============================================================================
# EXAMPLE CLIENT CODE
# ============================================================================

"""
USAGE EXAMPLES:

1. Python (requests library):
   
   import requests
   
   response = requests.post(
       "http://localhost:8000/predict",
       json={
           "pclass": 1,
           "sex": "female",
           "age": 25,
           "sibsp": 0,
           "parch": 0,
           "fare": 100.0,
           "embarked": "S"
       }
   )
   
   result = response.json()
   print(f"Survived: {result['survived']}")
   print(f"Probability: {result['survival_probability']}")


2. cURL:
   
   curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "pclass": 1,
       "sex": "female",
       "age": 25,
       "sibsp": 0,
       "parch": 0,
       "fare": 100.0,
       "embarked": "S"
     }'


3. JavaScript (fetch):
   
   fetch('http://localhost:8000/predict', {
       method: 'POST',
       headers: {
           'Content-Type': 'application/json'
       },
       body: JSON.stringify({
           pclass: 1,
           sex: 'female',
           age: 25,
           sibsp: 0,
           parch: 0,
           fare: 100.0,
           embarked: 'S'
       })
   })
   .then(response => response.json())
   .then(data => console.log(data));


4. Postman:
   - Method: POST
   - URL: http://localhost:8000/predict
   - Body (JSON):
     {
         "pclass": 1,
         "sex": "female",
         "age": 25,
         "sibsp": 0,
         "parch": 0,
         "fare": 100.0,
         "embarked": "S"
     }
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
