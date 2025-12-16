from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the model from MLflow
logged_model = 'runs:/<run_id>/<artifact_path>' # Replace with your MLflow run ID and artifact path
loaded_model = mlflow.pyfunc.load_model(logged_model)

class PredictionRequest(BaseModel):
    # Define the input features for prediction
    # Example: feature1: float
    pass

class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])
    
    # Make prediction
    prediction = loaded_model.predict(data)
    
    return PredictionResponse(prediction=prediction[0])

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
