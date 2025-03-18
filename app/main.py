# app.py
import os
import logging
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from pydantic import BaseModel
import mlflow
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SOM_API")

# Load configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "SOM_RGB_Training")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")


# Initialize FastAPI
# app = FastAPI(
#     title="SOM Inference API",
#     description="API for inference with Self-Organizing Map models",
#     version="1.0.0",
# )
@asynccontextmanager
async def lifespan(app):
    print("Application is starting...")  # Startup logic
    yield
    print("Application is shutting down...")  # Shutdown logic


app = FastAPI(
    lifespan=lifespan,
    title="SOM Inference API",
    description="API for inference with Self-Organizing Map models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define input/output models
class InputData(BaseModel):
    data: List[List[float]]


class MappingResult(BaseModel):
    positions: List[List[int]]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_info: Dict[str, Any]


# Set up MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variable for the model
model = None


@app.get("/")
async def load_model():
    """Load the model on startup."""
    global model
    try:
        logger.info(f"Loading model {MODEL_NAME} from MLflow")
        # Load the model from MLflow
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # We'll initialize with None and let the endpoints handle the error


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get model info
    model_info = {
        "name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "loaded_at": datetime.now().isoformat(),
    }

    return HealthResponse(status="healthy", version="1.0.0", model_info=model_info)


@app.post("/predict", response_model=MappingResult)
async def predict(input_data: InputData):
    """
    Map input data to SOM grid coordinates.

    Args:
        input_data: Input data points

    Returns:
        Mapped positions on the SOM grid
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input data to numpy array
        data = np.array(input_data.data)

        # Validate input shape
        if len(data.shape) != 2:
            raise HTTPException(status_code=400, detail="Input data must be a 2D array")

        # Perform inference
        start_time = datetime.now()
        positions = model.predict(data).tolist()
        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        return MappingResult(positions=positions, inference_time_ms=inference_time)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/metadata")
async def get_metadata():
    """Return model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get model details
    # This depends on your specific SOM implementation
    try:
        # Try to get som attributes if available
        metadata = {
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
            "width": model._model_impl.python_model.som.width,
            "height": model._model_impl.python_model.som.height,
            "input_dim": model._model_impl.python_model.som.input_dim,
        }
    except:
        # Fallback to generic metadata
        metadata = {
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
        }

    return metadata


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
