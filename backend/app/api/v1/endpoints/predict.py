"""
Prediction endpoints for churn prediction API.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import numpy as np

from app.schemas.prediction import (
    ChurnFeaturesRequest,
    ChurnPredictionResponse,
    ChurnBatchRequest,
    ChurnBatchResponse,
    HealthResponse,
    ModelInfoResponse
)
from app.core.mlflow_model import get_model, get_model_info
from app.core.preprocessing_inference import InferencePreprocessor
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global preprocessor instance (initialized at startup)
_inference_preprocessor: InferencePreprocessor = None


def get_preprocessor() -> InferencePreprocessor:
    """Get the inference preprocessor instance."""
    global _inference_preprocessor
    if _inference_preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not initialized. Please check server startup."
        )
    return _inference_preprocessor


def init_preprocessor():
    """Initialize the inference preprocessor (called at startup)."""
    global _inference_preprocessor
    try:
        _inference_preprocessor = InferencePreprocessor()
        logger.info("Inference preprocessor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        raise


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    try:
        info = get_model_info()
        if info:
            return ModelInfoResponse(**info)
        else:
            return ModelInfoResponse()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(
    request: ChurnFeaturesRequest,
    preprocessor: InferencePreprocessor = Depends(get_preprocessor)
):
    """
    Predict churn for a single customer.
    
    Parameters:
    -----------
    request : ChurnFeaturesRequest
        Customer features for prediction
        
    Returns:
    --------
    ChurnPredictionResponse
        Prediction result with probability and label
    """
    try:
        # Convert Pydantic model to dict (using aliases)
        payload = request.model_dump(by_alias=True)
        
        # Preprocess the request
        X = preprocessor.transform_request(payload)
        
        # Get model and make prediction
        model = get_model()
        
        # Get probability
        proba = model.predict_proba(X)[0]
        churn_probability = float(proba[1])  # Probability of churn (class 1)
        
        # Get label based on threshold
        churn_label = int(churn_probability >= settings.PREDICTION_THRESHOLD)
        
        # Get model version info
        model_info = get_model_info()
        model_version = model_info.get('version') if model_info else None
        
        return ChurnPredictionResponse(
            churn_probability=churn_probability,
            churn_label=churn_label,
            model_version=str(model_version) if model_version else None
        )
        
    except ValueError as e:
        logger.error(f"Validation error in prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=ChurnBatchResponse)
async def predict_churn_batch(
    request: ChurnBatchRequest,
    preprocessor: InferencePreprocessor = Depends(get_preprocessor)
):
    """
    Predict churn for multiple customers in batch.
    
    Parameters:
    -----------
    request : ChurnBatchRequest
        List of customer features for prediction
        
    Returns:
    --------
    ChurnBatchResponse
        List of predictions
    """
    try:
        model = get_model()
        predictions = []
        
        # Get model version info
        model_info = get_model_info()
        model_version = model_info.get('version') if model_info else None
        
        for customer in request.customers:
            # Convert to dict
            payload = customer.model_dump(by_alias=True)
            
            # Preprocess
            X = preprocessor.transform_request(payload)
            
            # Predict
            proba = model.predict_proba(X)[0]
            churn_probability = float(proba[1])
            churn_label = int(churn_probability >= settings.PREDICTION_THRESHOLD)
            
            predictions.append(
                ChurnPredictionResponse(
                    churn_probability=churn_probability,
                    churn_label=churn_label,
                    model_version=str(model_version) if model_version else None
                )
            )
        
        return ChurnBatchResponse(
            predictions=predictions,
            total=len(predictions)
        )
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

