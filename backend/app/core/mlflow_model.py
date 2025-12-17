"""
MLflow model loading and management utilities.
"""
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Optional[Any] = None
_model_info: Optional[Dict[str, Any]] = None


def initialize_mlflow():
    """Initialize MLflow with tracking URI."""
    try:
        # Convert relative path to absolute if needed
        tracking_uri = settings.MLFLOW_TRACKING_URI
        if tracking_uri.startswith("file:///./"):
            # Convert relative path to absolute
            project_root = Path(__file__).parent.parent.parent.parent
            relative_path = tracking_uri.replace("file:///./", "")
            absolute_path = (project_root / relative_path).resolve()
            tracking_uri = f"file:///{absolute_path.as_posix()}"
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    except Exception as e:
        logger.error(f"Failed to set MLflow tracking URI: {e}")
        raise


def get_model():
    """
    Load and cache the model from MLflow Model Registry.
    
    Returns:
        The loaded MLflow model (sklearn model)
    
    Raises:
        Exception: If model cannot be loaded from MLflow
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    try:
        initialize_mlflow()
        logger.info(f"Loading model from MLflow: {settings.MLFLOW_MODEL_URI}")
        _model_cache = mlflow.sklearn.load_model(settings.MLFLOW_MODEL_URI)
        logger.info("Model loaded successfully from MLflow")
        return _model_cache
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        raise


def get_model_info() -> Optional[Dict[str, Any]]:
    """
    Get model metadata from MLflow.
    
    Returns:
        Dictionary containing model information (version, run_id, etc.)
    """
    global _model_info
    
    if _model_info is not None:
        return _model_info
    
    try:
        initialize_mlflow()
        
        # Parse model URI to extract model name and stage
        if settings.MLFLOW_MODEL_URI.startswith("models:/"):
            uri_parts = settings.MLFLOW_MODEL_URI.replace("models:/", "").split("/")
            model_name = uri_parts[0]
            stage_or_version = uri_parts[1] if len(uri_parts) > 1 else None
            
            client = mlflow.tracking.MlflowClient()
            
            if stage_or_version and stage_or_version in ["Production", "Staging", "Archived", "None"]:
                # Get model version by stage
                model_versions = client.get_latest_versions(model_name, stages=[stage_or_version])
                if model_versions:
                    mv = model_versions[0]
                    _model_info = {
                        "model_name": model_name,
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "run_id": mv.run_id,
                        "creation_timestamp": mv.creation_timestamp,
                    }
            else:
                # Get model version by version number
                mv = client.get_model_version(model_name, stage_or_version)
                _model_info = {
                    "model_name": model_name,
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "creation_timestamp": mv.creation_timestamp,
                }
            
            logger.info(f"Model info retrieved: {_model_info}")
            return _model_info
        else:
            logger.warning("Model URI format not supported for info extraction")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def clear_model_cache():
    """Clear the model cache (useful for testing or model updates)."""
    global _model_cache, _model_info
    _model_cache = None
    _model_info = None
    logger.info("Model cache cleared")

