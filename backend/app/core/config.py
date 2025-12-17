"""
Application configuration settings.
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", 
        "file:///./notebooks/mlruns"
    )
    MLFLOW_MODEL_URI: str = os.getenv(
        "MLFLOW_MODEL_URI",
        "models:/churn_prediction_Stacking_LR/Production"
    )
    
    # Preprocessing Configuration
    PROCESSORS_DIR: str = os.getenv(
        "PROCESSORS_DIR",
        "notebooks/processors"
    )
    REFERENCE_DATE: str = os.getenv(
        "REFERENCE_DATE",
        "2024-12-11"
    )
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Bank Churn Prediction API"
    VERSION: str = "1.0.0"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Prediction Configuration
    PREDICTION_THRESHOLD: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

