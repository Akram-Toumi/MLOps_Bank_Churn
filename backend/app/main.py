"""
FastAPI application main entry point.
"""
import sys
from pathlib import Path

# Add backend directory to Python path to allow imports
# This ensures 'app' module can be found when running from project root
# Note: For uvicorn reloader to work, set PYTHONPATH=backend before running uvicorn
backend_dir = Path(__file__).parent.parent.resolve()
backend_dir_str = str(backend_dir)
if backend_dir_str not in sys.path:
    sys.path.insert(0, backend_dir_str)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.mlflow_model import get_model, initialize_mlflow
from app.core.preprocessing_inference import InferencePreprocessor
from app.api.v1.endpoints.predict import init_preprocessor
from app.api.v1 import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    
    try:
        # Initialize MLflow
        initialize_mlflow()
        logger.info("MLflow initialized")
        
        # Load model (cache it)
        model = get_model()
        logger.info(f"Model loaded: {type(model).__name__}")
        
        # Initialize preprocessor
        init_preprocessor()
        logger.info("Preprocessor initialized")
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Bank Churn Prediction API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

