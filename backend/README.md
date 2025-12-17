# Bank Churn Prediction Backend

FastAPI backend for bank churn prediction using MLflow models.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (see `.env.example`)

3. Run the server:
```bash
uvicorn app.main:app --reload
```

## Configuration

Environment variables:
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `MLFLOW_MODEL_URI`: Model URI in format `models:/model_name/stage`
- `PROCESSORS_DIR`: Directory containing preprocessing artifacts
- `REFERENCE_DATE`: Reference date for age calculation
- `PREDICTION_THRESHOLD`: Threshold for churn classification (default: 0.5)

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

