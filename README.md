# Bank Churn Prediction API

A full-stack application for predicting bank customer churn using FastAPI backend and React frontend, with MLflow model management.

## Architecture

- **Backend**: FastAPI application that loads models from MLflow and preprocesses data using the training pipeline
- **Frontend**: React TypeScript application with a form-based UI for customer data input
- **MLflow**: Model registry and tracking for model versioning and management

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       └── endpoints/
│   │   │           └── predict.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── mlflow_model.py
│   │   │   └── preprocessing_inference.py
│   │   ├── schemas/
│   │   │   └── prediction.py
│   │   └── main.py
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── Dockerfile
├── preprocessing_churn_class.py
├── notebooks/
│   ├── processors/
│   └── mlruns/
├── docker-compose.yml
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- MLflow model registered in the Model Registry
- Preprocessed data artifacts (scaler, label encoders, feature names) in `notebooks/processors/`

## Setup

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (create `.env` file in `backend/` or set system variables):
```bash
MLFLOW_TRACKING_URI=file:///./notebooks/mlruns
MLFLOW_MODEL_URI=models:/churn_prediction_Stacking_LR/Production
PROCESSORS_DIR=notebooks/processors
```

3. Run the backend:
```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Create `.env` file (optional, defaults to `http://localhost:8000`):
```bash
VITE_API_BASE_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Docker Deployment

### Using Docker Compose

1. Build and run all services:
```bash
docker-compose up --build
```

2. Access the application:
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

### Individual Docker Containers

**Backend:**
```bash
docker build -t churn-backend -f backend/Dockerfile .
docker run -p 8000:8000 churn-backend
```

**Frontend:**
```bash
cd frontend
docker build -t churn-frontend .
docker run -p 3000:80 churn-frontend
```

## API Endpoints

### Health Check
```
GET /api/v1/predict/health
```

### Model Information
```
GET /api/v1/predict/model-info
```

### Single Prediction
```
POST /api/v1/predict/predict
Content-Type: application/json

{
  "Date of Birth": "1985-06-15",
  "Gender": "Male",
  "Marital Status": "Married",
  "Number of Dependents": 2,
  "Occupation": "Engineer",
  "Education Level": "Bachelor",
  "Customer Tenure": 24,
  "Customer Segment": "Retail",
  "Preferred Communication Channel": "Email",
  "Balance": 50000.0,
  "NumOfProducts": 2,
  "Credit Score": 650,
  "Credit History Length": 60,
  "Outstanding Loans": 10000.0,
  "Income": 75000.0,
  "NumComplaints": 0
}
```

### Batch Prediction
```
POST /api/v1/predict/predict/batch
Content-Type: application/json

{
  "customers": [
    { ... },
    { ... }
  ]
}
```

## Development

### Backend Development

The backend uses FastAPI with automatic reloading. Make changes to the code and they will be reflected immediately.

### Frontend Development

The frontend uses Vite with hot module replacement. Changes to React components will update automatically.

## Model Management

The application loads models from MLflow Model Registry. To update the model:

1. Train and register a new model version in MLflow
2. Update the `MLFLOW_MODEL_URI` environment variable or transition the model to Production stage
3. Restart the backend service

## Preprocessing

The inference preprocessing pipeline mirrors the training pipeline:
- Date feature engineering
- Feature creation (income per dependent, credit utilization, etc.)
- Missing value imputation
- Categorical encoding (one-hot and label encoding)
- Outlier handling
- Feature scaling

All preprocessing artifacts are loaded from `notebooks/processors/` directory.

## Troubleshooting

### Model Not Found
- Verify MLflow tracking URI is correct
- Check that the model exists in the Model Registry
- Ensure model URI format: `models:/model_name/stage`

### Preprocessing Errors
- Verify processors directory exists and contains required files:
  - `scaler.pkl`
  - `label_encoders.pkl`
  - `feature_names.pkl`
  - `smote_config.pkl`

### CORS Errors
- Update `CORS_ORIGINS` in `backend/app/core/config.py` to include your frontend URL

## License

MIT

