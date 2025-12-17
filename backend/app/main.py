import os
from functools import lru_cache
from typing import List

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Feature order as used during training (preprocessed_data.csv without "Churn Flag")
FEATURE_ORDER: List[str] = [
    "Gender",
    "Number of Dependents",
    "Income",
    "Customer Tenure",
    "Credit Score",
    "Credit History Length",
    "Outstanding Loans",
    "Balance",
    "NumOfProducts",
    "NumComplaints",
    "Age",
    "Income_Per_Dependent",
    "Balance_Per_Product",
    "Credit_Utilization",
    "Loan_To_Balance_Ratio",
    "Products_Per_Year",
    "Complaints_Per_Year",
    "High_Value_Customer",
    "At_Risk",
    "Marital Status_Married",
    "Marital Status_Single",
    "Education Level_Diploma",
    "Education Level_High School",
    "Education Level_Master's",
    "Customer Segment_Retail",
    "Customer Segment_SME",
    "Preferred Communication Channel_Phone",
    "Age_Group_26-35",
    "Age_Group_36-45",
    "Age_Group_46-55",
    "Age_Group_56-65",
    "Age_Group_65+",
    "Tenure_Group_6-12m",
    "Tenure_Group_1-2y",
    "Tenure_Group_2y+",
    "Credit_Category_Fair",
    "Credit_Category_Good",
    "Credit_Category_Very Good",
    "Credit_Category_Excellent",
    "Occupation_Encoded",
]


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction")
# Default model name based on the training notebook; override with env var if needed
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn_prediction_Stacking_LR")
N_FEATURES = int(os.getenv("N_FEATURES", str(len(FEATURE_ORDER))))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ChurnFeatures(BaseModel):
    """
    Input payload for churn prediction.

    For now this expects a preprocessed feature vector of length N_FEATURES (default: 40),
    in the same order as during training.
    """

    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


class CustomerDetails(BaseModel):
    """
    Raw customer details used to build the feature vector expected by the model.

    The fields mirror the original dataset and preprocessing notebook.
    """

    gender: str  # "Male" or "Female"
    number_of_dependents: int
    income: float
    customer_tenure: float  # in years (or approximate)
    credit_score: float
    credit_history_length: float
    outstanding_loans: float
    balance: float
    num_of_products: int
    num_complaints: int
    age: int

    marital_status: str  # e.g. "Married", "Single", "Divorced"
    education_level: str  # e.g. "High School", "Diploma", "Master's"
    customer_segment: str  # e.g. "Retail", "SME", "Corporate"
    preferred_communication_channel: str  # e.g. "Phone", "Email"

    age_group: str  # e.g. "18-25", "26-35", "36-45", "46-55", "56-65", "65+"
    tenure_group: str  # e.g. "0-6m", "6-12m", "1-2y", "2y+"
    credit_category: str  # "Poor", "Fair", "Good", "Very Good", "Excellent"

    # We don't have the original LabelEncoder mapping, so we treat occupation as optional
    # and fall back to a neutral encoded value when it's missing.
    occupation_encoded: int | None = None


def build_feature_vector(customer: CustomerDetails) -> List[float]:
    """
    Convert raw customer details into the 40-element feature vector used during training.
    """

    gender_num = 1.0 if customer.gender.lower() == "male" else 0.0

    deps = float(customer.number_of_dependents)
    income = float(customer.income)
    tenure = float(customer.customer_tenure) if customer.customer_tenure > 0 else 1.0
    credit_score = float(customer.credit_score)
    credit_history_len = float(customer.credit_history_length)
    outstanding_loans = float(customer.outstanding_loans)
    balance = float(customer.balance)
    num_products = float(customer.num_of_products) if customer.num_of_products > 0 else 1.0
    num_complaints = float(customer.num_complaints)
    age = float(customer.age)

    # Engineered features (matching the notebook formulas where known)
    income_per_dep = income / (customer.number_of_dependents + 1) if income > 0 else 0.0
    balance_per_product = balance / num_products if balance > 0 else 0.0
    credit_utilization = outstanding_loans / income if income > 0 else 0.0
    loan_to_balance_ratio = outstanding_loans / balance if balance > 0 else 0.0
    products_per_year = num_products / tenure if tenure > 0 else num_products
    complaints_per_year = num_complaints / tenure if tenure > 0 else num_complaints

    # Simple business rules for high value / at risk customers
    high_value = float(
        (balance > 200_000) and (income > 60_000) and (credit_score >= 700)
    )
    at_risk = float(
        (credit_score < 600)
        or (num_complaints >= 3)
        or (credit_utilization > 0.6)
    )

    # One-hot encodings for categorical variables
    marital = customer.marital_status.strip().lower()
    marital_married = 1.0 if marital == "married" else 0.0
    marital_single = 1.0 if marital == "single" else 0.0

    edu = customer.education_level.strip().lower()
    edu_diploma = 1.0 if "diploma" in edu else 0.0
    edu_high_school = 1.0 if "high school" in edu or "high_school" in edu else 0.0
    edu_masters = 1.0 if "master" in edu else 0.0

    seg = customer.customer_segment.strip().lower()
    seg_retail = 1.0 if seg == "retail" else 0.0
    seg_sme = 1.0 if seg == "sme" else 0.0

    comm = customer.preferred_communication_channel.strip().lower()
    comm_phone = 1.0 if comm == "phone" else 0.0

    age_group = customer.age_group.strip().lower()
    age_26_35 = 1.0 if "26-35" in age_group else 0.0
    age_36_45 = 1.0 if "36-45" in age_group else 0.0
    age_46_55 = 1.0 if "46-55" in age_group else 0.0
    age_56_65 = 1.0 if "56-65" in age_group else 0.0
    age_65_plus = 1.0 if "65+" in age_group or age_group.endswith("65+") else 0.0

    tenure_group = customer.tenure_group.strip().lower()
    ten_6_12 = 1.0 if "6-12" in tenure_group else 0.0
    ten_1_2 = 1.0 if "1-2" in tenure_group else 0.0
    ten_2_plus = 1.0 if "2y+" in tenure_group or "2+" in tenure_group else 0.0

    credit_cat = customer.credit_category.strip().lower()
    credit_fair = 1.0 if "fair" in credit_cat else 0.0
    credit_good = 1.0 if credit_cat == "good" else 0.0
    credit_very_good = 1.0 if "very good" in credit_cat or "very_good" in credit_cat else 0.0
    credit_excellent = 1.0 if "excellent" in credit_cat else 0.0

    occupation_encoded = float(customer.occupation_encoded) if customer.occupation_encoded is not None else 0.0

    features_map = {
        "Gender": gender_num,
        "Number of Dependents": deps,
        "Income": income,
        "Customer Tenure": tenure,
        "Credit Score": credit_score,
        "Credit History Length": credit_history_len,
        "Outstanding Loans": outstanding_loans,
        "Balance": balance,
        "NumOfProducts": num_products,
        "NumComplaints": num_complaints,
        "Age": age,
        "Income_Per_Dependent": income_per_dep,
        "Balance_Per_Product": balance_per_product,
        "Credit_Utilization": credit_utilization,
        "Loan_To_Balance_Ratio": loan_to_balance_ratio,
        "Products_Per_Year": products_per_year,
        "Complaints_Per_Year": complaints_per_year,
        "High_Value_Customer": high_value,
        "At_Risk": at_risk,
        "Marital Status_Married": marital_married,
        "Marital Status_Single": marital_single,
        "Education Level_Diploma": edu_diploma,
        "Education Level_High School": edu_high_school,
        "Education Level_Master's": edu_masters,
        "Customer Segment_Retail": seg_retail,
        "Customer Segment_SME": seg_sme,
        "Preferred Communication Channel_Phone": comm_phone,
        "Age_Group_26-35": age_26_35,
        "Age_Group_36-45": age_36_45,
        "Age_Group_46-55": age_46_55,
        "Age_Group_56-65": age_56_65,
        "Age_Group_65+": age_65_plus,
        "Tenure_Group_6-12m": ten_6_12,
        "Tenure_Group_1-2y": ten_1_2,
        "Tenure_Group_2y+": ten_2_plus,
        "Credit_Category_Fair": credit_fair,
        "Credit_Category_Good": credit_good,
        "Credit_Category_Very Good": credit_very_good,
        "Credit_Category_Excellent": credit_excellent,
        "Occupation_Encoded": occupation_encoded,
    }

    # Build the final vector in the exact order used during training
    return [float(features_map[name]) for name in FEATURE_ORDER]


app = FastAPI(
    title="Bank Churn Prediction API",
    version="0.1.0",
    description="FastAPI backend for the MLOps Bank Churn project connected to MLflow.",
)

# Allow the React frontend to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_mlflow_model(stage: str = "Production"):
    """
    Lazily load and cache the model from MLflow Model Registry.

    The model name and tracking URI are configured via environment variables:
    - MLFLOW_TRACKING_URI
    - MLFLOW_MODEL_NAME
    """

    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(
            f"Could not load MLflow model from '{model_uri}': {exc}"
        ) from exc
    return model


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.get("/")
def root():
    """
    Basic welcome endpoint.
    """
    return {
        "message": "Bank Churn Prediction API is running",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/mlflow/config")
def mlflow_config():
    """
    Expose basic MLflow configuration used by the backend.
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "model_name": MLFLOW_MODEL_NAME,
        "n_features": N_FEATURES,
    }


@app.get("/mlflow/runs")
def list_mlflow_runs(limit: int = 10):
    """
    List recent MLflow runs for the configured experiment.
    """
    try:
        runs_df = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            max_results=limit,
            order_by=["start_time DESC"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error while listing MLflow runs: {exc}"
        ) from exc

    if runs_df.empty:
        return []

    # Keep only a few useful columns if they exist
    preferred_cols = [
        "run_id",
        "status",
        "metrics.roc_auc",
        "metrics.f1_score",
        "params.model_name",
        "start_time",
        "end_time",
    ]
    cols = [c for c in preferred_cols if c in runs_df.columns]
    return runs_df[cols].to_dict(orient="records")


def _predict_from_features(features: List[float]) -> PredictionResponse:
    try:
        model = get_mlflow_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    X = np.array(features, dtype=float).reshape(1, -1)

    try:
        proba = float(model.predict_proba(X)[0, 1])
    except AttributeError:
        # Fallback if the model does not expose predict_proba
        pred_raw = model.predict(X)[0]
        proba = float(pred_raw)

    prediction = int(proba >= 0.5)
    return PredictionResponse(prediction=prediction, probability=proba)


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(payload: ChurnFeatures):
    """
    Predict churn using the model loaded from MLflow Model Registry.

    The input must be the same preprocessed feature vector used during training.
    """
    if len(payload.features) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} features, got {len(payload.features)}",
        )

    return _predict_from_features(payload.features)


@app.post("/predict_customer", response_model=PredictionResponse)
def predict_churn_from_customer(customer: CustomerDetails):
    """
    Predict churn directly from raw customer details.

    The backend converts the raw fields into the 40 engineered features
    and uses the MLflow-deployed model for inference.
    """
    features = build_feature_vector(customer)
    if len(features) != N_FEATURES:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: built {len(features)} features, expected {N_FEATURES}",
        )
    return _predict_from_features(features)
