"""
Pydantic schemas for prediction requests and responses.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import date


class ChurnFeaturesRequest(BaseModel):
    """Request schema for churn prediction."""
    
    # Customer Identification (optional)
    customer_id: Optional[str] = Field(None, alias="CustomerId", description="Customer ID")
    surname: Optional[str] = Field(None, alias="Surname", description="Customer surname")
    first_name: Optional[str] = Field(None, alias="First Name", description="Customer first name")
    
    # Personal Information
    date_of_birth: date = Field(..., alias="Date of Birth", description="Customer date of birth")
    gender: str = Field(..., alias="Gender", description="Gender (Male/Female)")
    marital_status: str = Field(..., alias="Marital Status", description="Marital status")
    number_of_dependents: int = Field(..., alias="Number of Dependents", ge=0, description="Number of dependents")
    occupation: str = Field(..., alias="Occupation", description="Occupation")
    education_level: str = Field(..., alias="Education Level", description="Education level")
    
    # Account Information
    customer_tenure: int = Field(..., alias="Customer Tenure", ge=0, description="Customer tenure in months (converted from years in frontend)")
    customer_segment: str = Field(..., alias="Customer Segment", description="Customer segment")
    preferred_communication_channel: str = Field(..., alias="Preferred Communication Channel", description="Preferred communication channel")
    balance: float = Field(..., alias="Balance", ge=0, description="Account balance")
    num_of_products: int = Field(..., alias="NumOfProducts", ge=1, description="Number of products")
    
    # Credit Information
    credit_score: int = Field(..., alias="Credit Score", ge=300, le=850, description="Credit score")
    credit_history_length: int = Field(..., alias="Credit History Length", ge=0, description="Credit history length in months")
    outstanding_loans: float = Field(..., alias="Outstanding Loans", ge=0, description="Outstanding loans amount")
    
    # Engagement
    income: float = Field(..., alias="Income", ge=0, description="Annual income")
    num_complaints: int = Field(..., alias="NumComplaints", ge=0, description="Number of complaints")
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "CustomerId": "123456",
                "Surname": "Smith",
                "First Name": "John",
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
        }


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn")
    churn_label: int = Field(..., ge=0, le=1, description="Predicted churn label (0=No, 1=Yes)")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "churn_probability": 0.75,
                "churn_label": 1,
                "model_version": "1"
            }
        }


class ChurnBatchRequest(BaseModel):
    """Request schema for batch churn predictions."""
    
    customers: List[ChurnFeaturesRequest] = Field(..., description="List of customer features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "CustomerId": "123456",
                        "Surname": "Smith",
                        "First Name": "John",
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
                ]
            }
        }


class ChurnBatchResponse(BaseModel):
    """Response schema for batch churn predictions."""
    
    predictions: List[ChurnPredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: Optional[str] = None
    version: Optional[str] = None
    stage: Optional[str] = None
    run_id: Optional[str] = None

