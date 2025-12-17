"""
Inference preprocessing module that reuses ChurnPreprocessor logic.
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add parent directory to path to import preprocessing_churn_class
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from preprocessing_churn_class import ChurnPreprocessor
from app.core.config import settings

logger = logging.getLogger(__name__)


class InferencePreprocessor:
    """
    Preprocessor for inference that applies the same transformations
    as the training pipeline but for single predictions.
    """
    
    def __init__(self, processors_dir: Optional[str] = None):
        """
        Initialize the inference preprocessor.
        
        Parameters:
        -----------
        processors_dir : str, optional
            Directory containing saved processors. Defaults to settings.PROCESSORS_DIR
        """
        self.processors_dir = Path(processors_dir or settings.PROCESSORS_DIR)
        
        # Load saved processors
        self.scaler = self._load_processor("scaler.pkl")
        self.label_encoders = self._load_processor("label_encoders.pkl")
        self.feature_names = self._load_processor("feature_names.pkl")
        self.smote_config = self._load_processor("smote_config.pkl")
        
        if self.feature_names is None:
            raise ValueError("feature_names.pkl not found. Please run training pipeline first.")
        
        self.expected_features = self.feature_names.get('all_features', [])
        self.numerical_features = self.feature_names.get('numerical_features', [])
        
        logger.info(f"Loaded processors from {self.processors_dir}")
        logger.info(f"Expected features: {len(self.expected_features)}")
    
    def _load_processor(self, filename: str):
        """Load a processor from pickle file."""
        filepath = self.processors_dir / filename
        if not filepath.exists():
            logger.warning(f"Processor file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return None
    
    def transform_request(self, payload: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single prediction request into model-ready format.
        
        Parameters:
        -----------
        payload : dict
            Dictionary containing customer features matching the original CSV columns
            
        Returns:
        --------
        np.ndarray
            Preprocessed feature array ready for model prediction
        """
        try:
            # Convert payload to DataFrame (single row)
            df = pd.DataFrame([payload])
            
            # Apply preprocessing steps (same as training pipeline)
            df_processed = self._apply_preprocessing(df)
            
            # Ensure feature alignment with training data
            df_aligned = self._align_features(df_processed)
            
            # Scale numerical features
            if self.scaler is not None:
                df_aligned[self.numerical_features] = self.scaler.transform(
                    df_aligned[self.numerical_features]
                )
            
            # Convert to numpy array in the correct feature order
            X = df_aligned[self.expected_features].values
            
            return X
            
        except Exception as e:
            logger.error(f"Error in transform_request: {e}")
            raise ValueError(f"Preprocessing failed: {str(e)}")
    
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same preprocessing steps as training."""
        df_clean = df.copy()
        
        # 1. Drop unnecessary columns (if present)
        columns_to_drop = [
            'RowNumber', 'CustomerId', 'Surname', 'First Name',
            'Address', 'Contact Information', 'Churn Reason', 'Churn Date'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        df_clean = df_clean.drop(columns=columns_to_drop)
        
        # 2. Engineer date features
        if 'Date of Birth' in df_clean.columns:
            df_clean['Date of Birth'] = pd.to_datetime(
                df_clean['Date of Birth'],
                format='%Y-%m-%d',
                errors='coerce'
            )
            reference_date = pd.Timestamp(settings.REFERENCE_DATE)
            df_clean['Age'] = (reference_date - df_clean['Date of Birth']).dt.days / 365.25
            df_clean['Age'] = df_clean['Age'].round(0).astype(int)
            
            # Create age groups
            df_clean['Age_Group'] = pd.cut(
                df_clean['Age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
            df_clean = df_clean.drop(columns=['Date of Birth'])
        
        # 3. Engineer features
        if 'Income' in df_clean.columns and 'Number of Dependents' in df_clean.columns:
            df_clean['Income_Per_Dependent'] = (
                df_clean['Income'] / (df_clean['Number of Dependents'] + 1)
            )
        
        if 'Balance' in df_clean.columns and 'NumOfProducts' in df_clean.columns:
            df_clean['Balance_Per_Product'] = (
                df_clean['Balance'] / df_clean['NumOfProducts']
            )
        
        if 'Outstanding Loans' in df_clean.columns and 'Income' in df_clean.columns:
            df_clean['Credit_Utilization'] = (
                df_clean['Outstanding Loans'] / df_clean['Income']
            )
        
        if 'Outstanding Loans' in df_clean.columns and 'Balance' in df_clean.columns:
            df_clean['Loan_To_Balance_Ratio'] = (
                df_clean['Outstanding Loans'] / (df_clean['Balance'] + 1)
            )
        
        if 'Customer Tenure' in df_clean.columns:
            df_clean['Tenure_Group'] = pd.cut(
                df_clean['Customer Tenure'],
                bins=[0, 6, 12, 24, 30],
                labels=['0-6m', '6-12m', '1-2y', '2y+']
            )
        
        if 'Credit Score' in df_clean.columns:
            df_clean['Credit_Category'] = pd.cut(
                df_clean['Credit Score'],
                bins=[0, 579, 669, 739, 799, 850],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            )
        
        if 'NumOfProducts' in df_clean.columns and 'Customer Tenure' in df_clean.columns:
            df_clean['Products_Per_Year'] = (
                df_clean['NumOfProducts'] / (df_clean['Customer Tenure'] / 12 + 0.1)
            )
        
        if 'NumComplaints' in df_clean.columns and 'Customer Tenure' in df_clean.columns:
            df_clean['Complaints_Per_Year'] = (
                df_clean['NumComplaints'] / (df_clean['Customer Tenure'] / 12 + 0.1)
            )
        
        # High value customer flag (using training quantiles if available)
        # For inference, we'll use a simple threshold approach
        if 'Balance' in df_clean.columns and 'NumOfProducts' in df_clean.columns:
            # Use reasonable defaults for inference
            balance_threshold = df_clean['Balance'].quantile(0.75) if len(df_clean) > 1 else 100000
            df_clean['High_Value_Customer'] = (
                (df_clean['Balance'] > balance_threshold) & 
                (df_clean['NumOfProducts'] >= 3)
            ).astype(int)
        
        # At-risk flag
        if 'NumComplaints' in df_clean.columns and 'Balance' in df_clean.columns:
            complaints_median = df_clean['NumComplaints'].median() if len(df_clean) > 1 else 0
            balance_median = df_clean['Balance'].median() if len(df_clean) > 1 else 50000
            df_clean['At_Risk'] = (
                (df_clean['NumComplaints'] > complaints_median) & 
                (df_clean['Balance'] < balance_median)
            ).astype(int)
        
        # 4. Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # 5. Encode categorical variables
        if 'Gender' in df_clean.columns:
            df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
        
        # One-hot encoding
        categorical_to_encode = [
            'Marital Status', 'Education Level', 'Customer Segment',
            'Preferred Communication Channel', 'Age_Group', 'Tenure_Group',
            'Credit_Category'
        ]
        categorical_to_encode = [col for col in categorical_to_encode if col in df_clean.columns]
        
        if categorical_to_encode:
            df_encoded = pd.get_dummies(
                df_clean,
                columns=categorical_to_encode,
                drop_first=True,
                dtype=int
            )
        else:
            df_encoded = df_clean.copy()
        
        # Label encoding for Occupation
        if 'Occupation' in df_encoded.columns and self.label_encoders and 'Occupation' in self.label_encoders:
            le = self.label_encoders['Occupation']
            # Handle unseen categories
            if df_encoded['Occupation'].iloc[0] in le.classes_:
                df_encoded['Occupation_Encoded'] = le.transform([df_encoded['Occupation'].iloc[0]])[0]
            else:
                # Use most common class as default
                df_encoded['Occupation_Encoded'] = 0
            df_encoded = df_encoded.drop(columns=['Occupation'])
        
        # 6. Handle outliers (cap at percentiles)
        outlier_cols = [
            'Income', 'Outstanding Loans', 'Balance', 'Income_Per_Dependent',
            'Balance_Per_Product', 'Credit_Utilization'
        ]
        for col in outlier_cols:
            if col in df_encoded.columns:
                # For single row, use simple clipping based on reasonable ranges
                # In production, you might want to load saved percentile values
                if col == 'Income':
                    df_encoded[col] = df_encoded[col].clip(lower=0, upper=500000)
                elif col == 'Outstanding Loans':
                    df_encoded[col] = df_encoded[col].clip(lower=0, upper=200000)
                elif col == 'Balance':
                    df_encoded[col] = df_encoded[col].clip(lower=0, upper=1000000)
        
        return df_encoded
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure feature alignment with training data.
        Add missing columns (with zeros) and remove extra columns.
        """
        # Create a DataFrame with all expected features
        df_aligned = pd.DataFrame(columns=self.expected_features)
        
        # Copy existing columns
        for col in self.expected_features:
            if col in df.columns:
                df_aligned[col] = df[col].values
            else:
                # Fill missing columns with 0
                df_aligned[col] = 0
        
        # Ensure correct order
        df_aligned = df_aligned[self.expected_features]
        
        return df_aligned

