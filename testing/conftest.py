"""
Pytest fixtures pour les tests
"""

import pytest
import pandas as pd
import pickle
from pathlib import Path
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_data():
    """Charge un échantillon de données pour les tests"""
    return pd.read_csv("data/train/part1.csv").head(1000)

@pytest.fixture
def preprocessor():
    """Charge le preprocessor"""
    preprocessor_path = "notebooks/processors/preprocessor.pkl"
    
    # Try to load existing preprocessor
    if Path(preprocessor_path).exists():
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
    
    # If preprocessor.pkl doesn't exist, try to create it from components
    processor_dir = "notebooks/processors"
    scaler_path = Path(processor_dir) / "scaler.pkl"
    label_encoders_path = Path(processor_dir) / "label_encoders.pkl"
    feature_names_path = Path(processor_dir) / "feature_names.pkl"
    
    if scaler_path.exists() and label_encoders_path.exists() and feature_names_path.exists():
        try:
            from scripts.preprocess_production import ProductionDataPreprocessor
            
            # Create preprocessor instance
            prep = ProductionDataPreprocessor()
            prep.load_processors(processor_dir)
            
            # Save it for future use
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(prep, f)
            
            return prep
        except Exception as e:
            print(f"Warning: Could not create preprocessor from components: {e}")
            return None
    
    return None

@pytest.fixture
def production_data():
    """Charge les données de production si disponibles"""
    prod_path = "data/production/bank_churn_prod.csv"
    if Path(prod_path).exists():
        return pd.read_csv(prod_path)
    return None
