"""
Pytest fixtures pour les tests
"""

import pytest
import pandas as pd
import pickle
from pathlib import Path

@pytest.fixture
def sample_data():
    """Charge un échantillon de données pour les tests"""
    return pd.read_csv("data/train/part1.csv").head(1000)

@pytest.fixture
def preprocessor():
    """Charge le preprocessor"""
    preprocessor_path = "notebooks/processors/preprocessor.pkl"
    if Path(preprocessor_path).exists():
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
    return None

@pytest.fixture
def production_data():
    """Charge les données de production si disponibles"""
    prod_path = "data/production/bank_churn_prod.csv"
    if Path(prod_path).exists():
        return pd.read_csv(prod_path)
    return None
