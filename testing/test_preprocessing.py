"""
Tests unitaires pour le preprocessing
"""

import pytest
import pandas as pd
import numpy as np

@pytest.mark.unit
def test_preprocessor_exists(preprocessor):
    """Vérifie que le preprocessor existe"""
    assert preprocessor is not None, "Preprocessor not found"

@pytest.mark.unit
def test_preprocessor_transform(preprocessor, sample_data):
    """Vérifie que le preprocessor peut transformer les données"""
    if preprocessor is None:
        pytest.skip("No preprocessor available")
    
    target_col = 'Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn'
    X = sample_data.drop(columns=[target_col])
    
    X_transformed = preprocessor.transform(X)
    
    assert X_transformed is not None
    assert len(X_transformed) == len(X)
    print(f"Transformed shape: {X_transformed.shape}")

@pytest.mark.unit
def test_no_inf_values(sample_data):
    """Vérifie qu'il n'y a pas de valeurs infinies"""
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not np.isinf(sample_data[col]).any(), f"Infinite values in {col}"

@pytest.mark.unit
def test_data_types(sample_data):
    """Vérifie les types de données"""
    # Au moins quelques colonnes numériques
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) > 0, "No numeric columns found"
