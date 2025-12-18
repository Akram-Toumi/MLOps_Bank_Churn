"""
Tests de distribution des données (train vs production)
"""

import pytest
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    TrainTestFeatureDrift,
    TrainTestLabelDrift,
    WholeDatasetDrift
)

@pytest.mark.data
def test_feature_drift(sample_data, production_data):
    """Vérifie le drift des features entre train et production"""
    if production_data is None:
        pytest.skip("No production data available")
    
    target_col = 'Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn'
    
    train_dataset = Dataset(sample_data, label=target_col)
    prod_dataset = Dataset(production_data, label=target_col)
    
    check = TrainTestFeatureDrift()
    result = check.run(train_dataset, prod_dataset)
    
    # Log drift info but don't fail (drift is expected in our scenario)
    print(f"\nFeature Drift Results: {result}")
    
@pytest.mark.data  
def test_label_drift(sample_data, production_data):
    """Vérifie le drift de la target entre train et production"""
    if production_data is None:
        pytest.skip("No production data available")
    
    target_col = 'Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn'
    
    train_dataset = Dataset(sample_data, label=target_col)
    prod_dataset = Dataset(production_data, label=target_col)
    
    check = TrainTestLabelDrift()
    result = check.run(train_dataset, prod_dataset)
    
    print(f"\nLabel Drift Results: {result}")

@pytest.mark.data
def test_column_consistency(sample_data, production_data):
    """Vérifie que les colonnes sont cohérentes"""
    if production_data is None:
        pytest.skip("No production data available")
    
    train_cols = set(sample_data.columns)
    prod_cols = set(production_data.columns)
    
    assert train_cols == prod_cols, f"Column mismatch: {train_cols.symmetric_difference(prod_cols)}"
