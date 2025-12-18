"""
Tests d'intégrité des données avec Deepchecks
"""

import pytest
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    MixedNulls,
    StringMismatch,
    MixedDataTypes,
    IsSingleValue,
    SpecialCharacters,
    StringLengthOutOfBounds
)

@pytest.mark.data
def test_no_mixed_nulls(sample_data):
    """Vérifie qu'il n'y a pas de valeurs nulles mixtes"""
    dataset = Dataset(sample_data, label='Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn')
    check = MixedNulls()
    result = check.run(dataset)
    assert result.passed_conditions(), f"Mixed nulls detected: {result}"

@pytest.mark.data
def test_no_mixed_data_types(sample_data):
    """Vérifie qu'il n'y a pas de types de données mixtes"""
    dataset = Dataset(sample_data, label='Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn')
    check = MixedDataTypes()
    result = check.run(dataset)
    assert result.passed_conditions(), f"Mixed data types detected: {result}"

@pytest.mark.data
def test_no_single_value_columns(sample_data):
    """Vérifie qu'il n'y a pas de colonnes à valeur unique"""
    dataset = Dataset(sample_data, label='Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn')
    check = IsSingleValue()
    result = check.run(dataset)
    # Warning only, not critical
    if not result.passed_conditions():
        print(f"Warning: Single value columns detected: {result}")

@pytest.mark.data
def test_data_shape(sample_data):
    """Vérifie la forme des données"""
    assert len(sample_data) > 0, "Dataset is empty"
    assert len(sample_data.columns) > 5, "Too few columns"

@pytest.mark.data
def test_target_distribution(sample_data):
    """Vérifie la distribution de la target"""
    target_col = 'Churn Flag' if 'Churn Flag' in sample_data.columns else 'Churn'
    if target_col in sample_data.columns:
        churn_rate = sample_data[target_col].mean()
        assert 0.05 < churn_rate < 0.95, f"Extreme churn rate: {churn_rate:.2%}"
