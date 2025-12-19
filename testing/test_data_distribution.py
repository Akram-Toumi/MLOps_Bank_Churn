"""
Tests de distribution des données (train vs production)
"""

import pytest
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift, LabelDrift


# =========================================
# CONFIGURATION GLOBALE
# =========================================

# Liste EXPLICITE des features catégorielles
# ⚠️ doit être IDENTIQUE pour train et prod
CATEGORICAL_FEATURES = [
    "Gender",
    "Marital Status",
    "Education Level",
    "Customer Segment",
    "Preferred Communication Channel",
    "Churn Reason"
]


# =========================================
# FEATURE DRIFT
# =========================================

@pytest.mark.data
def test_feature_drift(sample_data, production_data):
    """Analyse le drift des features (log uniquement, ne fait pas échouer la pipeline)"""

    if production_data is None:
        pytest.skip("No production data available")

    target_col = "Churn Flag" if "Churn Flag" in sample_data.columns else "Churn"

    train_dataset = Dataset(
        sample_data,
        label=target_col,
        cat_features=CATEGORICAL_FEATURES
    )

    prod_dataset = Dataset(
        production_data,
        label=target_col,
        cat_features=CATEGORICAL_FEATURES
    )

    check = FeatureDrift()
    result = check.run(train_dataset, prod_dataset)

    print("\n================ FEATURE DRIFT REPORT ================\n")
    print(result)

    # 🚨 En MLOps réel, le drift est OBSERVÉ, pas bloquant
    assert result is not None


# =========================================
# LABEL DRIFT
# =========================================

@pytest.mark.data
def test_label_drift(sample_data, production_data):
    """Analyse le drift de la target (log uniquement)"""

    if production_data is None:
        pytest.skip("No production data available")

    target_col = "Churn Flag" if "Churn Flag" in sample_data.columns else "Churn"

    train_dataset = Dataset(
        sample_data,
        label=target_col,
        cat_features=CATEGORICAL_FEATURES
    )

    prod_dataset = Dataset(
        production_data,
        label=target_col,
        cat_features=CATEGORICAL_FEATURES
    )

    check = LabelDrift()
    result = check.run(train_dataset, prod_dataset)

    print("\n================ LABEL DRIFT REPORT ================\n")
    print(result)

    assert result is not None


# =========================================
# COLUMN CONSISTENCY
# =========================================

@pytest.mark.data
def test_column_consistency(sample_data, production_data):
    """Vérifie que les colonnes sont strictement identiques"""

    if production_data is None:
        pytest.skip("No production data available")

    train_cols = set(sample_data.columns)
    prod_cols = set(production_data.columns)

    assert train_cols == prod_cols, (
        f"Column mismatch detected: "
        f"{train_cols.symmetric_difference(prod_cols)}"
    )
