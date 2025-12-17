"""
Script de test pour déboguer le preprocessing
"""
import pickle
import pandas as pd
from pathlib import Path

# Charger les preprocessors
PROCESSORS_DIR = Path("notebooks/processors")

# Charger label encoders
with open(PROCESSORS_DIR / "label_encoders.pkl", 'rb') as f:
    label_encoders = pickle.load(f)

# Charger scaler
with open(PROCESSORS_DIR / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Charger feature names
with open(PROCESSORS_DIR / "feature_names.pkl", 'rb') as f:
    feature_names = pickle.load(f)

print("=== Label Encoders ===")
print(f"Colonnes: {list(label_encoders.keys())}")

print("\n=== Feature Names ===")
print(f"Type: {type(feature_names)}")
print(f"Contenu: {feature_names}")

print("\n=== Scaler ===")
print(f"Type: {type(scaler)}")
if hasattr(scaler, 'n_features_in_'):
    print(f"Nombre de features attendues: {scaler.n_features_in_}")

# Test avec des données d'exemple
test_data = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
}

df = pd.DataFrame([test_data])
print("\n=== Données d'entrée ===")
print(df)
print(f"Colonnes: {df.columns.tolist()}")

# Appliquer label encoding
print("\n=== Après label encoding ===")
for column, encoder in label_encoders.items():
    if column in df.columns:
        print(f"Encoding {column}: {df[column].values[0]} -> ", end="")
        df[column] = encoder.transform(df[column])
        print(f"{df[column].values[0]}")

print(df)

# Préparer pour le scaling
if isinstance(feature_names, dict):
    numerical_cols = feature_names.get('numerical_features', [])
    categorical_cols = feature_names.get('categorical_features_encoded', [])
    all_features = numerical_cols + categorical_cols
    print(f"\n=== Ordre des features pour scaling ===")
    print(f"Numerical: {numerical_cols}")
    print(f"Categorical: {categorical_cols}")
    print(f"Total: {len(all_features)} features")
    
    # Réorganiser les colonnes
    df = df[all_features]
    print(f"\nDataFrame réorganisé:")
    print(df)
    
    # Appliquer le scaling
    df_scaled = scaler.transform(df)
    print(f"\n=== Après scaling ===")
    print(df_scaled)
