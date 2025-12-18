"""
Script de réentraînement automatique
Se déclenche quand un drift est détecté
Charge le modèle de production, réentraîne sur données combinées, compare et promeut si meilleur
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pickle
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Bank_Churn_Retraining"
PREPROCESSOR_PATH = "notebooks/processors/preprocessor.pkl"
INITIAL_TRAINING_DATA = "data/train/part1.csv"
PRODUCTION_DATA = "data/production/bank_churn_prod.csv"

# Seuil d'amélioration pour promotion (2% ROC-AUC)
IMPROVEMENT_THRESHOLD = 0.02

print("=" * 80)
print("RÉENTRAÎNEMENT AUTOMATIQUE DU MODÈLE")
print("=" * 80)

# Configurer MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# ============================================================================
# 1. CHARGER LE MODÈLE DE PRODUCTION
# ============================================================================
print("\n📦 Chargement du modèle de production...")

try:
    # Récupérer le modèle en production
    prod_model_versions = client.get_latest_versions("BankChurnModel", stages=["Production"])
    
    if not prod_model_versions:
        print("⚠️  Aucun modèle en production trouvé")
        print("   Utilisation du dernier modèle enregistré...")
        # Fallback: prendre le dernier modèle
        all_versions = client.search_model_versions("name='BankChurnModel'")
        if not all_versions:
            print("❌ ERREUR: Aucun modèle trouvé dans le registry")
            sys.exit(1)
        prod_model_version = all_versions[0]
    else:
        prod_model_version = prod_model_versions[0]
    
    prod_run_id = prod_model_version.run_id
    prod_model_uri = f"runs:/{prod_run_id}/model"
    prod_model = mlflow.sklearn.load_model(prod_model_uri)
    
    # Récupérer les métriques du modèle de production
    prod_run = client.get_run(prod_run_id)
    prod_metrics = prod_run.data.metrics
    prod_roc_auc = prod_metrics.get('test_roc_auc', 0)
    
    print(f"✅ Modèle de production chargé")
    print(f"   Run ID: {prod_run_id}")
    print(f"   ROC-AUC: {prod_roc_auc:.4f}")
    print(f"   Version: {prod_model_version.version}")
    
except Exception as e:
    print(f"❌ ERREUR lors du chargement du modèle: {e}")
    sys.exit(1)

# ============================================================================
# 2. CHARGER ET COMBINER LES DONNÉES
# ============================================================================
print("\n📂 Chargement des données...")

# Charger les données d'entraînement initiales
df_train = pd.read_csv(INITIAL_TRAINING_DATA)
print(f"✅ Données initiales: {len(df_train):,} lignes")

# Charger les nouvelles données de production
df_prod = pd.read_csv(PRODUCTION_DATA)
print(f"✅ Données production: {len(df_prod):,} lignes")

# Combiner
df_combined = pd.concat([df_train, df_prod], ignore_index=True)
print(f"✅ Données combinées: {len(df_combined):,} lignes")

# ============================================================================
# 3. PRÉPROCESSING
# ============================================================================
print("\n🔧 Préprocessing...")

# Charger le preprocessor
if os.path.exists(PREPROCESSOR_PATH):
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print(f"✅ Preprocessor chargé: {PREPROCESSOR_PATH}")
else:
    print(f"⚠️  Preprocessor non trouvé: {PREPROCESSOR_PATH}")
    print("   Utilisation des données brutes...")
    preprocessor = None

# Séparer features et target
target_col = 'Churn Flag' if 'Churn Flag' in df_combined.columns else 'Churn'
X = df_combined.drop(columns=[target_col])
y = df_combined[target_col]

# Appliquer le preprocessing si disponible
if preprocessor:
    X_processed = preprocessor.transform(X)
else:
    # Preprocessing minimal
    X_processed = X.select_dtypes(include=[np.number]).fillna(0)

print(f"✅ Features: {X_processed.shape}")

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# 4. RÉENTRAÎNEMENT
# ============================================================================
print("\n🔄 Réentraînement du modèle...")

with mlflow.start_run(run_name="Retraining_After_Drift") as run:
    # Récupérer les paramètres du modèle de production
    prod_params = prod_run.data.params
    
    # Log des paramètres
    mlflow.log_params({
        "retraining": True,
        "base_model_run_id": prod_run_id,
        "training_rows": len(df_train),
        "production_rows": len(df_prod),
        "total_rows": len(df_combined)
    })
    
    # Réentraîner avec les mêmes paramètres
    from sklearn.ensemble import RandomForestClassifier
    
    # Utiliser les paramètres du modèle de production ou défauts
    model = RandomForestClassifier(
        n_estimators=int(prod_params.get('n_estimators', 100)),
        max_depth=int(prod_params.get('max_depth', 10)) if prod_params.get('max_depth') != 'None' else None,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ Modèle réentraîné")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log des métriques
    mlflow.log_metrics({
        "test_roc_auc": roc_auc,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall
    })
    
    # Log du modèle
    mlflow.sklearn.log_model(model, "model")
    
    new_run_id = run.info.run_id
    
    print(f"\n📊 Métriques du nouveau modèle:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")

# ============================================================================
# 5. COMPARAISON ET PROMOTION
# ============================================================================
print("\n🔍 Comparaison avec le modèle de production...")

improvement = roc_auc - prod_roc_auc
print(f"   Production ROC-AUC: {prod_roc_auc:.4f}")
print(f"   Nouveau ROC-AUC: {roc_auc:.4f}")
print(f"   Amélioration: {improvement:+.4f} ({improvement/prod_roc_auc*100:+.2f}%)")

if improvement > IMPROVEMENT_THRESHOLD:
    print(f"\n✅ Amélioration significative détectée (>{IMPROVEMENT_THRESHOLD:.2%})")
    print("   Promotion du nouveau modèle en production...")
    
    try:
        # Enregistrer le nouveau modèle dans le registry
        model_uri = f"runs:/{new_run_id}/model"
        mv = mlflow.register_model(model_uri, "BankChurnModel")
        
        # Promouvoir en production
        client.transition_model_version_stage(
            name="BankChurnModel",
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Modèle promu en production (version {mv.version})")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la promotion: {e}")
        print("   Le modèle a été entraîné mais pas promu")
else:
    print(f"\n⚠️  Amélioration insuffisante (<{IMPROVEMENT_THRESHOLD:.2%})")
    print("   Le modèle de production actuel est conservé")

print("\n" + "=" * 80)
print("✅ RÉENTRAÎNEMENT TERMINÉ")
print("=" * 80)
print(f"📊 Nouveau modèle Run ID: {new_run_id}")
print(f"🔗 MLflow UI: http://localhost:5000")
print("=" * 80)
