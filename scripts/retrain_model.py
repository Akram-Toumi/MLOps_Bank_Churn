"""
Script de réentraînement complet
Preprocess → Combine → Train all models → Compare → Promote
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
from pathlib import Path
import shutil
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform

# Import preprocessing
sys.path.append('scripts')
from preprocess_production import ProductionDataPreprocessor

# Configuration
# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "Bank_Churn_Retraining"
PRODUCTION_MODEL_NAME = "churn_prediction_Stacking_LR"
INITIAL_TRAINING_DATA = "notebooks/processors/preprocessed_data.pkl"
PRODUCTION_DATA_RAW = "data/production/bank_churn_prod.csv"
PROCESSOR_DIR = "notebooks/processors"
MODEL_REGISTRY_DIR = Path("notebooks/processors/model_registry")
IMPROVEMENT_THRESHOLD = 0.02  # 2% ROC-AUC improvement required
N_ITER = 5  # Reduced for faster retraining
CV_FOLDS = 3


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def register_to_mlflow(model, model_name, best_row, best_run_id, best_roc_auc, should_replace, current_model_version=None):
    """Register model to MLflow Model Registry (Logic accepted from binome)"""
    
    client = mlflow.tracking.MlflowClient()
    
    if should_replace:
        print(f"\n🔄 Registering new model to MLflow...")
        
        try:
            # Note: Model is already logged in 'best_run_id', we just need to register it from there
            model_uri = f"runs:/{best_run_id}/model"
            
            print(f"✅ Registering model from URI: {model_uri}")
            mv = mlflow.register_model(model_uri, model_name)
            
            # Wait for registration? usually returns version immediately
            new_version = mv.version
            print(f"   New version: {new_version}")
            
            # Archive old version in Production (if exists)
            if current_model_version:
                print(f"\n📦 Archiving old version...")
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=current_model_version.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
                    print(f"✅ Version {current_model_version.version} archived")
                except Exception as e:
                    print(f"⚠️  Could not archive old version: {e}")
            
            # Transition new version to Production
            print(f"\n🚀 Transitioning to Production...")
            client.transition_model_version_stage(
                name=model_name,
                version=new_version,
                stage="Production",
                archive_existing_versions=True
            )
            
            print(f"✅ Model promoted to Production")
            print(f"   Stage: Production")
            print(f"   Version: {new_version}")
            
            # Add metadata
            try:
                improvement_text = ""
                if current_model_version:
                    current_run = client.get_run(current_model_version.run_id)
                    current_roc_auc = current_run.data.metrics.get('roc_auc', 0)
                    roc_auc_diff = best_roc_auc - current_roc_auc
                    improvement_pct = (roc_auc_diff / current_roc_auc) * 100 if current_roc_auc > 0 else 0
                    improvement_text = f"\\n- Improvement over v{current_model_version.version}: +{roc_auc_diff:.4f} ({improvement_pct:+.2f}%)"
                
                client.update_model_version(
                    name=model_name,
                    version=new_version,
                    description=f"Best performing churn prediction model\\n- Model: {best_row['model']}\\n- Stage: {best_row['stage']}\\n- ROC-AUC: {best_roc_auc:.4f}\\n- F1-Score: {best_row['f1']:.4f}\\n- Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{improvement_text}"
                )
                
                # Add tags
                client.set_model_version_tag(
                    name=model_name,
                    version=new_version,
                    key="model_type",
                    value=best_row['model']
                )
                
                client.set_model_version_tag(
                    name=model_name,
                    version=new_version,
                    key="roc_auc",
                    value=str(round(best_roc_auc, 4))
                )
                
                print(f"✅ Metadata added (description + tags)")
            except Exception as e:
                print(f"⚠️  Could not add metadata: {e}")
                
            return new_version
                
        except Exception as e:
            print(f"⚠️  Error during registration: {e}")
            return None
        return None
    else:
        print(f"\n⏭️  New model not registered to Production")
        return None

def register_model_local(model, model_name, version="1.0.0", stage="production", metrics=None, run_id=None):
    """Register a model in the local registry (filesystem)"""
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create structure
    model_dir = MODEL_REGISTRY_DIR / model_name.replace(" ", "_")
    model_dir.mkdir(exist_ok=True)
    
    version_dir = model_dir / version
    version_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = version_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics or {},
        "run_id": run_id
    }
    
    with open(version_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Production link
    if stage == "production":
        prod_path = model_dir / "production.pkl"
        shutil.copy(model_path, prod_path)
    
    return str(model_path)

print("=" * 80)
print("RÉENTRAÎNEMENT AUTOMATIQUE - PIPELINE COMPLET")
print("=" * 80)

# ============================================================================
# 1. PREPROCESSING DES NOUVELLES DONNÉES
# ============================================================================
print("\n📊 ÉTAPE 1: Preprocessing des données de production")
print("=" * 80)

preprocessor = ProductionDataPreprocessor()
preprocessor.load_processors(PROCESSOR_DIR)
df_prod_processed = preprocessor.preprocess(pd.read_csv(PRODUCTION_DATA_RAW))

print(f"✅ Données preprocessées: {df_prod_processed.shape}")

# ============================================================================
# 2. CHARGEMENT ET COMBINAISON DES DONNÉES
# ============================================================================
print("\n📊 ÉTAPE 2: Combinaison avec données d'entraînement")
print("=" * 80)

# Charger les données initiales (SANS SMOTE)
with open(INITIAL_TRAINING_DATA, 'rb') as f:
    initial_data = pickle.load(f)

# Utiliser X_test et y_test (données non-SMOTE) au lieu de X_train
X_train_initial = initial_data['X_test']  # Données de test = non-SMOTE
y_train_initial = initial_data['y_test']

print(f"Données initiales (non-SMOTE): {X_train_initial.shape}")
print(f"Nouvelles données: {df_prod_processed.shape}")

# Séparer target des nouvelles données si présente
if 'Churn Flag' in df_prod_processed.columns:
    X_prod = df_prod_processed.drop('Churn Flag', axis=1)
    y_prod = df_prod_processed['Churn Flag']
    print(f"✅ Target trouvée dans production data")
else:
    print("⚠️  Pas de target dans les données de production")
    print("   Utilisation uniquement des données initiales pour réentraînement")
    X_prod = None
    y_prod = None

# Aligner les colonnes
common_cols = list(set(X_train_initial.columns) & set(X_prod.columns if X_prod is not None else X_train_initial.columns))
X_train_initial = X_train_initial[common_cols]

# Combiner si on a des données de production avec target
if X_prod is not None and y_prod is not None:
    X_prod = X_prod[common_cols]
    X_combined = pd.concat([X_train_initial, X_prod], ignore_index=True)
    y_combined = pd.concat([y_train_initial, y_prod], ignore_index=True)
    print(f"✅ Données combinées: {X_combined.shape}")
else:
    X_combined = X_train_initial
    y_combined = y_train_initial
    print(f"✅ Utilisation données initiales uniquement: {X_combined.shape}")

print(f"   Churn rate: {y_combined.mean():.2%}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# ============================================================================
# 3. CONFIGURATION MLflow
# ============================================================================
print("\n📊 ÉTAPE 3: Configuration MLflow")
print("=" * 80)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print(f"✅ MLflow configuré")
print(f"   Experiment: {EXPERIMENT_NAME}")

# ============================================================================
# 4. RÉCUPÉRATION DU MODÈLE DE PRODUCTION
# ============================================================================
print("\n📊 ÉTAPE 4: Chargement modèle de production")
print("=" * 80)

try:
    # Chercher le modèle par nom
    registered_models = client.search_registered_models(f"name='{PRODUCTION_MODEL_NAME}'")
    
    if registered_models:
        model_versions = client.search_model_versions(f"name='{PRODUCTION_MODEL_NAME}'")
        if model_versions:
            latest_version = max(model_versions, key=lambda x: int(x.version))
            prod_run_id = latest_version.run_id
            prod_model_uri = f"runs:/{prod_run_id}/model"
            
            # Charger le modèle
            prod_model = mlflow.sklearn.load_model(prod_model_uri)
            
            # Évaluer le modèle sur les données ACTUELLES (Fair Comparison)
            print(f"   Évaluation sur le test set actuel...")
            y_prod_proba = prod_model.predict_proba(X_test)[:, 1]
            prod_roc_auc = roc_auc_score(y_test, y_prod_proba)
            
            # Récupérer les métriques historiques (juste pour info)
            prod_run = client.get_run(prod_run_id)
            prod_metrics = prod_run.data.metrics
            prod_historical_auc = prod_metrics.get('roc_auc', prod_metrics.get('test_roc_auc', 0))
            
            print(f"✅ Modèle de production chargé")
            print(f"   Nom: {PRODUCTION_MODEL_NAME}")
            print(f"   Version: {latest_version.version}")
            print(f"   ROC-AUC (Historique): {prod_historical_auc:.4f}")
            print(f"   ROC-AUC (Actuel): {prod_roc_auc:.4f}")
        else:
            print("⚠️  Aucune version trouvée")
            prod_roc_auc = 0.5  # Baseline aléatoire

    else:
        print(f"⚠️  Modèle '{PRODUCTION_MODEL_NAME}' non trouvé")
        prod_roc_auc = 0.5
except Exception as e:
    print(f"⚠️  Erreur: {e}")
    prod_roc_auc = 0.5

# ============================================================================
# 5. ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n📊 ÉTAPE 5: Entraînement des modèles")
print("=" * 80)

# Configurations baseline
baseline_models = {
    'XGBoost': XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.05, random_state=42, eval_metric='auc', use_label_encoder=False),
    'LightGBM': LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=150, depth=7, learning_rate=0.05, random_state=42, verbose=False),
    'LogisticRegression': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=42, n_jobs=-1)
}

# Search spaces pour tuning
search_spaces = {
    'XGBoost': {'n_estimators': randint(100, 300), 'max_depth': randint(3, 10), 'learning_rate': uniform(0.01, 0.19)},
    'LightGBM': {'n_estimators': randint(100, 300), 'max_depth': randint(5, 12), 'learning_rate': uniform(0.01, 0.19)},
    'RandomForest': {'n_estimators': randint(100, 300), 'max_depth': [15, 20, 25, 30], 'min_samples_split': randint(2, 15)},
    'CatBoost': {'iterations': randint(100, 300), 'depth': randint(4, 10), 'learning_rate': uniform(0.01, 0.19)},
    'LogisticRegression': {'C': uniform(0.01, 5), 'l1_ratio': uniform(0, 1)}
}

all_results = []
trained_models = {}

# Train baseline
print("\n🚀 Baseline models...")
for name, model in baseline_models.items():
    print(f"  {name}...", end=" ")
    start = datetime.now()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    duration = (datetime.now() - start).total_seconds()
    
    with mlflow.start_run(run_name=f"{name}_baseline_retrain"):
        mlflow.log_params({'model': name, 'stage': 'baseline', 'retrain': True})
        mlflow.log_metrics({'roc_auc': roc_auc, 'f1_score': f1, 'duration': duration})
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
    
    trained_models[f"{name}_baseline"] = model
    all_results.append({'model': name, 'stage': 'baseline', 'roc_auc': roc_auc, 'f1': f1, 'run_id': run_id})
    
    print(f"ROC-AUC: {roc_auc:.4f}")

# Train finetuned
print("\n🔍 Finetuned models...")
for name, base_model in baseline_models.items():
    print(f"  {name}...", end=" ")
    start = datetime.now()
    
    search = RandomizedSearchCV(base_model, search_spaces[name], n_iter=N_ITER, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    duration = (datetime.now() - start).total_seconds()
    
    with mlflow.start_run(run_name=f"{name}_finetuned_retrain"):
        mlflow.log_params({'model': name, 'stage': 'finetuned', 'retrain': True, **search.best_params_})
        mlflow.log_metrics({'roc_auc': roc_auc, 'f1_score': f1, 'duration': duration})
        mlflow.sklearn.log_model(best_model, "model")
        run_id = mlflow.active_run().info.run_id
    
    trained_models[f"{name}_finetuned"] = best_model
    all_results.append({'model': name, 'stage': 'finetuned', 'roc_auc': roc_auc, 'f1': f1, 'run_id': run_id})
    
    print(f"ROC-AUC: {roc_auc:.4f}")

# Train Stacking
print("\n🚀 Stacking ensemble...")
estimators = [
    ('lgbm', trained_models['LightGBM_finetuned']),
    ('rf', trained_models['RandomForest_finetuned']),
    ('cat', trained_models['CatBoost_finetuned']),
    ('lr', trained_models['LogisticRegression_finetuned'])
]

meta_learner = LogisticRegression(penalty='elasticnet', C=0.5, l1_ratio=0.3, solver='saga', max_iter=1500, random_state=42, n_jobs=-1)
stacking = StackingClassifier(estimators=estimators, final_estimator=meta_learner, cv=3, stack_method='predict_proba', n_jobs=-1)

start = datetime.now()
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
y_proba = stacking.predict_proba(X_test)[:, 1]

roc_auc_stack = roc_auc_score(y_test, y_proba)
f1_stack = f1_score(y_test, y_pred)
duration = (datetime.now() - start).total_seconds()

with mlflow.start_run(run_name="Stacking_LR_retrain"):
    mlflow.log_params({'model': 'Stacking_LR', 'stage': 'ensemble', 'retrain': True})
    mlflow.log_metrics({'roc_auc': roc_auc_stack, 'f1_score': f1_stack, 'duration': duration})
    mlflow.sklearn.log_model(stacking, "model")
    stack_run_id = mlflow.active_run().info.run_id

trained_models['Stacking_LR_ensemble'] = stacking
all_results.append({'model': 'Stacking_LR', 'stage': 'ensemble', 'roc_auc': roc_auc_stack, 'f1': f1_stack, 'run_id': stack_run_id})

print(f"  Stacking_LR: ROC-AUC: {roc_auc_stack:.4f}")

# ============================================================================
# 6. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================================
print("\n📊 ÉTAPE 6: Sélection du meilleur modèle")
print("=" * 80)

df_results = pd.DataFrame(all_results)
best_idx = df_results['roc_auc'].idxmax()
best_result = df_results.loc[best_idx]

print(f"\n🏆 Meilleur modèle:")
print(f"   Nom: {best_result['model']} ({best_result['stage']})")
print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"   F1-Score: {best_result['f1']:.4f}")

# ============================================================================
# 7. COMPARAISON ET PROMOTION
# ============================================================================
print("\n📊 ÉTAPE 7: Comparaison avec production")
print("=" * 80)

improvement = best_result['roc_auc'] - prod_roc_auc
improvement = best_result['roc_auc'] - prod_roc_auc
print(f"   Production ROC-AUC (Sur Test Set): {prod_roc_auc:.4f}")
print(f"   Nouveau ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"   Amélioration: {improvement:+.4f} ({improvement/max(prod_roc_auc, 0.01)*100:+.2f}%)")

if improvement > IMPROVEMENT_THRESHOLD:
    print(f"\n✅ Amélioration significative (>{IMPROVEMENT_THRESHOLD:.2%})")
    print("   Promotion du nouveau modèle...")
    
    # Identify current model version object if it exists
    current_version_obj = None
    if prod_roc_auc > 0: # Means we found a production model
         try:
            versions = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["Production"])
            if versions:
                current_version_obj = versions[0]
         except:
            pass
    
    # Use the new robust registration function
    # Note: best_result has 'f1' key, but script used 'f1_score'. My dict has 'f1'.
    
    # Load best model from memory to pass if needed, but we use URI from run_id
    best_model_obj = trained_models[f"{best_result['model']}_{best_result['stage']}"]
    
    register_to_mlflow(
        model=best_model_obj,
        model_name=PRODUCTION_MODEL_NAME,
        best_row=best_result,
        best_run_id=best_result['run_id'],
        best_roc_auc=best_result['roc_auc'],
        should_replace=True,
        current_model_version=current_version_obj
    )

    print(f"✅ Modèle enregistré et promu dans MLflow (version Production)")

    # ---------------------------------------------------------
    # REGISTER TO LOCAL REGISTRY
    # ---------------------------------------------------------
    
    # Calculate metrics for local registry format
    # Note: best_result uses keys 'roc_auc', 'f1', need to map if necessary or pass as is
    local_metrics = {
        'roc_auc': best_result['roc_auc'],
        'f1_score': best_result['f1']
    }
    
    registry_name = f"Best_Churn_{best_result['model']}"
    print(f"\n📦 Enregistrement dans le registre local...")
    
    local_path = register_model_local(
        model=best_model_obj,
        model_name=registry_name,
        version="1.0.0", # Simplified versioning for local, or could implement increment logic
        stage="production",
        metrics=local_metrics,
        run_id=best_result['run_id']
    )
    
    print(f"✅ Modèle sauvegardé localement: {local_path}")

else:
    print(f"\n⚠️  Amélioration insuffisante (<{IMPROVEMENT_THRESHOLD:.2%})")
    print("   Modèle de production conservé")

print("\n" + "=" * 80)
print("✅ RÉENTRAÎNEMENT TERMINÉ")
print("=" * 80)
print(f"🔗 MLflow UI: http://localhost:5000")
