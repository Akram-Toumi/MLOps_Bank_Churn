"""
Script de comparaison de modèles pour MLOps
À intégrer dans churn_predection.ipynb à la fin du notebook

Ce script:
1. Charge le meilleur modèle des 12 entraînés
2. Charge le modèle actuellement en production (depuis MLflow ou fichier)
3. Compare leurs performances
4. Décide si on déploie le nouveau modèle
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

PRODUCTION_MODEL_PATH = "models/production_model.pkl"  # Modèle actuellement en prod
PRODUCTION_METRICS_PATH = "models/production_metrics.json"  # Métriques du modèle prod
IMPROVEMENT_THRESHOLD = 0.02  # 2% d'amélioration minimum pour déployer

# ============================================================================
# FONCTION: Évaluer un modèle
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Évalue un modèle et retourne ses métriques"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

# ============================================================================
# FONCTION: Comparer deux modèles
# ============================================================================

def compare_models(new_metrics, prod_metrics, threshold=0.02):
    """
    Compare les métriques de deux modèles
    
    Returns:
        dict: {
            'deploy': bool,
            'reason': str,
            'improvements': dict
        }
    """
    improvements = {
        'accuracy': new_metrics['accuracy'] - prod_metrics['accuracy'],
        'f1_score': new_metrics['f1_score'] - prod_metrics['f1_score'],
        'roc_auc': new_metrics['roc_auc'] - prod_metrics['roc_auc']
    }
    
    # Décision basée sur F1-score (métrique principale)
    f1_improvement = improvements['f1_score']
    
    if f1_improvement > threshold:
        decision = {
            'deploy': True,
            'reason': f'Amélioration significative du F1-score: +{f1_improvement:.2%}',
            'improvements': improvements
        }
    elif f1_improvement > 0:
        decision = {
            'deploy': False,
            'reason': f'Amélioration trop faible ({f1_improvement:.2%} < {threshold:.2%})',
            'improvements': improvements
        }
    else:
        decision = {
            'deploy': False,
            'reason': f'Dégradation des performances: {f1_improvement:.2%}',
            'improvements': improvements
        }
    
    return decision

# ============================================================================
# FONCTION: Sauvegarder le modèle en production
# ============================================================================

def save_production_model(model, metrics, model_name):
    """Sauvegarde le modèle comme nouveau modèle de production"""
    # Créer le dossier models s'il n'existe pas
    Path("models").mkdir(exist_ok=True)
    
    # Sauvegarder le modèle
    with open(PRODUCTION_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    # Sauvegarder les métriques
    metrics_to_save = {
        **metrics,
        'model_name': model_name,
        'deployment_date': pd.Timestamp.now().isoformat()
    }
    
    with open(PRODUCTION_METRICS_PATH, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"✅ Modèle {model_name} sauvegardé en production")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

# ============================================================================
# FONCTION: Logger dans MLflow
# ============================================================================

def log_to_mlflow(model, metrics, model_name, tags=None):
    """Log le modèle et ses métriques dans MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Log des métriques
        mlflow.log_metrics(metrics)
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "model")
        
        # Log des tags
        if tags:
            mlflow.set_tags(tags)
        
        print(f"✅ Modèle {model_name} loggé dans MLflow")

# ============================================================================
# SCRIPT PRINCIPAL À INTÉGRER DANS LE NOTEBOOK
# ============================================================================

"""
# ============================================================================
# ÉTAPE FINALE: COMPARAISON AVEC LE MODÈLE EN PRODUCTION
# ============================================================================

print("="*80)
print("COMPARAISON AVEC LE MODÈLE EN PRODUCTION")
print("="*80)

# 1. Identifier le meilleur modèle des 12 entraînés
# (Supposons que vous avez déjà un dictionnaire 'all_models_results' avec les résultats)

best_model_name = max(all_models_results, key=lambda x: all_models_results[x]['f1_score'])
best_model = trained_models[best_model_name]  # Votre modèle entraîné
best_metrics = all_models_results[best_model_name]

print(f"\\n🏆 Meilleur modèle entraîné: {best_model_name}")
print(f"   F1-Score: {best_metrics['f1_score']:.4f}")

# 2. Charger le modèle actuellement en production
production_exists = Path(PRODUCTION_MODEL_PATH).exists()

if production_exists:
    print(f"\\n📦 Chargement du modèle en production...")
    
    with open(PRODUCTION_MODEL_PATH, 'rb') as f:
        production_model = pickle.load(f)
    
    with open(PRODUCTION_METRICS_PATH, 'r') as f:
        production_metrics = json.load(f)
    
    print(f"   Modèle actuel: {production_metrics.get('model_name', 'Unknown')}")
    print(f"   F1-Score: {production_metrics['f1_score']:.4f}")
    
    # 3. Comparer les modèles
    print(f"\\n🔍 Comparaison des performances...")
    decision = compare_models(best_metrics, production_metrics, IMPROVEMENT_THRESHOLD)
    
    print(f"\\n{'='*80}")
    print("DÉCISION DE DÉPLOIEMENT")
    print(f"{'='*80}")
    print(f"\\nDéployer le nouveau modèle: {'✅ OUI' if decision['deploy'] else '❌ NON'}")
    print(f"Raison: {decision['reason']}")
    print(f"\\nAméliorations:")
    for metric, improvement in decision['improvements'].items():
        sign = '+' if improvement > 0 else ''
        print(f"  • {metric}: {sign}{improvement:.2%}")
    
    # 4. Déployer si décision positive
    if decision['deploy']:
        print(f"\\n🚀 Déploiement du nouveau modèle...")
        save_production_model(best_model, best_metrics, best_model_name)
        log_to_mlflow(best_model, best_metrics, best_model_name, 
                     tags={'status': 'production', 'replaced': production_metrics.get('model_name')})
    else:
        print(f"\\n⏸️  Conservation du modèle actuel en production")
        print(f"   Le nouveau modèle n'apporte pas d'amélioration suffisante")

else:
    # Pas de modèle en production, déployer directement
    print(f"\\n⚠️  Aucun modèle en production détecté")
    print(f"   Déploiement automatique du meilleur modèle...")
    save_production_model(best_model, best_metrics, best_model_name)
    log_to_mlflow(best_model, best_metrics, best_model_name, 
                 tags={'status': 'production', 'first_deployment': True})

print(f"\\n{'='*80}")
print("✅ PROCESSUS TERMINÉ")
print(f"{'='*80}")
"""
