# Guide d'intégration - Comparaison de modèles

## Objectif
Modifier `churn_predection.ipynb` pour comparer le meilleur des 12 modèles avec le modèle en production avant déploiement.

## Étapes d'intégration

### 1. Ajouter les imports au début du notebook

```python
import pickle
import json
from pathlib import Path
```

### 2. Copier les fonctions depuis `scripts/model_comparison.py`

Copier ces fonctions dans une nouvelle cellule après l'entraînement des 12 modèles :
- `evaluate_model()`
- `compare_models()`
- `save_production_model()`
- `log_to_mlflow()` (optionnel si MLflow déjà configuré)

### 3. Remplacer la cellule finale de déploiement

**AVANT** (logique actuelle - à supprimer) :
```python
# Sauvegarder le meilleur modèle
best_model_name = max(all_models_results, key=lambda x: all_models_results[x]['f1_score'])
best_model = trained_models[best_model_name]

with open(f'models/{best_model_name}_production.pkl', 'wb') as f:
    pickle.dump(best_model, f)
```

**APRÈS** (nouvelle logique - à ajouter) :
```python
# ============================================================================
# COMPARAISON AVEC LE MODÈLE EN PRODUCTION
# ============================================================================

PRODUCTION_MODEL_PATH = "models/production_model.pkl"
PRODUCTION_METRICS_PATH = "models/production_metrics.json"
IMPROVEMENT_THRESHOLD = 0.02  # 2%

print("="*80)
print("COMPARAISON AVEC LE MODÈLE EN PRODUCTION")
print("="*80)

# 1. Meilleur modèle des 12 entraînés
best_model_name = max(all_models_results, key=lambda x: all_models_results[x]['f1_score'])
best_model = trained_models[best_model_name]
best_metrics = all_models_results[best_model_name]

print(f"\\n🏆 Meilleur modèle: {best_model_name}")
print(f"   F1-Score: {best_metrics['f1_score']:.4f}")

# 2. Charger modèle production si existe
if Path(PRODUCTION_MODEL_PATH).exists():
    with open(PRODUCTION_MODEL_PATH, 'rb') as f:
        production_model = pickle.load(f)
    with open(PRODUCTION_METRICS_PATH, 'r') as f:
        production_metrics = json.load(f)
    
    print(f"\\n📦 Modèle actuel: {production_metrics.get('model_name')}")
    print(f"   F1-Score: {production_metrics['f1_score']:.4f}")
    
    # 3. Comparer
    decision = compare_models(best_metrics, production_metrics, IMPROVEMENT_THRESHOLD)
    
    print(f"\\n{'='*80}")
    print(f"Déployer: {'✅ OUI' if decision['deploy'] else '❌ NON'}")
    print(f"Raison: {decision['reason']}")
    
    # 4. Déployer si amélioration
    if decision['deploy']:
        save_production_model(best_model, best_metrics, best_model_name)
        print("🚀 Nouveau modèle déployé")
    else:
        print("⏸️  Modèle actuel conservé")
else:
    # Premier déploiement
    print("\\n⚠️  Premier déploiement")
    save_production_model(best_model, best_metrics, best_model_name)
```

### 4. Créer le dossier models

```python
Path("models").mkdir(exist_ok=True)
```

## Variables nécessaires

Assurez-vous que votre notebook a ces variables :
- `all_models_results` : dict avec les métriques de tous les modèles
- `trained_models` : dict avec les modèles entraînés
- `X_test`, `y_test` : données de test

## Exemple de structure `all_models_results`

```python
all_models_results = {
    'XGBoost_tuned': {
        'accuracy': 0.85,
        'f1_score': 0.82,
        'roc_auc': 0.88
    },
    'Random_Forest_tuned': {
        'accuracy': 0.83,
        'f1_score': 0.80,
        'roc_auc': 0.86
    },
    # ... autres modèles
}
```

## Test

Après intégration, exécuter le notebook devrait :
1. ✅ Entraîner les 12 modèles
2. ✅ Identifier le meilleur
3. ✅ Comparer avec le modèle prod (si existe)
4. ✅ Déployer seulement si amélioration > 2%
5. ✅ Sauvegarder dans `models/production_model.pkl`

## Fichiers créés

- `models/production_model.pkl` : Modèle en production
- `models/production_metrics.json` : Métriques du modèle prod
