"""
Script de déploiement local du modèle
Copie le modèle de production depuis MLflow vers le backend et redémarre l'API
"""

import os
import sys
import pickle
import shutil
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PRODUCTION_MODEL_NAME = "churn_prediction_Stacking_LR"
BACKEND_DIR = Path(__file__).parent.parent / "backend"
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
DEPLOYMENT_LOG = Path(__file__).parent.parent / "deployment_log.txt"

print("=" * 80)
print("DÉPLOIEMENT LOCAL DU MODÈLE")
print("=" * 80)

try:
    # 1. Connexion à MLflow
    print(f"\n📊 Connexion à MLflow: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    # 2. Récupérer le modèle de production
    print(f"\n🔍 Recherche du modèle de production: {PRODUCTION_MODEL_NAME}")
    
    try:
        # Chercher les versions en Production
        production_versions = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["Production"])
        
        if not production_versions:
            print(f"⚠️  Aucun modèle en Production trouvé")
            print(f"   Tentative de récupération de la dernière version...")
            
            # Fallback: prendre la dernière version enregistrée
            all_versions = client.search_model_versions(f"name='{PRODUCTION_MODEL_NAME}'")
            if all_versions:
                production_versions = [max(all_versions, key=lambda x: int(x.version))]
            else:
                raise ValueError(f"Aucune version du modèle '{PRODUCTION_MODEL_NAME}' trouvée")
        
        prod_version = production_versions[0]
        prod_run_id = prod_version.run_id
        
        print(f"✅ Modèle trouvé:")
        print(f"   Version: {prod_version.version}")
        print(f"   Run ID: {prod_run_id}")
        print(f"   Stage: {prod_version.current_stage}")
        
        # 3. Charger le modèle depuis MLflow
        print(f"\n📥 Chargement du modèle...")
        model_uri = f"runs:/{prod_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"✅ Modèle chargé: {type(model).__name__}")
        
        # 4. Sauvegarder le modèle dans le dossier notebooks
        print(f"\n💾 Sauvegarde du modèle dans {NOTEBOOKS_DIR}...")
        
        # Créer le dossier si nécessaire
        NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle avec le nom attendu par l'API
        model_filename = "Stacking_LR_ensemble.pkl"
        model_path = NOTEBOOKS_DIR / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # 5. Créer un fichier de métadonnées de déploiement
        deployment_info = {
            "deployed_at": datetime.now().isoformat(),
            "model_name": PRODUCTION_MODEL_NAME,
            "model_version": prod_version.version,
            "run_id": prod_run_id,
            "model_file": model_filename,
            "model_path": str(model_path)
        }
        
        # Sauvegarder les métadonnées
        metadata_path = NOTEBOOKS_DIR / "deployment_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"✅ Métadonnées sauvegardées: {metadata_path}")
        
        # 6. Logger le déploiement
        log_message = f"""
{'='*80}
DÉPLOIEMENT RÉUSSI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
Modèle: {PRODUCTION_MODEL_NAME}
Version: {prod_version.version}
Run ID: {prod_run_id}
Fichier: {model_filename}
Path: {model_path}
{'='*80}
"""
        
        with open(DEPLOYMENT_LOG, 'a') as f:
            f.write(log_message)
        
        print(f"\n✅ Déploiement enregistré dans {DEPLOYMENT_LOG}")
        
        # 7. Instructions pour redémarrer l'API
        print("\n" + "=" * 80)
        print("📋 PROCHAINES ÉTAPES")
        print("=" * 80)
        print("\nPour activer le nouveau modèle:")
        print("1. Arrêter l'API FastAPI si elle est en cours d'exécution")
        print("2. Redémarrer l'API avec:")
        print(f"   cd {BACKEND_DIR}")
        print("   python api.py")
        print("\nOu si vous utilisez uvicorn directement:")
        print("   uvicorn api:app --reload --host 0.0.0.0 --port 8000")
        
        print("\n" + "=" * 80)
        print("✅ DÉPLOIEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la récupération du modèle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Erreur lors du déploiement: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
