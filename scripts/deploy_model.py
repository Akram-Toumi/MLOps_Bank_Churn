"""
Deploy model to production-ready location
Prepares the latest MLflow Production model for deployment
"""

import os
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path
import json
from datetime import datetime

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PRODUCTION_MODEL_NAME = "churn_prediction_Stacking_LR"
DEPLOYMENT_DIR = Path("backend/models")
DEPLOYMENT_MODEL_PATH = DEPLOYMENT_DIR / "production_model.pkl"
DEPLOYMENT_METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"

print("=" * 80)
print("MODEL DEPLOYMENT PREPARATION")
print("=" * 80)

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

print(f"\n📊 MLflow URI: {MLFLOW_TRACKING_URI}")

# Create deployment directory
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✅ Deployment directory: {DEPLOYMENT_DIR}")

# Get Production model
try:
    print(f"\n🔍 Searching for Production model: {PRODUCTION_MODEL_NAME}")
    
    production_versions = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["Production"])
    
    if not production_versions:
        print("⚠️  No Production model found")
        exit(1)
    
    prod_version = production_versions[0]
    print(f"✅ Found Production model version: {prod_version.version}")
    
    # Load model
    model_uri = f"models:/{PRODUCTION_MODEL_NAME}/Production"
    print(f"\n📥 Loading model from: {model_uri}")
    
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded: {type(model).__name__}")
    
    # Save model to deployment location
    print(f"\n💾 Saving to: {DEPLOYMENT_MODEL_PATH}")
    with open(DEPLOYMENT_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    # Get model metrics
    run = client.get_run(prod_version.run_id)
    metrics = run.data.metrics
    
    # Create metadata
    metadata = {
        "model_name": PRODUCTION_MODEL_NAME,
        "version": prod_version.version,
        "run_id": prod_version.run_id,
        "deployed_at": datetime.now().isoformat(),
        "metrics": {
            "roc_auc": metrics.get("roc_auc", 0),
            "f1_score": metrics.get("f1_score", 0)
        },
        "model_uri": model_uri
    }
    
    # Save metadata
    with open(DEPLOYMENT_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {DEPLOYMENT_METADATA_PATH}")
    
    print("\n" + "=" * 80)
    print("📊 DEPLOYMENT SUMMARY")
    print("=" * 80)
    print(f"Model: {PRODUCTION_MODEL_NAME}")
    print(f"Version: {prod_version.version}")
    print(f"ROC-AUC: {metadata['metrics']['roc_auc']:.4f}")
    print(f"F1-Score: {metadata['metrics']['f1_score']:.4f}")
    print(f"Location: {DEPLOYMENT_MODEL_PATH}")
    print("=" * 80)
    print("✅ MODEL READY FOR DEPLOYMENT")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
