"""
Register best model to MLflow Model Registry
Compares with existing production model and decides whether to replace
"""

import pandas as pd
import pickle
import os
import json
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "churn_prediction"
RESULTS_PATH = 'training_results.pkl'
MODEL_REGISTRY_DIR = Path("../notebooks/processors/model_registry")
MINIMUM_IMPROVEMENT = 0.001  # Minimum improvement required (0.1%)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }


def register_model_local(model, model_name, version="1.0.0", stage="production", metrics=None, run_id=None):
    """Register a model in the local registry"""
    MODEL_REGISTRY_DIR.mkdir(exist_ok=True)
    
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
        import shutil
        prod_path = model_dir / "production.pkl"
        shutil.copy(model_path, prod_path)
    
    return str(model_path)


def load_from_registry(model_name, stage="production"):
    """Load a model from the local registry"""
    model_dir = MODEL_REGISTRY_DIR / model_name.replace(" ", "_")
    model_path = model_dir / f"{stage}.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    versions = [d for d in model_dir.iterdir() if d.is_dir()]
    if versions:
        latest_version = sorted(versions)[-1]
        with open(latest_version / "metadata.json", 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, metadata


# ============================================================================
# MLFLOW REGISTRATION
# ============================================================================

def register_to_mlflow(model, model_name, best_row, best_run_id, best_roc_auc, should_replace, current_model_version=None):
    """Register model to MLflow Model Registry"""
    
    client = mlflow.tracking.MlflowClient()
    
    if should_replace:
        print(f"\n🔄 Registering new model to MLflow...")
        
        try:
            # Log the model with mlflow.sklearn
            with mlflow.start_run(run_id=best_run_id):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
            
            print(f"✅ Model registered to MLflow Model Registry")
            
            # Get new version
            new_versions = client.get_latest_versions(model_name, stages=["None"])
            if new_versions:
                new_version = new_versions[0].version
                print(f"   New version: {new_version}")
                
                # Archive old version in Production (if exists)
                if current_model_version:
                    print(f"\n📦 Archiving old version...")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=current_model_version.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
                    print(f"✅ Version {current_model_version.version} archived")
                
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
                improvement_text = ""
                if current_model_version:
                    current_run = client.get_run(current_model_version.run_id)
                    current_roc_auc = current_run.data.metrics.get('roc_auc', 0)
                    roc_auc_diff = best_roc_auc - current_roc_auc
                    improvement_pct = (roc_auc_diff / current_roc_auc) * 100 if current_roc_auc > 0 else 0
                    improvement_text = f"\n- Improvement over v{current_model_version.version}: +{roc_auc_diff:.4f} ({improvement_pct:+.2f}%)"
                
                client.update_model_version(
                    name=model_name,
                    version=new_version,
                    description=f"""
Best performing churn prediction model
- Model: {best_row['model']}
- Stage: {best_row['stage']}
- ROC-AUC: {best_roc_auc:.4f}
- F1-Score: {best_row['f1_score']:.4f}
- Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{improvement_text}
                    """
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
                    key="training_stage",
                    value=best_row['stage']
                )
                
                client.set_model_version_tag(
                    name=model_name,
                    version=new_version,
                    key="roc_auc",
                    value=str(round(best_roc_auc, 4))
                )
                
                if current_model_version:
                    client.set_model_version_tag(
                        name=model_name,
                        version=new_version,
                        key="replaced_version",
                        value=str(current_model_version.version)
                    )
                    
                    client.set_model_version_tag(
                        name=model_name,
                        version=new_version,
                        key="improvement_pct",
                        value=f"{improvement_pct:.2f}"
                    )
                
                print(f"✅ Metadata added (description + tags)")
                return new_version
                
        except Exception as e:
            print(f"⚠️  Error during registration: {e}")
            print(f"   Model remains available via run_id: {best_run_id}")
            return None
    else:
        print(f"\n⏭️  New model not registered to Production")
        print(f"   Current model (v{current_model_version.version}) remains in place")
        
        # Optionally register to Staging for reference
        try:
            print(f"\n📝 Registering to Staging for reference...")
            with mlflow.start_run(run_id=best_run_id):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
            
            staging_versions = client.get_latest_versions(model_name, stages=["None"])
            if staging_versions:
                staging_version = staging_versions[0].version
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=staging_version,
                    stage="Staging",
                    archive_existing_versions=False
                )
                
                current_run = client.get_run(current_model_version.run_id)
                current_roc_auc = current_run.data.metrics.get('roc_auc', 0)
                roc_auc_diff = best_roc_auc - current_roc_auc
                improvement_pct = (roc_auc_diff / current_roc_auc) * 100 if current_roc_auc > 0 else 0
                
                client.update_model_version(
                    name=model_name,
                    version=staging_version,
                    description=f"""
Candidate model (not deployed)
- Model: {best_row['model']}
- ROC-AUC: {best_roc_auc:.4f}
- Rejected: Insufficient improvement over production model
- Difference: {roc_auc_diff:.4f} ({improvement_pct:.2f}%)
                    """
                )
                
                print(f"✅ Model registered to Staging (v{staging_version})")
                return staging_version
        except Exception as e:
            print(f"⚠️  Error registering to Staging: {e}")
            return None


# ============================================================================
# MAIN REGISTRATION PIPELINE
# ============================================================================

def main():
    """Main model registration pipeline"""
    
    print("="*80)
    print("REGISTER BEST MODEL TO MLFLOW MODEL REGISTRY")
    print("="*80)
    
    # 1. Setup MLflow
    print("\n📊 Setting up MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow configured")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # 2. Load training results
    print(f"\n📂 Loading training results...")
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ Error: {RESULTS_PATH} not found!")
        print("   Please run train.py first")
        return
    
    with open(RESULTS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    df_results = data['df_results']
    trained_models = data['trained_models']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"✅ Results loaded")
    print(f"   Total models: {len(df_results)}")
    
    # 3. Identify best model
    print("\n🔍 Identifying best model...")
    best_idx = df_results['roc_auc'].idxmax()
    best_row = df_results.loc[best_idx]
    
    best_model_name = best_row['model']
    best_stage = best_row['stage']
    best_run_id = best_row['run_id']
    best_roc_auc = best_row['roc_auc']
    
    print("\n🏆 BEST MODEL (ROC-AUC)")
    print("="*60)
    print(f"Model:     {best_model_name}")
    print(f"Stage:     {best_stage}")
    print(f"ROC-AUC:   {best_roc_auc:.4f}")
    print(f"F1-Score:  {best_row['f1_score']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall:    {best_row['recall']:.4f}")
    print(f"Run ID:    {best_run_id}")
    print("="*60)
    
    # 4. Load best model
    best_model_key = f"{best_model_name}_{best_stage}" if best_stage != 'ensemble' else best_model_name
    best_model = trained_models.get(best_model_key)
    
    if best_model is None:
        print(f"❌ Error: Could not find model {best_model_key}")
        return
    
    print(f"\n✅ Model loaded: {type(best_model).__name__}")
    
    # 5. Test loaded model
    print("\n🧪 Testing model...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    print("📊 Model performance:")
    for metric, value in test_metrics.items():
        print(f"   {metric:12s}: {value:.4f}")
    
    # 6. Check existing production model
    model_name = f"churn_prediction_{best_model_name}"
    client = mlflow.tracking.MlflowClient()
    
    print(f"\n🔍 Checking existing model: {model_name}")
    print(f"   New model - Run ID: {best_run_id}")
    print(f"   New model - ROC-AUC: {best_roc_auc:.4f}")
    
    try:
        # Search for Production versions
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if production_versions:
            # Production model exists
            current_model_version = production_versions[0]
            current_run_id = current_model_version.run_id
            
            print(f"\n✅ Production model found")
            print(f"   Current version: {current_model_version.version}")
            print(f"   Run ID: {current_run_id}")
            
            # Get current model metrics
            current_run = client.get_run(current_run_id)
            current_roc_auc = current_run.data.metrics.get('roc_auc', 0)
            
            print(f"   ROC-AUC: {current_roc_auc:.4f}")
            
            # COMPARISON: New model vs Current model
            print(f"\n⚖️  MODEL COMPARISON")
            print(f"   {'Metric':<20} {'Current':<15} {'New':<15} {'Difference':<15}")
            print(f"   {'-'*65}")
            
            roc_auc_diff = best_roc_auc - current_roc_auc
            improvement_pct = (roc_auc_diff / current_roc_auc) * 100 if current_roc_auc > 0 else 0
            
            print(f"   {'ROC-AUC':<20} {current_roc_auc:<15.4f} {best_roc_auc:<15.4f} {roc_auc_diff:+.4f} ({improvement_pct:+.2f}%)")
            
            # Decision: Replace or not
            if best_roc_auc > current_roc_auc + MINIMUM_IMPROVEMENT:
                print(f"\n✅ DECISION: REPLACEMENT")
                print(f"   New model is better (improvement > {MINIMUM_IMPROVEMENT:.4f})")
                should_replace = True
            else:
                print(f"\n❌ DECISION: KEEP CURRENT")
                print(f"   Current model remains in production")
                print(f"   Insufficient improvement (threshold: {MINIMUM_IMPROVEMENT:.4f})")
                should_replace = False
        else:
            # No Production model
            print(f"\n⚠️  No Production model found")
            print(f"   New model will be automatically promoted to Production")
            should_replace = True
            current_model_version = None
    
    except Exception as e:
        # Model not in registry
        print(f"\n⚠️  Model not found in Registry")
        print(f"   First registration - will be promoted to Production")
        should_replace = True
        current_model_version = None
    
    # 7. Register to MLflow
    print("\n" + "="*80)
    print("📦 MLflow Model Registry Registration")
    print("="*80)
    
    new_version = register_to_mlflow(
        best_model, model_name, best_row, best_run_id, 
        best_roc_auc, should_replace, current_model_version
    )
    
    # 8. Register to local registry
    print("\n" + "="*80)
    print("📦 Local Model Registry")
    print("="*80)
    
    registry_name = f"Best_Churn_{best_model_name}"
    print(f"\n💾 Saving to local registry...")
    
    model_path = register_model_local(
        model=best_model,
        model_name=registry_name,
        version="1.0.0",
        stage="production",
        metrics=test_metrics,
        run_id=best_run_id
    )
    
    print(f"✅ Model saved to local registry")
    print(f"   Name: {registry_name}")
    print(f"   Version: 1.0.0")
    print(f"   Stage: production")
    print(f"   Path: {model_path}")
    
    # 9. Test loading from local registry
    print(f"\n🔄 Testing local registry...")
    loaded_from_registry, metadata = load_from_registry(registry_name, stage="production")
    
    test_pred = loaded_from_registry.predict(X_test[:5])
    print(f"✅ Model loaded successfully from local registry")
    print(f"   Test predictions: {test_pred}")
    
    # 10. Display all versions
    print("\n" + "="*80)
    print("📋 Model Version History")
    print("="*80)
    
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        if all_versions:
            print(f"\n{'Version':<10} {'Stage':<15} {'ROC-AUC':<12} {'Created':<20}")
            print("-" * 60)
            
            for mv in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
                # Get ROC-AUC metric
                try:
                    run = client.get_run(mv.run_id)
                    roc_auc = run.data.metrics.get('roc_auc', 0)
                    roc_auc_str = f"{roc_auc:.4f}"
                except:
                    roc_auc_str = "N/A"
                
                created = datetime.fromtimestamp(mv.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M')
                stage = mv.current_stage
                
                # Mark production version
                marker = "🏆 " if stage == "Production" else "   "
                
                print(f"{marker}{mv.version:<10} {stage:<15} {roc_auc_str:<12} {created:<20}")
            
            print(f"\n✅ {len(all_versions)} version(s) found")
        else:
            print("⚠️  No versions found")
            
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    # 11. Final summary
    print("\n" + "="*80)
    print("📊 REGISTRATION SUMMARY")
    print("="*80)
    
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0]
            prod_run = client.get_run(prod_version.run_id)
            prod_roc_auc = prod_run.data.metrics.get('roc_auc', 0)
            
            print(f"\n🏆 Production Model:")
            print(f"   Name: {model_name}")
            print(f"   Version: {prod_version.version}")
            print(f"   ROC-AUC: {prod_roc_auc:.4f}")
            print(f"   Run ID: {prod_version.run_id}")
            
            if should_replace and current_model_version:
                print(f"\n✅ Action: Replacement")
                print(f"   Old version: {current_model_version.version}")
                print(f"   New version: {prod_version.version}")
                print(f"   Improvement: +{roc_auc_diff:.4f} ({improvement_pct:+.2f}%)")
            elif not should_replace and current_model_version:
                print(f"\n⏭️  Action: Keep Current")
                print(f"   Reason: Insufficient improvement")
            else:
                print(f"\n✅ Action: First Registration")
        else:
            print("\n⚠️  No Production model found")
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
    
    # 12. Usage example
    print("\n" + "="*80)
    print("💡 USAGE IN PRODUCTION")
    print("="*80)
    
    print(f"""
# Load from MLflow Model Registry (always best version):
import mlflow

model = mlflow.sklearn.load_model("models:/{model_name}/Production")

# Make predictions:
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Load specific version:
model_v1 = mlflow.sklearn.load_model("models:/{model_name}/1")

# Load from run_id:
model_from_run = mlflow.sklearn.load_model("runs:/{best_run_id}/model")

# Load from local registry:
from register_best_model import load_from_registry
model, metadata = load_from_registry("{registry_name}", stage="production")
    """)
    
    print("\n" + "="*80)
    print("✅ REGISTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()