"""
Train churn prediction models with MLflow tracking
Trains baseline, tuned, and ensemble models
"""

# Imports
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from pathlib import Path

# MLflow
import mlflow
import mlflow.sklearn

# ML Models
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Metrics
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
DATA_PATH = '../notebooks/processors/preprocessed_data.pkl'
N_ITER = 10  # Number of iterations for tuning
CV_FOLDS = 3  # Cross-validation folds

# Disable incompatible features (security)
os.environ["MLFLOW_ENABLE_LOGGED_MODEL_CREATION"] = "false"


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


def log_model_mlflow(model, model_name, stage, metrics, duration, X_train, best_params=None):
    """Log a model to MLflow"""
    with mlflow.start_run(run_name=f"{model_name}_{stage}"):
        # Log params
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('stage', stage)
        mlflow.log_param('n_features', X_train.shape[1])
        
        # Log best params if available
        if best_params:
            for k, v in best_params.items():
                try:
                    mlflow.log_param(f'best_{k}', v)
                except:
                    pass
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric('training_duration', duration)
        
        # Save model locally
        model_filename = f"{model_name}_{stage}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Log as artifact
        try:
            mlflow.log_artifact(model_filename)
        except:
            pass
        
        run_id = mlflow.active_run().info.run_id
        return run_id, model_filename


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def get_baseline_config():
    """Get baseline model configurations"""
    return {
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False,
            tree_method='hist'
        ),
        
        'LightGBM': LGBMClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=40,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            importance_type='gain'
        ),
        
        'Random_Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            warm_start=False
        ),
        
        'CatBoost': CatBoostClassifier(
            iterations=150,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=128,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=False,
            task_type='CPU',
            bootstrap_type='Bernoulli',
            subsample=0.85
        ),
        
        'Logistic_Regression_ElasticNet': LogisticRegression(
            penalty='elasticnet',
            C=1.0,
            l1_ratio=0.5,
            solver='saga',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            warm_start=False
        )
    }


def get_search_spaces():
    """Get hyperparameter search spaces"""
    return {
        'XGBoost': {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'min_child_weight': randint(1, 10),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2)
        },
        
        'LightGBM': {
            'n_estimators': randint(100, 500),
            'max_depth': randint(-1, 15),
            'learning_rate': uniform(0.01, 0.29),
            'num_leaves': randint(15, 150),
            'min_child_samples': randint(10, 60),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2),
            'min_split_gain': uniform(0, 0.1)
        },
        
        'Random_Forest': {
            'n_estimators': randint(100, 500),
            'max_depth': [15, 20, 25, 30, 35, None],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
            'max_samples': uniform(0.7, 0.3),
            'class_weight': ['balanced', 'balanced_subsample']
        },
        
        'CatBoost': {
            'iterations': randint(100, 500),
            'depth': randint(4, 11),
            'learning_rate': uniform(0.01, 0.29),
            'l2_leaf_reg': uniform(1, 9),
            'border_count': [32, 64, 128, 200, 254],
            'random_strength': uniform(0, 2)
        },
        
        'Logistic_Regression_ElasticNet': {
            'C': uniform(0.001, 10),
            'l1_ratio': uniform(0, 1),
            'max_iter': randint(500, 2000),
            'tol': uniform(1e-5, 1e-3)
        }
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_baseline_models(baseline_config, X_train, y_train, X_test, y_test):
    """Train baseline models"""
    baseline_results = []
    trained_models = {}
    
    print("🚀 Training BASELINE models...\n")
    
    for name, model in baseline_config.items():
        print(f"📊 {name}...", end=" ")
        start = datetime.now()
        
        # Training
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        duration = (datetime.now() - start).total_seconds()
        
        # Log to MLflow
        run_id, model_file = log_model_mlflow(model, name, 'baseline', metrics, duration, X_train)
        
        # Store
        trained_models[f"{name}_baseline"] = model
        baseline_results.append({
            'model': name,
            'stage': 'baseline',
            'run_id': run_id,
            **metrics,
            'duration': duration
        })
        
        print(f"ROC-AUC: {metrics['roc_auc']:.4f} ({duration:.1f}s)")
    
    print("\n✅ Baseline completed!")
    return baseline_results, trained_models


def train_tuned_models(baseline_config, search_spaces, X_train, y_train, X_test, y_test, n_iter, cv_folds):
    """Train tuned models with hyperparameter optimization"""
    tuned_results = []
    trained_models = {}
    
    print(f"🔍 Fine-Tuning ({n_iter} iterations × {cv_folds} folds)...\n")
    
    for name, base_model in baseline_config.items():
        print(f"📊 {name}...", end=" ")
        start = datetime.now()
        
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            base_model,
            search_spaces[name],
            n_iter=n_iter,
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        duration = (datetime.now() - start).total_seconds()
        
        # Log to MLflow
        run_id, model_file = log_model_mlflow(
            best_model, name, 'tuned', metrics, duration, X_train, search.best_params_
        )
        
        # Store
        trained_models[f"{name}_tuned"] = best_model
        tuned_results.append({
            'model': name,
            'stage': 'tuned',
            'run_id': run_id,
            **metrics,
            'duration': duration
        })
        
        print(f"ROC-AUC: {metrics['roc_auc']:.4f} ({duration:.1f}s)")
    
    print("\n✅ Fine-tuning completed!")
    return tuned_results, trained_models


def train_ensemble_models(trained_models, X_train, y_train, X_test, y_test):
    """Train ensemble models"""
    estimators = [
        ('lgbm', trained_models['LightGBM_tuned']),
        ('rf', trained_models['Random_Forest_tuned']),
        ('cat', trained_models['CatBoost_tuned']),
        ('lr', trained_models['Logistic_Regression_ElasticNet_tuned'])
    ]
    
    ensemble_results = []
    ensemble_models = {}
    
    print("🚀 Training ENSEMBLE models...\n")
    
    # 1. Stacking with Logistic Regression
    print("📊 Stacking (LogReg)...", end=" ")
    
    meta_learner = LogisticRegression(
        penalty='elasticnet',
        C=0.5,
        l1_ratio=0.3,
        solver='saga',
        max_iter=1500,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    start = datetime.now()
    
    stacking_lr = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False,
        verbose=0
    )
    
    stacking_lr.fit(X_train, y_train)
    y_pred = stacking_lr.predict(X_test)
    y_proba = stacking_lr.predict_proba(X_test)[:, 1]
    
    metrics_stack_lr = calculate_metrics(y_test, y_pred, y_proba)
    duration = (datetime.now() - start).total_seconds()
    
    run_id_lr, _ = log_model_mlflow(stacking_lr, 'Stacking_LR', 'ensemble', metrics_stack_lr, duration, X_train)
    ensemble_models['Stacking_LR'] = stacking_lr
    ensemble_results.append({
        'model': 'Stacking_LR',
        'stage': 'ensemble',
        'run_id': run_id_lr,
        **metrics_stack_lr,
        'duration': duration
    })
    
    print(f"ROC-AUC: {metrics_stack_lr['roc_auc']:.4f} ({duration:.1f}s)")
    
    # 2. Voting Classifier (soft voting)
    print("📊 Voting (Soft)...", end=" ")
    start = datetime.now()
    
    voting_soft = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1,
        flatten_transform=True,
        verbose=False
    )
    
    voting_soft.fit(X_train, y_train)
    y_pred = voting_soft.predict(X_test)
    y_proba = voting_soft.predict_proba(X_test)[:, 1]
    
    metrics_voting_soft = calculate_metrics(y_test, y_pred, y_proba)
    duration = (datetime.now() - start).total_seconds()
    
    run_id_soft, _ = log_model_mlflow(voting_soft, 'Voting_Soft', 'ensemble', metrics_voting_soft, duration, X_train)
    ensemble_models['Voting_Soft'] = voting_soft
    ensemble_results.append({
        'model': 'Voting_Soft',
        'stage': 'ensemble',
        'run_id': run_id_soft,
        **metrics_voting_soft,
        'duration': duration
    })
    
    print(f"ROC-AUC: {metrics_voting_soft['roc_auc']:.4f} ({duration:.1f}s)")
    
    print("\n✅ Ensembles completed!")
    return ensemble_results, ensemble_models


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print("CHURN PREDICTION MODEL TRAINING")
    print("="*80)
    
    # 1. Setup MLflow
    print("\n📊 Setting up MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow configured")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    
    # 2. Load data
    print(f"\n📂 Loading preprocessed data...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("✅ Data loaded")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    # 3. Get configurations
    baseline_config = get_baseline_config()
    search_spaces = get_search_spaces()
    
    # 4. Train baseline models
    print("\n" + "="*80)
    baseline_results, trained_models_baseline = train_baseline_models(
        baseline_config, X_train, y_train, X_test, y_test
    )
    
    # 5. Train tuned models
    print("\n" + "="*80)
    tuned_results, trained_models_tuned = train_tuned_models(
        baseline_config, search_spaces, X_train, y_train, X_test, y_test, N_ITER, CV_FOLDS
    )
    
    # Combine trained models
    trained_models = {**trained_models_baseline, **trained_models_tuned}
    
    # 6. Train ensemble models
    print("\n" + "="*80)
    ensemble_results, trained_models_ensemble = train_ensemble_models(
        trained_models_tuned, X_train, y_train, X_test, y_test
    )
    
    trained_models.update(trained_models_ensemble)
    
    # 7. Combine all results
    all_results = baseline_results + tuned_results + ensemble_results
    df_results = pd.DataFrame(all_results)
    
    # 8. Display results
    print("\n" + "="*80)
    print("📊 ALL MODELS RESULTS")
    print("="*80)
    print(df_results[['model', 'stage', 'roc_auc', 'f1_score', 'duration']].to_string(index=False))
    
    print("\n🏆 Top 5 models (ROC-AUC):")
    top5 = df_results.nlargest(5, 'roc_auc')[['model', 'stage', 'roc_auc', 'f1_score']]
    print(top5.to_string(index=False))
    
    # 9. Save results and models
    print("\n💾 Saving results...")
    
    # Save results DataFrame
    results_path = 'training_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'df_results': df_results,
            'trained_models': trained_models,
            'X_test': X_test,
            'y_test': y_test
        }, f)
    print(f"✅ Results saved to {results_path}")
    
    # 10. Final summary
    print("\n" + "="*80)
    print("🎉 TRAINING SUMMARY")
    print("="*80)
    
    print(f"\n📊 Models trained:")
    print(f"   • Baseline:  {len(baseline_results)} models")
    print(f"   • Tuned:     {len(tuned_results)} models (n_iter={N_ITER})")
    print(f"   • Ensemble:  {len(ensemble_results)} models")
    print(f"   • TOTAL:     {len(all_results)} models")
    
    best_idx = df_results['roc_auc'].idxmax()
    best_row = df_results.loc[best_idx]
    
    print(f"\n🏆 Best model:")
    print(f"   • Name:      {best_row['model']}")
    print(f"   • Stage:     {best_row['stage']}")
    print(f"   • ROC-AUC:   {best_row['roc_auc']:.4f}")
    print(f"   • F1-Score:  {best_row['f1_score']:.4f}")
    print(f"   • Run ID:    {best_row['run_id']}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n💡 Next steps:")
    print("   1. Review results in MLflow UI")
    print("   2. Run register_best_model.py to register the best model")
    print("   3. Deploy to production")


if __name__ == "__main__":
    main()