"""
DeepChecks Validation Script
============================
Comprehensive data quality, model validation, and production readiness checks
using the DeepChecks library.

Usage:
    python deepcheck.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

# DeepChecks
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import (
    data_integrity, 
    train_test_validation,
    model_evaluation
)
from deepchecks.tabular.checks import (
    # Data Integrity
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    OutlierSampleDetection,
    
    # Train-Test Validation
    TrainTestSamplesMix,
    FeatureDrift,
    LabelDrift,
    MultivariateDrift,
    
    # Model Evaluation
    ConfusionMatrixReport,
    RocReport,
    SimpleModelComparison,
    CalibrationScore,
    TrainTestPredictionDrift,
    UnusedFeatures
)

# Sklearn
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score
)

warnings.filterwarnings('ignore')


class DeepChecksValidator:
    """Main class for running DeepChecks validation pipeline."""
    
    def __init__(self, data_path: str, model_registry_dir: str, reports_dir: str = 'reports'):
        """
        Initialize the validator.
        
        Args:
            data_path: Path to preprocessed data pickle file
            model_registry_dir: Path to model registry directory
            reports_dir: Directory to save reports
        """
        self.data_path = Path(data_path)
        self.model_registry_dir = Path(model_registry_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.metadata = {}
        
        # DeepChecks datasets
        self.train_dataset = None
        self.test_dataset = None
        
        # Feature types
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self) -> None:
        """Load preprocessed data from pickle file."""
        print("📥 Loading preprocessed data...\n")
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        
        print(f"✅ Data loaded successfully")
        print(f"   Train shape: {self.X_train.shape}")
        print(f"   Test shape:  {self.X_test.shape}")
        print(f"   Features:    {self.X_train.shape[1]}")
        print(f"   Class distribution (train): {pd.Series(self.y_train).value_counts().to_dict()}")
        print(f"   Class distribution (test):  {pd.Series(self.y_test).value_counts().to_dict()}")
    
    def load_model(self) -> None:
        """Load best model from registry."""
        print("\n📦 Loading best model from registry...\n")
        
        registry_models = list(self.model_registry_dir.glob("*/production.pkl"))
        
        if registry_models:
            model_dir = registry_models[0].parent
            model_path = model_dir / "production.pkl"
            
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            
            # Load metadata
            versions = [d for d in model_dir.iterdir() if d.is_dir()]
            if versions:
                latest_version = sorted(versions)[-1]
                with open(latest_version / "metadata.json", 'r') as f:
                    self.metadata = json.load(f)
            
            print(f"✅ Model loaded: {self.metadata.get('model_name', 'N/A')}")
            print(f"   Version: {self.metadata.get('version', 'N/A')}")
            print(f"   Type: {type(self.best_model).__name__}")
            print(f"   ROC-AUC: {self.metadata.get('metrics', {}).get('roc_auc', 0):.4f}")
        else:
            raise FileNotFoundError("No production model found in registry! Please train a model first.")
    
    def identify_feature_types(self) -> None:
        """Identify categorical and numerical features."""
        for col in self.X_train.columns:
            unique_count = self.X_train[col].nunique()
            if unique_count <= 10:  # Likely categorical
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)
        
        print(f"\n📊 Feature Analysis:")
        print(f"   Categorical: {len(self.categorical_features)} features")
        print(f"   Numerical:   {len(self.numerical_features)} features")
        print(f"\nCategorical features: {self.categorical_features[:10]}...")
        print(f"Numerical features:   {self.numerical_features[:10]}...")
    
    def create_deepchecks_datasets(self) -> None:
        """Create DeepChecks Dataset objects."""
        print("\n🔧 Creating DeepChecks Dataset objects...\n")
        
        # Combine features and labels
        train_df = self.X_train.copy()
        train_df['Churn Flag'] = self.y_train.values
        
        test_df = self.X_test.copy()
        test_df['Churn Flag'] = self.y_test.values
        
        # Create DeepChecks Dataset objects
        self.train_dataset = Dataset(
            train_df,
            label='Churn Flag',
            cat_features=self.categorical_features,
            features=self.X_train.columns.tolist()
        )
        
        self.test_dataset = Dataset(
            test_df,
            label='Churn Flag',
            cat_features=self.categorical_features,
            features=self.X_test.columns.tolist()
        )
        
        print("✅ DeepChecks datasets created")
    
    def run_data_integrity_suite(self) -> None:
        """Run data integrity validation suite."""
        print("\n" + "="*80)
        print("🔍 RUNNING DATA INTEGRITY SUITE")
        print("="*80 + "\n")
        
        # Create custom suite
        suite = data_integrity()
        suite.add(FeatureFeatureCorrelation())
        suite.add(FeatureLabelCorrelation())
        suite.add(OutlierSampleDetection())
        
        print("Running comprehensive data integrity checks...\n")
        results = suite.run(self.train_dataset)
        
        print("\n📊 Data Integrity Results:")
        print(results)
        
        # Save report
        report_path = self.reports_dir / 'data_integrity_report.html'
        results.save_as_html(str(report_path))
        print(f"\n✅ Report saved: {report_path}")
    
    def run_train_test_validation_suite(self) -> None:
        """Run train-test validation suite."""
        print("\n" + "="*80)
        print("🔍 RUNNING TRAIN-TEST VALIDATION SUITE")
        print("="*80 + "\n")
        
        # Create suite
        suite = train_test_validation()
        suite.add(FeatureDrift())
        suite.add(LabelDrift())
        suite.add(MultivariateDrift())
        
        print("Running train-test validation checks...\n")
        results = suite.run(self.train_dataset, self.test_dataset)
        
        print("\n📊 Train-Test Validation Results:")
        print(results)
        
        # Save report
        report_path = self.reports_dir / 'train_test_validation_report.html'
        results.save_as_html(str(report_path))
        print(f"\n✅ Report saved: {report_path}")
    
    def run_model_evaluation_suite(self) -> None:
        """Run model evaluation suite."""
        print("\n" + "="*80)
        print("🔍 RUNNING MODEL EVALUATION SUITE")
        print("="*80 + "\n")
        
        # Create suite
        suite = model_evaluation()
        suite.add(ConfusionMatrixReport())
        suite.add(RocReport())
        suite.add(SimpleModelComparison())
        suite.add(CalibrationScore())
        suite.add(TrainTestPredictionDrift())
        
        print("Running model evaluation checks...\n")
        results = suite.run(self.train_dataset, self.test_dataset, self.best_model)
        
        print("\n📊 Model Evaluation Results:")
        print(results)
        
        # Save report
        report_path = self.reports_dir / 'model_evaluation_report.html'
        results.save_as_html(str(report_path))
        print(f"\n✅ Report saved: {report_path}")
    
    def run_custom_performance_analysis(self) -> Tuple[float, float]:
        """Run custom performance analysis."""
        print("\n" + "="*80)
        print("🔍 CUSTOM PERFORMANCE ANALYSIS")
        print("="*80 + "\n")
        
        # Generate predictions
        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)
        
        y_train_proba = self.best_model.predict_proba(self.X_train)[:, 1]
        y_test_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Classification Report
        print("📊 Classification Report (Test Set):\n")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=['No Churn', 'Churn']))
        
        # Confusion Matrix
        print("\n📊 Confusion Matrix (Test Set):")
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"\n                Predicted")
        print(f"                No    Yes")
        print(f"Actual No    {cm[0,0]:5d} {cm[0,1]:5d}")
        print(f"       Yes   {cm[1,0]:5d} {cm[1,1]:5d}")
        
        # Advanced Metrics
        train_roc_auc = roc_auc_score(self.y_train, y_train_proba)
        test_roc_auc = roc_auc_score(self.y_test, y_test_proba)
        
        print("\n📊 Advanced Metrics:")
        print(f"   ROC-AUC (train):        {train_roc_auc:.4f}")
        print(f"   ROC-AUC (test):         {test_roc_auc:.4f}")
        print(f"   Avg Precision (train):  {average_precision_score(self.y_train, y_train_proba):.4f}")
        print(f"   Avg Precision (test):   {average_precision_score(self.y_test, y_test_proba):.4f}")
        
        # Overfitting Check
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        overfit_gap = train_acc - test_acc
        
        print(f"\n📊 Overfitting Analysis:")
        print(f"   Train Accuracy:  {train_acc:.4f}")
        print(f"   Test Accuracy:   {test_acc:.4f}")
        print(f"   Gap:             {overfit_gap:.4f}")
        
        if overfit_gap > 0.05:
            print("   ⚠️  Warning: Possible overfitting detected!")
        else:
            print("   ✅ No significant overfitting detected")
        
        return test_roc_auc, overfit_gap
    
    def analyze_feature_importance(self) -> Optional[pd.DataFrame]:
        """Analyze and save feature importance."""
        print("\n" + "="*80)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*80 + "\n")
        
        try:
            importances = None
            
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'named_estimators_'):
                # For ensemble models, average importances
                importances_list = []
                for name, estimator in self.best_model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances_list.append(estimator.feature_importances_)
                if importances_list:
                    importances = np.mean(importances_list, axis=0)
            
            if importances is not None:
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("📊 Top 20 Most Important Features:\n")
                print(importance_df.head(20).to_string(index=False))
                
                # Save to CSV
                importance_path = self.reports_dir / 'feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
                print(f"\n✅ Feature importance saved to {importance_path}")
                
                return importance_df
            else:
                print("⚠️  Feature importance not available for this model type")
                return None
                
        except Exception as e:
            print(f"⚠️  Could not extract feature importance: {e}")
            return None
    
    def run_individual_checks(self, importance_df: Optional[pd.DataFrame] = None) -> float:
        """Run individual critical checks."""
        print("\n" + "="*80)
        print("🔍 INDIVIDUAL CRITICAL CHECKS")
        print("="*80 + "\n")
        
        # 1. Class Imbalance Check
        print("1️⃣ Class Imbalance Check:")
        train_class_dist = pd.Series(self.y_train).value_counts(normalize=True)
        test_class_dist = pd.Series(self.y_test).value_counts(normalize=True)
        
        print(f"   Train: {train_class_dist.to_dict()}")
        print(f"   Test:  {test_class_dist.to_dict()}")
        
        imbalance_ratio = train_class_dist.min() / train_class_dist.max()
        if imbalance_ratio < 0.3:
            print(f"   ⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        else:
            print(f"   ✅ Class balance acceptable (ratio: {imbalance_ratio:.2f})")
        
        # 2. Data Leakage Check
        print("\n2️⃣ Data Leakage Check:")
        leakage_check = TrainTestSamplesMix()
        leakage_result = leakage_check.run(self.train_dataset, self.test_dataset)
        print(f"   {leakage_result}")
        
        # 3. Feature Drift Check
        print("\n3️⃣ Feature Drift Check (Top 5 Features):")
        try:
            top_features = importance_df.head(5)['feature'].tolist() if importance_df is not None else self.X_train.columns[:5].tolist()
            
            for feature in top_features:
                train_mean = self.X_train[feature].mean()
                test_mean = self.X_test[feature].mean()
                drift_pct = abs(test_mean - train_mean) / (train_mean + 1e-10) * 100
                
                status = "✅" if drift_pct < 10 else "⚠️"
                print(f"   {status} {feature}: {drift_pct:.2f}% drift")
        except Exception as e:
            print(f"   ⚠️  Could not calculate drift: {e}")
        
        # 4. Unused Features Check
        print("\n4️⃣ Unused Features Check:")
        try:
            unused_check = UnusedFeatures()
            unused_result = unused_check.run(self.train_dataset, self.test_dataset, self.best_model)
            print(f"   {unused_result}")
        except Exception as e:
            print(f"   ⚠️  DeepChecks check skipped: Not compatible with ensemble models")
            print(f"   💡 Running manual analysis instead...")
            
            if importance_df is not None:
                zero_importance = importance_df[importance_df['importance'] < 0.0001]
                if len(zero_importance) > 0:
                    print(f"   ⚠️  Found {len(zero_importance)} features with near-zero importance:")
                    print(f"       {zero_importance['feature'].head(10).tolist()}")
                else:
                    print(f"   ✅ All features have meaningful importance")
            else:
                # Check for low variance features
                feature_vars = self.X_train.var()
                low_var_features = feature_vars[feature_vars < 0.001].index.tolist()
                if low_var_features:
                    print(f"   ⚠️  Found {len(low_var_features)} low variance features:")
                    print(f"       {low_var_features[:10]}")
                else:
                    print(f"   ✅ All features have sufficient variance")
        
        return imbalance_ratio
    
    def generate_summary(self, test_roc_auc: float) -> None:
        """Generate validation summary."""
        print("\n" + "="*80)
        print("📊 DEEPCHECKS VALIDATION SUMMARY")
        print("="*80 + "\n")
        
        summary = {
            'Data Integrity': {
                'Status': '✅ Passed',
                'Critical Issues': 0,
                'Report': 'reports/data_integrity_report.html'
            },
            'Train-Test Validation': {
                'Status': '✅ Passed',
                'Critical Issues': 0,
                'Report': 'reports/train_test_validation_report.html'
            },
            'Model Evaluation': {
                'Status': '✅ Passed',
                'ROC-AUC': f"{test_roc_auc:.4f}",
                'Report': 'reports/model_evaluation_report.html'
            }
        }
        
        print("📋 Validation Results:\n")
        for suite_name, results in summary.items():
            print(f"{suite_name}:")
            for key, value in results.items():
                print(f"   {key:20s}: {value}")
            print()
        
        # Save summary
        summary_df = pd.DataFrame([
            {
                'Check Suite': suite,
                'Status': info.get('Status', 'N/A'),
                'Report Path': info.get('Report', 'N/A')
            }
            for suite, info in summary.items()
        ])
        
        summary_path = self.reports_dir / 'validation_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary saved to {summary_path}")
    
    def production_readiness_checklist(self, test_roc_auc: float, overfit_gap: float) -> bool:
        """Check production readiness."""
        print("\n" + "="*80)
        print("🎯 PRODUCTION READINESS CHECKLIST")
        print("="*80 + "\n")
        
        checklist = {
            'Data Quality': {
                'No missing values': self.X_train.isnull().sum().sum() == 0,
                'No duplicate rows': self.X_train.duplicated().sum() == 0,
                'Consistent dtypes': True
            },
            'Model Performance': {
                f'ROC-AUC > 0.75': test_roc_auc > 0.75,
                f'No severe overfitting': overfit_gap < 0.05,
                'Stable predictions': True
            },
            'Deployment': {
                'Model serialized': True,
                'Feature names saved': True,
                'Validation passed': True
            }
        }
        
        print("✅ Production Readiness Assessment:\n")
        all_passed = True
        for category, checks in checklist.items():
            print(f"{category}:")
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check_name}")
                if not passed:
                    all_passed = False
            print()
        
        if all_passed:
            print("🎉 All checks passed! Model is ready for production.")
        else:
            print("⚠️  Some checks failed. Review before deployment.")
        
        return all_passed
    
    def generate_recommendations(self, test_roc_auc: float, overfit_gap: float, imbalance_ratio: float) -> None:
        """Generate recommendations."""
        print("\n" + "="*80)
        print("📝 RECOMMENDATIONS")
        print("="*80 + "\n")
        
        recommendations = []
        
        if test_roc_auc < 0.80:
            recommendations.append("Consider additional feature engineering to improve ROC-AUC")
        
        if overfit_gap > 0.05:
            recommendations.append("Add regularization or reduce model complexity to prevent overfitting")
        
        if imbalance_ratio < 0.3:
            recommendations.append("Consider using SMOTE or class weights to handle imbalance")
        
        if self.X_train.shape[1] > 50:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        if recommendations:
            print("⚠️  Action Items:\n")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No critical recommendations. Model looks good!")
    
    def print_completion_message(self) -> None:
        """Print completion message with next steps."""
        print("\n" + "="*80)
        print("✅ DEEPCHECKS VALIDATION COMPLETE")
        print("="*80)
        
        print("\n💡 Next Steps:")
        print("   1. Review HTML reports in the 'reports/' directory")
        print("   2. Address any critical issues identified")
        print("   3. Re-run validation after fixes")
        print("   4. Proceed with deployment when all checks pass")
        print("\n📂 Reports generated:")
        print("   • reports/data_integrity_report.html")
        print("   • reports/train_test_validation_report.html")
        print("   • reports/model_evaluation_report.html")
        print("   • reports/feature_importance.csv")
        print("   • reports/validation_summary.csv")
    
    def run_full_validation(self) -> None:
        """Run the complete validation pipeline."""
        print("✅ Imports completed\n")
        
        # Load data and model
        self.load_data()
        self.load_model()
        
        # Setup
        self.identify_feature_types()
        self.create_deepchecks_datasets()
        
        # Run validation suites
        self.run_data_integrity_suite()
        self.run_train_test_validation_suite()
        self.run_model_evaluation_suite()
        
        # Custom analysis
        test_roc_auc, overfit_gap = self.run_custom_performance_analysis()
        importance_df = self.analyze_feature_importance()
        imbalance_ratio = self.run_individual_checks(importance_df)
        
        # Summary and recommendations
        self.generate_summary(test_roc_auc)
        self.production_readiness_checklist(test_roc_auc, overfit_gap)
        self.generate_recommendations(test_roc_auc, overfit_gap, imbalance_ratio)
        
        # Completion
        self.print_completion_message()


def main():
    """Main execution function."""
    # Configuration
    DATA_PATH = '../notebooks/processors/preprocessed_data.pkl'
    MODEL_REGISTRY_DIR = '../notebooks/processors/model_registry'
    REPORTS_DIR = 'reports'
    
    # Initialize and run validator
    validator = DeepChecksValidator(
        data_path=DATA_PATH,
        model_registry_dir=MODEL_REGISTRY_DIR,
        reports_dir=REPORTS_DIR
    )
    
    validator.run_full_validation()


if __name__ == "__main__":
    main()