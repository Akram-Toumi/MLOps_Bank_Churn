# ============================================================
# MLFLOW UI SETUP AND TROUBLESHOOTING GUIDE
# ============================================================

"""
ISSUE: Models not showing in MLflow UI at http://127.0.0.1:5000/

SOLUTION: Follow these steps
"""

# ============================================================
# STEP 1: CHECK YOUR MLFLOW TRACKING URI
# ============================================================

import mlflow
import os

# Print current tracking URI
print("="*80)
print("CURRENT MLFLOW CONFIGURATION")
print("="*80)
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Current Directory: {os.getcwd()}")
print()

# Check if mlruns directory exists
if os.path.exists("mlruns"):
    print("✅ mlruns directory found!")
    
    # List experiments
    experiments = os.listdir("mlruns")
    print(f"   Experiments found: {len([e for e in experiments if os.path.isdir(os.path.join('mlruns', e))])}")
    
    for exp in experiments:
        exp_path = os.path.join("mlruns", exp)
        if os.path.isdir(exp_path) and exp != ".trash":
            print(f"   • {exp}")
else:
    print("❌ mlruns directory NOT found!")
    print("   Solution: Make sure you run the training script first")

print()

# ============================================================
# STEP 2: PROPERLY CONFIGURE MLFLOW IN YOUR TRAINING SCRIPT
# ============================================================

print("="*80)
print("HOW TO FIX YOUR TRAINING SCRIPT")
print("="*80)
print("""
Add this at the VERY BEGINNING of your training script:

import mlflow

# Set tracking URI to local directory
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
mlflow.set_experiment("Churn_Prediction_Pipeline")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Experiment: {mlflow.get_experiment_by_name('Churn_Prediction_Pipeline')}")
""")

print()

# ============================================================
# STEP 3: START MLFLOW UI
# ============================================================

print("="*80)
print("HOW TO START MLFLOW UI")
print("="*80)
print("""
OPTION 1: Start from command line (RECOMMENDED)
----------------------------------------------
Open a terminal in your project directory and run:

    mlflow ui

This will start the UI on http://127.0.0.1:5000


OPTION 2: Start on a different port
-----------------------------------
If port 5000 is busy:

    mlflow ui --port 5001

Then access: http://127.0.0.1:5001


OPTION 3: Start with specific backend store
-------------------------------------------
If you want to specify the mlruns location:

    mlflow ui --backend-store-uri ./mlruns

""")

# ============================================================
# STEP 4: VERIFY YOUR RUNS ARE LOGGED
# ============================================================

print("="*80)
print("VERIFY YOUR RUNS")
print("="*80)

try:
    # Try to get the experiment
    client = mlflow.tracking.MlflowClient()
    
    experiments = client.search_experiments()
    
    if experiments:
        print("✅ Experiments found:")
        for exp in experiments:
            print(f"\n   Experiment: {exp.name}")
            print(f"   ID: {exp.experiment_id}")
            
            # Get runs for this experiment
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            print(f"   Runs: {len(runs)}")
            
            if runs:
                for run in runs[:3]:  # Show first 3 runs
                    print(f"      • {run.info.run_name} (ID: {run.info.run_id})")
    else:
        print("❌ No experiments found!")
        print("   Make sure you've run your training script")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Make sure MLflow is properly configured")

print()

# ============================================================
# STEP 5: COMPLETE WORKING EXAMPLE
# ============================================================

print("="*80)
print("COMPLETE WORKING EXAMPLE")
print("="*80)
print("""
Here's a minimal example that WILL show up in MLflow UI:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. SETUP MLFLOW
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Test_Experiment")

# 2. CREATE DATA
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. TRAIN WITH MLFLOW
with mlflow.start_run(run_name="Test_Model"):
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Model logged! Accuracy: {accuracy:.4f}")

# 4. START UI
# Now run in terminal: mlflow ui
```

After running this, start MLflow UI and you WILL see your run!
""")

print()

# ============================================================
# TROUBLESHOOTING CHECKLIST
# ============================================================

print("="*80)
print("TROUBLESHOOTING CHECKLIST")
print("="*80)
print("""
☐ 1. mlflow.set_tracking_uri() is called BEFORE any logging
☐ 2. mlflow.set_experiment() is called to set experiment name
☐ 3. mlflow.start_run() is used for each training run
☐ 4. Metrics/params are logged inside the 'with' block
☐ 5. mlruns directory exists in your project folder
☐ 6. MLflow UI is started with: mlflow ui
☐ 7. You're accessing the correct URL (check terminal output)
☐ 8. No firewall blocking port 5000
☐ 9. Training script ran successfully without errors
☐ 10. You refreshed the browser page after training
""")

print()

# ============================================================
# QUICK TEST
# ============================================================

print("="*80)
print("RUNNING QUICK TEST")
print("="*80)

try:
    # Quick test to ensure MLflow works
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MLflow_Test")
    
    with mlflow.start_run(run_name="Quick_Test"):
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 0.95)
        print("✅ Test run logged successfully!")
        print("   Now start MLflow UI: mlflow ui")
        print("   Then check: http://127.0.0.1:5000")
        
except Exception as e:
    print(f"❌ Test failed: {e}")

print()
print("="*80)
print("SETUP COMPLETE")
print("="*80)
print("Next steps:")
print("1. Run your training script")
print("2. Open terminal and run: mlflow ui")
print("3. Open browser: http://127.0.0.1:5000")
print("="*80)