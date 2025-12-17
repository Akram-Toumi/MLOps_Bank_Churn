"""
Script simplifié pour générer les données de production avec drift
Utilise le fichier preprocessed_data.csv qui a des colonnes cohérentes
"""

import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = "data/churn.csv"  # Fichier avec 30000 lignes
OUTPUT_FILE = "data/production/bank_churn_prod.csv"
START_ROW = 20000
END_ROW = 30000
DRIFT_INTENSITY = 0.3  # 30% drift pour être sûr qu'il soit détecté

print("=" * 80)
print("GÉNÉRATION DES DONNÉES DE PRODUCTION")
print("=" * 80)

# Chargement
print(f"\n📂 Chargement du dataset...")
df_full = pd.read_csv(INPUT_FILE)
print(f"✅ Dataset chargé: {df_full.shape[0]:,} lignes")
print(f"📋 Colonnes: {list(df_full.columns[:10])}")

# Extraction
df_prod = df_full.iloc[START_ROW:END_ROW].copy()
print(f"✅ Subset extrait: {df_prod.shape[0]:,} lignes")

# Application du drift
print(f"\n🔄 Application du drift (intensité: {DRIFT_INTENSITY*100}%)...")

# Drift sur Balance
if 'Balance' in df_prod.columns:
    balance_mult = np.random.normal(1.2, 0.1, size=len(df_prod))
    df_prod['Balance'] = df_prod['Balance'] * balance_mult
    df_prod['Balance'] = df_prod['Balance'].clip(0, None)
    print("  ✓ Drift sur Balance (+20%)")

# Drift sur Credit Score
credit_col = 'Credit Score' if 'Credit Score' in df_prod.columns else 'CreditScore'
if credit_col in df_prod.columns:
    credit_shift = np.random.normal(-15, 5, size=len(df_prod))
    df_prod[credit_col] = df_prod[credit_col] + credit_shift
    df_prod[credit_col] = df_prod[credit_col].clip(300, 850).astype(int)
    print(f"  ✓ Drift sur {credit_col} (-15 points)")

# Drift sur Income
if 'Income' in df_prod.columns:
    income_mult = np.random.normal(1.1, 0.05, size=len(df_prod))
    df_prod['Income'] = df_prod['Income'] * income_mult
    df_prod['Income'] = df_prod['Income'].clip(0, None)
    print("  ✓ Drift sur Income (+10%)")

# Drift sur Churn
churn_col = 'Churn Flag' if 'Churn Flag' in df_prod.columns else 'Churn'
if churn_col in df_prod.columns:
    churn_mask = (df_prod[churn_col] == 0) & (np.random.random(len(df_prod)) < DRIFT_INTENSITY * 0.3)
    df_prod.loc[churn_mask, churn_col] = 1
    print(f"  ✓ Drift sur {churn_col} (+{DRIFT_INTENSITY*30:.1f}%)")

# Sauvegarde
print(f"\n💾 Sauvegarde...")
import os
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_prod.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Fichier sauvegardé: {OUTPUT_FILE}")
print(f"   Taille: {len(df_prod):,} lignes × {len(df_prod.columns)} colonnes")

print("\n" + "=" * 80)
print("✅ GÉNÉRATION TERMINÉE")
print("=" * 80)
print(f"📁 Fichier: {OUTPUT_FILE}")
print(f"📊 Lignes: {len(df_prod):,}")
print(f"🔄 Drift: {DRIFT_INTENSITY*100}%")
print("\n💡 Prochaine étape: python monitoring/run_monitoring.py")
print("=" * 80)
