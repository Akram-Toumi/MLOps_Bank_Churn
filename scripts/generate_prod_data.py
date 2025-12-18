"""
Script pour générer les données de production avec drift significatif
Applique des transformations pour garantir la détection du drift par Evidently
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Configuration
BATCH_DIR = "data/batches"
OUTPUT_FILE = "data/production/bank_churn_prod.csv"

# Déterminer quel batch utiliser (argument ou défaut)
if len(sys.argv) > 1:
    batch_name = sys.argv[1]  # "batch1" ou "batch2"
else:
    batch_name = "batch1"  # Par défaut

INPUT_FILE = f"{BATCH_DIR}/{batch_name}.csv"

print("=" * 80)
print("GÉNÉRATION DES DONNÉES DE PRODUCTION AVEC DRIFT")
print("=" * 80)

# Vérifier que le batch existe
if not os.path.exists(INPUT_FILE):
    print(f"\n❌ ERREUR: Batch non trouvé: {INPUT_FILE}")
    print(f"\n💡 Exécutez d'abord: python scripts/split_dataset.py")
    sys.exit(1)

# Chargement
print(f"\n📂 Chargement du batch: {batch_name}")
df_prod = pd.read_csv(INPUT_FILE)
print(f"✅ Batch chargé: {df_prod.shape[0]:,} lignes")

# ============================================================================
# APPLICATION DU DRIFT SIGNIFICATIF
# ============================================================================
print(f"\n🔄 Application du drift significatif pour {batch_name}...")

# Drift différent selon le batch
if batch_name == "batch1":
    print("   Stratégie Batch 1: Drift modéré")
    
    # 1. Réduire l'âge moyen de 10 ans
    if 'Age' in df_prod.columns:
        df_prod['Age'] = df_prod['Age'] - 10
        df_prod['Age'] = df_prod['Age'].clip(18, 100)
        print("   ✓ Age: -10 ans")
    
    # 2. Augmenter la proportion de Germany
    if 'Geography' in df_prod.columns:
        # Convertir 30% des non-Germany en Germany
        mask = (df_prod['Geography'] != 'Germany') & (np.random.random(len(df_prod)) < 0.3)
        df_prod.loc[mask, 'Geography'] = 'Germany'
        print(f"   ✓ Geography: +30% Germany")
    
    # 3. Diminuer les soldes de 40%
    if 'Balance' in df_prod.columns:
        df_prod['Balance'] = df_prod['Balance'] * 0.6
        print("   ✓ Balance: -40%")
    
    # 4. Augmenter Credit Score
    credit_col = 'Credit Score' if 'Credit Score' in df_prod.columns else 'CreditScore'
    if credit_col in df_prod.columns:
        df_prod[credit_col] = df_prod[credit_col] + 50
        df_prod[credit_col] = df_prod[credit_col].clip(300, 850)
        print(f"   ✓ {credit_col}: +50 points")

elif batch_name == "batch2":
    print("   Stratégie Batch 2: Drift fort")
    
    # 1. Réduire l'âge moyen de 15 ans
    if 'Age' in df_prod.columns:
        df_prod['Age'] = df_prod['Age'] - 15
        df_prod['Age'] = df_prod['Age'].clip(18, 100)
        print("   ✓ Age: -15 ans")
    
    # 2. Augmenter massivement Germany
    if 'Geography' in df_prod.columns:
        # Convertir 50% des non-Germany en Germany
        mask = (df_prod['Geography'] != 'Germany') & (np.random.random(len(df_prod)) < 0.5)
        df_prod.loc[mask, 'Geography'] = 'Germany'
        print(f"   ✓ Geography: +50% Germany")
    
    # 3. Diminuer les soldes de 60%
    if 'Balance' in df_prod.columns:
        df_prod['Balance'] = df_prod['Balance'] * 0.4
        print("   ✓ Balance: -60%")
    
    # 4. Augmenter significativement Credit Score
    credit_col = 'Credit Score' if 'Credit Score' in df_prod.columns else 'CreditScore'
    if credit_col in df_prod.columns:
        df_prod[credit_col] = df_prod[credit_col] + 80
        df_prod[credit_col] = df_prod[credit_col].clip(300, 850)
        print(f"   ✓ {credit_col}: +80 points")
    
    # 5. Modifier EstimatedSalary
    if 'EstimatedSalary' in df_prod.columns:
        df_prod['EstimatedSalary'] = df_prod['EstimatedSalary'] * 1.3
        print("   ✓ EstimatedSalary: +30%")
    
    # 6. Augmenter NumOfProducts
    if 'NumOfProducts' in df_prod.columns:
        # Augmenter de 1 pour 40% des clients
        mask = np.random.random(len(df_prod)) < 0.4
        df_prod.loc[mask, 'NumOfProducts'] = df_prod.loc[mask, 'NumOfProducts'] + 1
        df_prod['NumOfProducts'] = df_prod['NumOfProducts'].clip(1, 4)
        print("   ✓ NumOfProducts: +1 pour 40% des clients")

# Créer le dossier de sortie
Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)

# Sauvegarde
print(f"\n💾 Sauvegarde...")
df_prod.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Fichier sauvegardé: {OUTPUT_FILE}")
print(f"   Taille: {len(df_prod):,} lignes × {len(df_prod.columns)} colonnes")

# Statistiques
if 'Churn Flag' in df_prod.columns or 'Churn' in df_prod.columns:
    churn_col = 'Churn Flag' if 'Churn Flag' in df_prod.columns else 'Churn'
    churn_rate = df_prod[churn_col].mean()
    print(f"\n📊 Statistiques:")
    print(f"   Churn rate: {churn_rate:.2%}")

if 'Geography' in df_prod.columns:
    geo_dist = df_prod['Geography'].value_counts(normalize=True)
    print(f"   Geography distribution:")
    for geo, pct in geo_dist.items():
        print(f"      {geo}: {pct:.1%}")

print("\n" + "=" * 80)
print("✅ GÉNÉRATION TERMINÉE")
print("=" * 80)
print(f"📁 Fichier: {OUTPUT_FILE}")
print(f"📊 Lignes: {len(df_prod):,}")
print(f"🔄 Batch: {batch_name}")
print(f"⚡ Drift appliqué: {'Modéré' if batch_name == 'batch1' else 'Fort'}")
print("\n💡 Prochaine étape: python monitoring/run_monitoring.py")
print("=" * 80)
