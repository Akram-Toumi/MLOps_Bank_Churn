"""
Script pour diviser le dataset en 3 parties ET appliquer les transformations de drift
Part 1: 0-30,000 (référence - pas de modification)
Part 2: 30,000-60,000 (Batch 1 - crise économique)
Part 3: 60,000-fin (Batch 2 - changement démographique)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
INPUT_FILE = "data/bank_customer_churn.csv"
OUTPUT_DIR = "data/batches"

# Points de division
PART1_END = 30000
PART2_END = 60000

print("=" * 80)
print("DIVISION DU DATASET + APPLICATION DU DRIFT")
print("=" * 80)

# Créer les dossiers de sortie
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path("data/train").mkdir(parents=True, exist_ok=True)

# Charger le dataset complet
print(f"\n📂 Chargement de {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
total_rows = len(df)
print(f"✅ Dataset chargé: {total_rows:,} lignes")

# Vérifier qu'on a assez de données
if total_rows < PART2_END:
    print(f"\n⚠️  WARNING: Dataset a seulement {total_rows:,} lignes")
    print(f"   Ajustement des points de division...")
    PART1_END = int(total_rows * 0.33)
    PART2_END = int(total_rows * 0.67)

# Division
print(f"\n✂️  Division du dataset...")
print(f"   Part 1: 0 → {PART1_END:,} ({PART1_END:,} lignes)")
print(f"   Part 2: {PART1_END:,} → {PART2_END:,} ({PART2_END - PART1_END:,} lignes)")
print(f"   Part 3: {PART2_END:,} → {total_rows:,} ({total_rows - PART2_END:,} lignes)")

part1 = df.iloc[:PART1_END].copy()
part2 = df.iloc[PART1_END:PART2_END].copy()
part3 = df.iloc[PART2_END:].copy()

# ============================================================================
# PART 1: Référence (pas de modification)
# ============================================================================
print(f"\n� Part 1: Données de référence (pas de modification)")
part1_file = "data/train/part1.csv"
part1.to_csv(part1_file, index=False)
print(f"✅ Sauvegardée: {part1_file}")

# ============================================================================
# PART 2 (BATCH 1): Crise économique
# ============================================================================
print(f"\n📉 Part 2 (Batch 1): Application crise économique...")

# 1. Baisse des revenus (-20%)
if 'Income' in part2.columns:
    part2['Income'] = part2['Income'] * 0.8
    print("   ✓ Income: -20%")

# 2. Augmentation des prêts (+30%)
if 'Outstanding Loans' in part2.columns:
    part2['Outstanding Loans'] = part2['Outstanding Loans'] * 1.3
    print("   ✓ Outstanding Loans: +30%")

# 3. Baisse des soldes (-25%)
if 'Balance' in part2.columns:
    part2['Balance'] = part2['Balance'] * 0.75
    part2['Balance'] = part2['Balance'].clip(0, None)
    print("   ✓ Balance: -25%")

# 4. Baisse du Credit Score (-30 points)
if 'Credit Score' in part2.columns:
    part2['Credit Score'] = part2['Credit Score'] - 30
    part2['Credit Score'] = part2['Credit Score'].clip(300, 850)
    print("   ✓ Credit Score: -30 points")

part2_file = f"{OUTPUT_DIR}/batch1.csv"
part2.to_csv(part2_file, index=False)
print(f"✅ Sauvegardée: {part2_file}")

# ============================================================================
# PART 3 (BATCH 2): Changement démographique
# ============================================================================
print(f"\n👥 Part 3 (Batch 2): Application changement démographique...")

# 1. Rajeunir les clients (Date of Birth +10 ans)
if 'Date of Birth' in part3.columns:
    part3['Date of Birth'] = pd.to_datetime(part3['Date of Birth'], errors='coerce')
    part3['Date of Birth'] = part3['Date of Birth'] + pd.DateOffset(years=10)
    print("   ✓ Date of Birth: +10 ans")

# 2. Augmenter le niveau d'éducation
if 'Education Level' in part3.columns:
    mask = (part3['Education Level'] == 'High School') & (np.random.random(len(part3)) < 0.4)
    part3.loc[mask, 'Education Level'] = 'Bachelor'
    mask = (part3['Education Level'] == 'Bachelor') & (np.random.random(len(part3)) < 0.3)
    part3.loc[mask, 'Education Level'] = 'Master'
    print("   ✓ Education Level: Augmentation diplômes")

# 3. Augmenter les produits numériques
if 'NumOfProducts' in part3.columns:
    mask = (part3['NumOfProducts'] < 4) & (np.random.random(len(part3)) < 0.4)
    part3.loc[mask, 'NumOfProducts'] = part3.loc[mask, 'NumOfProducts'] + 1
    print("   ✓ NumOfProducts: +1 pour 40%")

# 4. Réduire la tenure (nouveaux clients)
if 'Customer Tenure' in part3.columns:
    part3['Customer Tenure'] = part3['Customer Tenure'] * 0.6
    part3['Customer Tenure'] = part3['Customer Tenure'].clip(0, None).astype(int)
    print("   ✓ Customer Tenure: -40%")

# 5. Augmenter les revenus (jeunes diplômés)
if 'Income' in part3.columns:
    part3['Income'] = part3['Income'] * 1.15
    print("   ✓ Income: +15%")

# 6. Modifier le statut marital
if 'Marital Status' in part3.columns:
    mask = (part3['Marital Status'] == 'Married') & (np.random.random(len(part3)) < 0.3)
    part3.loc[mask, 'Marital Status'] = 'Single'
    print("   ✓ Marital Status: +30% Single")

part3_file = f"{OUTPUT_DIR}/batch2.csv"
part3.to_csv(part3_file, index=False)
print(f"✅ Sauvegardée: {part3_file}")

# ============================================================================
# STATISTIQUES
# ============================================================================
print("\n" + "=" * 80)
print("STATISTIQUES")
print("=" * 80)

churn_col = 'Churn Flag' if 'Churn Flag' in df.columns else 'Churn'

print(f"\nPart 1 (Référence):")
print(f"  Lignes: {len(part1):,}")
if churn_col in part1.columns:
    print(f"  Churn rate: {part1[churn_col].mean():.2%}")

print(f"\nPart 2 (Batch 1 - Crise):")
print(f"  Lignes: {len(part2):,}")
if churn_col in part2.columns:
    print(f"  Churn rate: {part2[churn_col].mean():.2%}")
if 'Income' in part2.columns:
    print(f"  Income moyen: {part2['Income'].mean():,.0f}")

print(f"\nPart 3 (Batch 2 - Démographie):")
print(f"  Lignes: {len(part3):,}")
if churn_col in part3.columns:
    print(f"  Churn rate: {part3[churn_col].mean():.2%}")
if 'Income' in part3.columns:
    print(f"  Income moyen: {part3['Income'].mean():,.0f}")

print("\n" + "=" * 80)
print("✅ DIVISION ET TRANSFORMATION TERMINÉES")
print("=" * 80)
print(f"\n📁 Fichiers créés:")
print(f"   {part1_file} (référence)")
print(f"   {part2_file} (crise économique)")
print(f"   {part3_file} (changement démographique)")
print("\n💡 Ces fichiers peuvent maintenant être commités dans Git")
print("💡 Jenkins copiera automatiquement ces batches transformés")
print("=" * 80)
