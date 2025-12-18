"""
Script pour diviser le dataset en 3 parties pour la simulation de data drift
Part 1: 0-30,000 (déjà utilisé pour l'entraînement initial)
Part 2: 30,000-60,000 (Batch 1 - premier drift)
Part 3: 60,000-fin (Batch 2 - deuxième drift)
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
INPUT_FILE = "data/bank_customer_churn.csv"
OUTPUT_DIR = "data/batches"

# Points de division
PART1_END = 30000
PART2_END = 60000

print("=" * 80)
print("DIVISION DU DATASET EN 3 PARTIES")
print("=" * 80)

# Créer le dossier de sortie
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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

part1 = df.iloc[:PART1_END]
part2 = df.iloc[PART1_END:PART2_END]
part3 = df.iloc[PART2_END:]

# Sauvegarder
print(f"\n💾 Sauvegarde des parties...")

# Part 1 (référence - déjà utilisée pour training)
part1_file = "data/train/part1.csv"
Path("data/train").mkdir(parents=True, exist_ok=True)
part1.to_csv(part1_file, index=False)
print(f"✅ Part 1 sauvegardée: {part1_file}")

# Part 2 (Batch 1)
part2_file = f"{OUTPUT_DIR}/batch1.csv"
part2.to_csv(part2_file, index=False)
print(f"✅ Part 2 sauvegardée: {part2_file}")

# Part 3 (Batch 2)
part3_file = f"{OUTPUT_DIR}/batch2.csv"
part3.to_csv(part3_file, index=False)
print(f"✅ Part 3 sauvegardée: {part3_file}")

# Statistiques
print("\n" + "=" * 80)
print("STATISTIQUES")
print("=" * 80)
print(f"\nPart 1 (Training initial):")
print(f"  Lignes: {len(part1):,}")
print(f"  Churn rate: {part1['Churn Flag'].mean():.2%}" if 'Churn Flag' in part1.columns else "")

print(f"\nPart 2 (Batch 1):")
print(f"  Lignes: {len(part2):,}")
print(f"  Churn rate: {part2['Churn Flag'].mean():.2%}" if 'Churn Flag' in part2.columns else "")

print(f"\nPart 3 (Batch 2):")
print(f"  Lignes: {len(part3):,}")
print(f"  Churn rate: {part3['Churn Flag'].mean():.2%}" if 'Churn Flag' in part3.columns else "")

print("\n" + "=" * 80)
print("✅ DIVISION TERMINÉE")
print("=" * 80)
print(f"\n📁 Fichiers créés:")
print(f"   {part1_file}")
print(f"   {part2_file}")
print(f"   {part3_file}")
print("\n💡 Prochaine étape: Utiliser batch1.csv pour simuler le premier drift")
print("=" * 80)
