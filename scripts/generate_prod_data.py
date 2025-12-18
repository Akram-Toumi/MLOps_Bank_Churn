"""
Script pour générer les données de production à partir des batches réels
Au lieu de créer un drift synthétique, on charge les vrais batches
"""

import pandas as pd
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
print("GÉNÉRATION DES DONNÉES DE PRODUCTION")
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
print(f"📋 Colonnes: {list(df_prod.columns[:10])}")

# Créer le dossier de sortie
Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)

# Sauvegarde
print(f"\n💾 Sauvegarde...")
df_prod.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Fichier sauvegardé: {OUTPUT_FILE}")
print(f"   Taille: {len(df_prod):,} lignes × {len(df_prod.columns)} colonnes")

# Statistiques
if 'Churn Flag' in df_prod.columns:
    churn_rate = df_prod['Churn Flag'].mean()
    print(f"\n📊 Statistiques:")
    print(f"   Churn rate: {churn_rate:.2%}")

print("\n" + "=" * 80)
print("✅ GÉNÉRATION TERMINÉE")
print("=" * 80)
print(f"📁 Fichier: {OUTPUT_FILE}")
print(f"📊 Lignes: {len(df_prod):,}")
print(f"🔄 Batch: {batch_name}")
print("\n💡 Prochaine étape: python monitoring/run_monitoring.py")
print("=" * 80)
