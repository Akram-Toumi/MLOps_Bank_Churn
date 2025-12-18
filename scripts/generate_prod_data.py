"""
Script pour copier les batches pré-transformés vers production
Les transformations de drift sont déjà appliquées dans batch1.csv et batch2.csv
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Configuration
BATCH_DIR = "data/batches"
OUTPUT_FILE = "data/production/bank_churn_prod.csv"

# Déterminer quel batch utiliser
if len(sys.argv) > 1:
    batch_name = sys.argv[1]  # "batch1" ou "batch2"
else:
    batch_name = "batch1"

INPUT_FILE = f"{BATCH_DIR}/{batch_name}.csv"

print("=" * 80)
print("COPIE DU BATCH VERS PRODUCTION")
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

# Créer le dossier de sortie
Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)

# Copie simple
df_prod.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Copié vers: {OUTPUT_FILE}")

print("\n" + "=" * 80)
print("✅ COPIE TERMINÉE")
print("=" * 80)
print(f"🔄 Batch: {batch_name}")
print(f"📊 Lignes: {len(df_prod):,}")
print("\n💡 Prochaine étape: python monitoring/run_monitoring.py")
print("=" * 80)
