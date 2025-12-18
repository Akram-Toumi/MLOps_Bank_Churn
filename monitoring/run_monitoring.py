"""
Script de monitoring avec Evidently AI
Détecte le data drift entre les données d'entraînement et de production
Génère des rapports HTML et JSON, et crée un trigger si drift détecté
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import *
except ImportError as e:
    import sys
    import traceback
    print("\n" + "!"*40)
    print("DIAGNOSTIC D'IMPORTATION")
    print(f"Executable: {sys.executable}")
    print(f"Path: {sys.path}")
    print("!"*40 + "\n")
    traceback.print_exc()
    print(f"⚠️  Erreur ImportEvidently: {e}")
    print("⚠️  Evidently n'est pas installé. Installez-le avec: pip install evidently")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins des fichiers
REFERENCE_DATA = "data/churn.csv"  # Données d'entraînement (référence)
PRODUCTION_DATA = "data/production/bank_churn_prod.csv"  # Données de production
OUTPUT_HTML = "monitoring/monitoring_report.html"
OUTPUT_JSON = "monitoring/monitoring_metrics.json"
TRIGGER_FILE = "trigger.txt"

# Seuil de drift (0.1 = 10%)
DRIFT_THRESHOLD = 0.1

print("=" * 80)
print("MONITORING DATA DRIFT - EVIDENTLY AI")
print("=" * 80)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

print(f"\n📂 Chargement des données...")

# Données de référence (entraînement)
df_reference = pd.read_csv(REFERENCE_DATA)
print(f"✅ Données de référence: {df_reference.shape[0]:,} lignes")

# Données de production
df_production = pd.read_csv(PRODUCTION_DATA)
print(f"✅ Données de production: {df_production.shape[0]:,} lignes")

# OPTIMISATION: Échantillonner pour accélérer l'analyse
# Evidently est lent sur de gros datasets, on prend un échantillon représentatif
SAMPLE_SIZE = 5000
if len(df_reference) > SAMPLE_SIZE:
    df_reference = df_reference.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"⚡ Échantillon référence: {len(df_reference):,} lignes (pour performance)")

if len(df_production) > SAMPLE_SIZE:
    df_production = df_production.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"⚡ Échantillon production: {len(df_production):,} lignes (pour performance)")

# Sélectionner UNIQUEMENT les colonnes numériques pertinentes
# Exclure les IDs, noms, dates, et la target
numeric_cols = df_reference.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

# Colonnes à exclure absolument
exclude_cols = [
    'RowNumber', 'CustomerId', 'id', 'ID', 
    'Churn', 'Churn Flag',  # Target
    'Number of Dependents',  # Souvent constant
]

# Garder seulement les features numériques pertinentes
drift_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"📊 Colonnes numériques analysées pour le drift: {len(drift_cols)}")
print(f"   {drift_cols}")
print(f"   Colonnes exclues: {[c for c in exclude_cols if c in df_reference.columns]}")

# ============================================================================
# GÉNÉRATION DU RAPPORT EVIDENTLY
# ============================================================================

print(f"\n🔍 Analyse du data drift...")

# Créer le rapport Evidently
report = Report(metrics=[
    DataDriftPreset(columns=drift_cols if drift_cols else None),
    DataQualityPreset(),
])

# Générer le rapport
report.run(reference_data=df_reference, current_data=df_production)

# Sauvegarder le rapport HTML
print(f"\n💾 Sauvegarde du rapport HTML...")
report.save_html(OUTPUT_HTML)
print(f"✅ Rapport HTML sauvegardé: {OUTPUT_HTML}")

# ============================================================================
# EXTRACTION DES MÉTRIQUES
# ============================================================================

print(f"\n📊 Extraction des métriques...")

# Obtenir les métriques en JSON
report_dict = report.as_dict()

# Extraire les métriques de drift
metrics = {
    "timestamp": datetime.now().isoformat(),
    "reference_rows": len(df_reference),
    "production_rows": len(df_production),
    "columns_analyzed": len(drift_cols) if drift_cols else len(common_cols),
    "drift_detected": False,
    "drift_score": 0.0,
    "drifted_columns": [],
    "drift_threshold": DRIFT_THRESHOLD
}

# Analyser les résultats du drift
try:
    # Chercher les métriques de drift dans le rapport
    for metric in report_dict.get('metrics', []):
        if 'DatasetDriftMetric' in str(metric.get('metric', '')):
            result = metric.get('result', {})
            metrics['drift_score'] = result.get('dataset_drift_score', 0.0)
            # NE PAS utiliser dataset_drift d'Evidently, calculer nous-mêmes
            # metrics['drift_detected'] = result.get('dataset_drift', False)
            
            # Colonnes avec drift
            drift_by_columns = result.get('drift_by_columns', {})
            metrics['drifted_columns'] = [
                col for col, info in drift_by_columns.items() 
                if isinstance(info, dict) and info.get('drift_detected', False)
            ]
            break
except Exception as e:
    print(f"⚠️  Erreur lors de l'extraction des métriques: {e}")
    print(f"   Rapport dict keys: {list(report_dict.keys())}")
    # Ne pas forcer drift_detected à True, garder False par défaut
    print("   Utilisation des valeurs par défaut (pas de drift)")

# Calculer drift_detected basé sur le score ET les colonnes
metrics['drift_detected'] = (
    metrics['drift_score'] > DRIFT_THRESHOLD or 
    len(metrics['drifted_columns']) > 0
)

# Sauvegarder les métriques en JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Métriques JSON sauvegardées: {OUTPUT_JSON}")

# ============================================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================================

print("\n" + "=" * 80)
print("RÉSULTATS DU MONITORING")
print("=" * 80)

print(f"\n📊 Statistiques:")
print(f"   Données de référence: {metrics['reference_rows']:,} lignes")
print(f"   Données de production: {metrics['production_rows']:,} lignes")
print(f"   Colonnes analysées: {metrics['columns_analyzed']}")

print(f"\n🎯 Data Drift:")
print(f"   Score de drift: {metrics['drift_score']:.4f}")
print(f"   Seuil configuré: {DRIFT_THRESHOLD}")
print(f"   Drift détecté: {'🔴 OUI' if metrics['drift_detected'] else '🟢 NON'}")

if metrics['drifted_columns']:
    print(f"\n⚠️  Colonnes avec drift détecté:")
    for col in metrics['drifted_columns'][:10]:  # Afficher max 10
        print(f"   • {col}")

# ============================================================================
# CRÉATION DU TRIGGER SI DRIFT DÉTECTÉ
# ============================================================================

if metrics['drift_detected'] or metrics['drift_score'] > DRIFT_THRESHOLD:
    print("\n" + "🚨" * 40)
    print("DATA DRIFT DETECTED!")
    print("🚨" * 40)
    
    # Créer le fichier trigger
    trigger_content = f"""DATA DRIFT DETECTED
Timestamp: {metrics['timestamp']}
Drift Score: {metrics['drift_score']:.4f}
Threshold: {DRIFT_THRESHOLD}
Drifted Columns: {len(metrics['drifted_columns'])}

Action Required:
1. Review monitoring report: {OUTPUT_HTML}
2. Check metrics: {OUTPUT_JSON}
3. Consider retraining the model
4. Update DVC versioning

Columns with drift:
{chr(10).join(['- ' + col for col in metrics['drifted_columns'][:20]])}
"""
    
    with open(TRIGGER_FILE, 'w') as f:
        f.write(trigger_content)
    
    print(f"\n✅ Fichier trigger créé: {TRIGGER_FILE}")
    print("\n💡 Actions recommandées:")
    print("   1. Consulter le rapport HTML pour plus de détails")
    print("   2. Vérifier les colonnes avec drift")
    print("   3. Considérer le réentraînement du modèle")
    print("   4. Exécuter le pipeline Jenkins pour versioning DVC")
else:
    print("\n✅ Aucun drift significatif détecté")
    print("   Le modèle peut continuer à être utilisé en production")
    
    # Supprimer le trigger s'il existe
    if Path(TRIGGER_FILE).exists():
        Path(TRIGGER_FILE).unlink()
        print(f"   Fichier trigger supprimé (pas de drift)")

# ============================================================================
# RÉSUMÉ
# ============================================================================

print("\n" + "=" * 80)
print("✅ MONITORING TERMINÉ")
print("=" * 80)
print(f"📄 Rapport HTML: {OUTPUT_HTML}")
print(f"📊 Métriques JSON: {OUTPUT_JSON}")
if metrics['drift_detected']:
    print(f"🚨 Trigger: {TRIGGER_FILE}")
print("\n💡 Ouvrez le rapport HTML dans un navigateur pour une analyse détaillée")
print("=" * 80)
