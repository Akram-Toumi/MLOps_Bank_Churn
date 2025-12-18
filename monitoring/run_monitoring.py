"""
Script de monitoring avec détection de drift robuste
Utilise des tests statistiques au lieu de se fier uniquement à Evidently
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
from datetime import datetime

# Configuration
REFERENCE_DATA = "data/train/part1.csv"
PRODUCTION_DATA = "data/production/bank_churn_prod.csv"
OUTPUT_HTML = "monitoring/monitoring_report.html"
OUTPUT_JSON = "monitoring/monitoring_metrics.json"
TRIGGER_FILE = "trigger.txt"
DRIFT_THRESHOLD = 0.1

print("=" * 80)
print("MONITORING DATA DRIFT - TESTS STATISTIQUES")
print("=" * 80)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

print(f"\n📂 Chargement des données...")

df_reference = pd.read_csv(REFERENCE_DATA)
print(f"✅ Données de référence: {df_reference.shape[0]:,} lignes")

df_production = pd.read_csv(PRODUCTION_DATA)
print(f"✅ Données de production: {df_production.shape[0]:,} lignes")

# Échantillonner pour performance
SAMPLE_SIZE = 5000
if len(df_reference) > SAMPLE_SIZE:
    df_reference = df_reference.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"⚡ Échantillon référence: {len(df_reference):,} lignes")

if len(df_production) > SAMPLE_SIZE:
    df_production = df_production.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"⚡ Échantillon production: {len(df_production):,} lignes")

# Sélectionner colonnes numériques
numeric_cols = df_reference.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

exclude_cols = ['RowNumber', 'CustomerId', 'id', 'ID', 'Churn', 'Churn Flag', 'Number of Dependents']
drift_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"\n📊 Colonnes numériques analysées: {len(drift_cols)}")
print(f"   {drift_cols}")

# ============================================================================
# DÉTECTION DE DRIFT PAR TESTS STATISTIQUES
# ============================================================================

print(f"\n🔍 Analyse du drift par tests statistiques...")

drift_results = {}
drifted_columns = []

for col in drift_cols:
    if col not in df_production.columns:
        continue
    
    ref_values = df_reference[col].dropna()
    prod_values = df_production[col].dropna()
    
    # Test de Kolmogorov-Smirnov
    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, prod_values)
    
    # Calculer l'effet (différence relative des moyennes)
    ref_mean = ref_values.mean()
    prod_mean = prod_values.mean()
    effect_size = abs(prod_mean - ref_mean) / (ref_mean + 1e-10)
    
    # Drift détecté si p-value < 0.05 ET effet > 5%
    drift_detected = (ks_pvalue < 0.05) and (effect_size > 0.05)
    
    drift_results[col] = {
        'ks_statistic': float(ks_stat),
        'p_value': float(ks_pvalue),
        'effect_size': float(effect_size),
        'drift_detected': bool(drift_detected),  # Convert to Python bool
        'ref_mean': float(ref_mean),
        'prod_mean': float(prod_mean),
        'change_pct': float((prod_mean - ref_mean) / ref_mean * 100)
    }
    
    if drift_detected:
        drifted_columns.append(col)
        print(f"   🔴 {col}: KS={ks_stat:.3f}, p={ks_pvalue:.4f}, effet={effect_size:.1%}, Δ={drift_results[col]['change_pct']:+.1f}%")
    else:
        print(f"   🟢 {col}: KS={ks_stat:.3f}, p={ks_pvalue:.4f}, effet={effect_size:.1%}")

# Calculer le score de drift global (moyenne des effect sizes)
drift_score = np.mean([r['effect_size'] for r in drift_results.values()])
drift_detected_global = len(drifted_columns) > 0

# ============================================================================
# GÉNÉRATION DU RAPPORT HTML
# ============================================================================

print(f"\n💾 Génération du rapport HTML...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .drift {{ color: red; font-weight: bold; }}
        .no-drift {{ color: green; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .drift-row {{ background-color: #ffebee; }}
    </style>
</head>
<body>
    <h1>📊 Data Drift Monitoring Report</h1>
    <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Reference samples: {len(df_reference):,}</p>
        <p>Production samples: {len(df_production):,}</p>
        <p>Columns analyzed: {len(drift_cols)}</p>
        <p>Drift score: <strong>{drift_score:.4f}</strong></p>
        <p>Threshold: {DRIFT_THRESHOLD}</p>
        <p>Status: <span class="{'drift' if drift_detected_global else 'no-drift'}">
            {'🔴 DRIFT DETECTED' if drift_detected_global else '🟢 NO DRIFT'}
        </span></p>
        <p>Drifted columns: {len(drifted_columns)}</p>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Column</th>
            <th>KS Statistic</th>
            <th>P-Value</th>
            <th>Effect Size</th>
            <th>Ref Mean</th>
            <th>Prod Mean</th>
            <th>Change %</th>
            <th>Drift</th>
        </tr>
"""

for col, result in drift_results.items():
    row_class = 'drift-row' if result['drift_detected'] else ''
    drift_icon = '🔴' if result['drift_detected'] else '🟢'
    html_content += f"""
        <tr class="{row_class}">
            <td>{col}</td>
            <td>{result['ks_statistic']:.4f}</td>
            <td>{result['p_value']:.4f}</td>
            <td>{result['effect_size']:.2%}</td>
            <td>{result['ref_mean']:.2f}</td>
            <td>{result['prod_mean']:.2f}</td>
            <td>{result['change_pct']:+.1f}%</td>
            <td>{drift_icon}</td>
        </tr>
    """

html_content += """
    </table>
</body>
</html>
"""

Path(os.path.dirname(OUTPUT_HTML)).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Rapport HTML sauvegardé: {OUTPUT_HTML}")

# ============================================================================
# SAUVEGARDE DES MÉTRIQUES JSON
# ============================================================================

metrics = {
    "timestamp": datetime.now().isoformat(),
    "reference_rows": len(df_reference),
    "production_rows": len(df_production),
    "columns_analyzed": len(drift_cols),
    "drift_detected": drift_detected_global,
    "drift_score": float(drift_score),
    "drifted_columns": drifted_columns,
    "drift_threshold": DRIFT_THRESHOLD,
    "detailed_results": drift_results
}

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
print(f"   Données de référence: {len(df_reference):,} lignes")
print(f"   Données de production: {len(df_production):,} lignes")
print(f"   Colonnes analysées: {len(drift_cols)}")

print(f"\n🎯 Data Drift:")
print(f"   Score de drift: {drift_score:.4f}")
print(f"   Seuil configuré: {DRIFT_THRESHOLD}")
print(f"   Drift détecté: {'🔴 OUI' if drift_detected_global else '🟢 NON'}")

if drifted_columns:
    print(f"\n⚠️  Colonnes avec drift détecté ({len(drifted_columns)}):")
    for col in drifted_columns:
        r = drift_results[col]
        print(f"   • {col}: {r['change_pct']:+.1f}% (p={r['p_value']:.4f})")

# ============================================================================
# CRÉATION DU TRIGGER SI DRIFT DÉTECTÉ
# ============================================================================

if drift_detected_global:
    print("\n" + "🚨" * 40)
    print("DATA DRIFT DETECTED!")
    print("🚨" * 40)
    
    trigger_content = f"""DATA DRIFT DETECTED
Timestamp: {datetime.now().isoformat()}
Drift Score: {drift_score:.4f}
Threshold: {DRIFT_THRESHOLD}
Drifted Columns: {len(drifted_columns)}

Action Required:
1. Review monitoring report: {OUTPUT_HTML}
2. Check metrics: {OUTPUT_JSON}
3. Retraining will be triggered automatically

Columns with drift:
"""
    for col in drifted_columns:
        r = drift_results[col]
        trigger_content += f"- {col}: {r['change_pct']:+.1f}%\n"
    
    with open(TRIGGER_FILE, 'w') as f:
        f.write(trigger_content)
    
    print(f"\n✅ Fichier trigger créé: {TRIGGER_FILE}")
else:
    print(f"\n✅ Aucun drift significatif détecté")
    if os.path.exists(TRIGGER_FILE):
        os.remove(TRIGGER_FILE)

print("\n" + "=" * 80)
print("✅ MONITORING TERMINÉ")
print("=" * 80)
