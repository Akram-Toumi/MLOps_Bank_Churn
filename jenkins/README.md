# Jenkins CI/CD pour MLOps

Ce dossier contient la configuration Jenkins pour le pipeline MLOps automatisé.

## 📋 Prérequis

### Installation de Jenkins

1. **Télécharger Jenkins**:
   ```bash
   # Windows
   # Télécharger depuis: https://www.jenkins.io/download/
   ```

2. **Installer Jenkins**:
   - Exécuter l'installateur
   - Choisir le port (par défaut: 8080)
   - Installer les plugins recommandés

3. **Plugins requis**:
   - Pipeline
   - Git
   - HTML Publisher
   - Email Extension (optionnel)

### Configuration du Job

1. **Créer un nouveau Pipeline**:
   - Nouveau Item → Pipeline
   - Nom: `MLOps-Bank-Churn-Monitoring`

2. **Configuration du Pipeline**:
   - Definition: Pipeline script from SCM
   - SCM: Git
   - Repository URL: `<votre-repo>`
   - Script Path: `jenkins/Jenkinsfile`

3. **Déclencheurs**:
   - ✅ Build périodiquement: `0 2 * * *` (tous les jours à 2h)
   - ✅ Poll SCM (optionnel)

## 🔄 Pipeline Stages

### Stage 1: Data Drift Monitoring
- Exécute `monitoring/run_monitoring.py`
- Génère rapport HTML Evidently
- Archive les métriques JSON

### Stage 2: Check Data Drift
- Vérifie l'existence de `trigger.txt`
- Lit les métriques de drift
- Décide des actions suivantes

### Stage 3: DVC Versioning
- **Condition**: Exécuté seulement si drift détecté
- Initialise DVC si nécessaire
- Versionne les données de production
- Commit et push vers remote DVC

### Stage 4: Notification
- Affiche les alertes dans les logs
- Envoie notifications (email/Slack)
- Archive les rapports

## 📊 Rapports Générés

- **Evidently Report**: Rapport HTML interactif
- **Metrics JSON**: Métriques exportées
- **Trigger File**: Fichier d'alerte si drift

## 🚀 Exécution Manuelle

### Sans Jenkins

Si Jenkins n'est pas installé, vous pouvez exécuter manuellement:

```bash
# 1. Monitoring
python monitoring/run_monitoring.py

# 2. Vérifier le drift
if exist trigger.txt (
    echo "Drift détecté!"
    type trigger.txt
)

# 3. DVC (si drift)
dvc add data/production/bank_churn_prod.csv
git add data/production/bank_churn_prod.csv.dvc
git commit -m "DVC: Version après drift"
dvc push
```

### Avec Jenkins

1. Aller sur Jenkins: `http://localhost:8080`
2. Sélectionner le job `MLOps-Bank-Churn-Monitoring`
3. Cliquer sur "Build Now"
4. Consulter les logs et rapports

## 🔧 Configuration

### Variables d'Environnement

Dans le Jenkinsfile:
```groovy
environment {
    PYTHON_ENV = "${PROJECT_DIR}/.venv/Scripts/python.exe"
    DRIFT_THRESHOLD = "0.1"
}
```

### Déclencheurs

```groovy
triggers {
    cron('0 2 * * *')  // Tous les jours à 2h
}
```

## 📝 Logs et Debugging

### Consulter les logs
- Jenkins UI → Job → Build History → Console Output

### Rapports archivés
- Jenkins UI → Job → Build → Artifacts

### Rapport Evidently
- Jenkins UI → Job → Evidently Data Drift Report

## ⚠️ Troubleshooting

### Erreur: Python not found
```bash
# Vérifier le chemin Python dans Jenkinsfile
environment {
    PYTHON_ENV = "C:/path/to/python.exe"
}
```

### Erreur: DVC not initialized
```bash
# Initialiser DVC manuellement
dvc init
dvc remote add -d local_storage ./dvc_storage
```

### Erreur: Permission denied
```bash
# Donner les droits à Jenkins
# Windows: Exécuter Jenkins en tant qu'administrateur
```

## 📚 Ressources

- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Pipeline Syntax](https://www.jenkins.io/doc/book/pipeline/syntax/)
- [DVC Documentation](https://dvc.org/doc)
- [Evidently AI](https://docs.evidentlyai.com/)

## 🎯 Prochaines Étapes

1. ✅ Installer Jenkins
2. ✅ Configurer le job
3. ✅ Tester le pipeline
4. ⏳ Configurer les notifications
5. ⏳ Intégrer avec MLflow
