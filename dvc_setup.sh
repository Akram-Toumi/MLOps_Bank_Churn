#!/bin/bash
# Script d'initialisation et configuration DVC
# Utilise un remote local pour la démonstration

echo "================================================================================"
echo "INITIALISATION DVC (Data Version Control)"
echo "================================================================================"

# ============================================================================
# ÉTAPE 1: Initialiser DVC
# ============================================================================

echo -e "\n📦 Initialisation de DVC..."

# Initialiser DVC dans le projet
dvc init --force

if [ $? -eq 0 ]; then
    echo "✅ DVC initialisé avec succès"
else
    echo "⚠️  DVC déjà initialisé ou erreur"
fi

# ============================================================================
# ÉTAPE 2: Configurer le remote local
# ============================================================================

echo -e "\n🔧 Configuration du remote storage local..."

# Créer le dossier de stockage DVC
mkdir -p ./dvc_storage

# Ajouter le remote local
dvc remote add -d local_storage ./dvc_storage --force

echo "✅ Remote local configuré: ./dvc_storage"

# ============================================================================
# ÉTAPE 3: Versionner les données de production
# ============================================================================

echo -e "\n📊 Versioning des données de production..."

# Vérifier si le fichier existe
if [ -f "data/production/bank_churn_prod.csv" ]; then
    # Ajouter les données de production à DVC
    dvc add data/production/bank_churn_prod.csv
    
    echo "✅ Données de production ajoutées à DVC"
    echo "   Fichier .dvc créé: data/production/bank_churn_prod.csv.dvc"
else
    echo "⚠️  Fichier data/production/bank_churn_prod.csv non trouvé"
    echo "   Exécutez d'abord: python scripts/generate_prod_data.py"
fi

# ============================================================================
# ÉTAPE 4: Commit Git
# ============================================================================

echo -e "\n💾 Commit des métadonnées DVC dans Git..."

# Ajouter les fichiers DVC à Git
git add .dvc/config .dvc/.gitignore
git add data/production/.gitignore
git add data/production/bank_churn_prod.csv.dvc 2>/dev/null || true

# Commit
git commit -m "DVC: Initialize and version production data" || echo "Rien à commiter"

echo "✅ Métadonnées DVC commitées"

# ============================================================================
# ÉTAPE 5: Push vers le remote
# ============================================================================

echo -e "\n☁️  Push des données vers le remote DVC..."

dvc push

if [ $? -eq 0 ]; then
    echo "✅ Données pushées vers le remote local"
else
    echo "⚠️  Erreur lors du push DVC"
fi

# ============================================================================
# RÉSUMÉ
# ============================================================================

echo -e "\n================================================================================"
echo "✅ CONFIGURATION DVC TERMINÉE"
echo "================================================================================"
echo ""
echo "📁 Structure DVC:"
echo "   .dvc/              - Configuration DVC"
echo "   dvc_storage/       - Stockage local des données"
echo "   *.dvc              - Métadonnées des fichiers versionnés"
echo ""
echo "🔄 Commandes DVC utiles:"
echo "   dvc status         - Vérifier l'état"
echo "   dvc diff           - Voir les différences"
echo "   dvc pull           - Récupérer les données"
echo "   dvc push           - Envoyer les données"
echo ""
echo "💡 Prochaine étape: Exécuter le monitoring"
echo "   python monitoring/run_monitoring.py"
echo "================================================================================"
