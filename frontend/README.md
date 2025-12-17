# Frontend Streamlit - Bank Churn Prediction

Interface utilisateur moderne pour la prédiction de churn bancaire.

## 📋 Prérequis

- Python 3.8+
- Backend FastAPI en cours d'exécution sur `http://127.0.0.1:8000`

## 🚀 Installation

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## ▶️ Lancement de l'Application

```bash
streamlit run frontend/app.py
```

L'application sera accessible sur : **http://localhost:8501**

## 📱 Fonctionnalités

### Interface Utilisateur
- ✅ Design moderne et élégant avec CSS personnalisé
- ✅ Formulaire organisé en 3 colonnes thématiques
- ✅ Validation des entrées utilisateur
- ✅ Indicateur de connexion API

### Informations Collectées

#### 👤 Informations Personnelles
- Genre
- Âge
- Statut marital
- Niveau d'éducation
- Profession
- Nombre de personnes à charge

#### 💰 Informations Financières
- Score de crédit (300-850)
- Revenu annuel
- Solde du compte
- Prêts en cours
- Salaire estimé

#### 🏦 Informations Bancaires
- Pays de résidence
- Ancienneté (en mois)
- Nombre de produits
- Possession de carte de crédit
- Statut de membre actif
- Segment client
- Canal de communication préféré
- Nombre de plaintes

### Résultats Affichés

#### Si Risque de Churn Élevé (prediction = 1)
- 🔴 Carte d'alerte rouge
- Probabilité de churn en %
- Recommandations d'actions immédiates
- Recommandations d'actions préventives

#### Si Client Fidèle (prediction = 0)
- 🟢 Carte de succès verte
- Probabilité de churn en %
- Recommandations de maintien de relation

### Gestion des Erreurs

L'application gère automatiquement:
- ✅ Perte de connexion avec l'API
- ✅ Erreurs de validation
- ✅ Timeouts de requête
- ✅ Erreurs serveur

## 🎨 Personnalisation

### Modifier l'URL de l'API

Dans `app.py`, ligne 13:
```python
API_URL = "http://127.0.0.1:8000"
```

### Modifier les Styles

Les styles CSS sont définis dans la section "STYLES CSS PERSONNALISÉS" du fichier `app.py`.

## 📊 Exemple d'Utilisation

1. Lancez le backend FastAPI:
```bash
uvicorn backend.api:app --reload
```

2. Lancez le frontend Streamlit:
```bash
streamlit run frontend/app.py
```

3. Ouvrez votre navigateur sur `http://localhost:8501`

4. Remplissez le formulaire avec les informations du client

5. Cliquez sur "🔮 Prédire le Risque de Churn"

6. Consultez les résultats et recommandations

## 🔧 Dépannage

### L'application ne se connecte pas à l'API

**Vérifications:**
1. L'API backend est-elle lancée ?
   ```bash
   curl http://127.0.0.1:8000/health
   ```

2. Le port 8000 est-il accessible ?

3. L'URL dans `app.py` est-elle correcte ?

### Erreur lors du lancement

**Solution:**
```bash
# Réinstaller les dépendances
pip install --upgrade streamlit requests pandas
```

### L'interface ne s'affiche pas correctement

**Solution:**
- Vider le cache de Streamlit: `Ctrl + C` puis relancer
- Rafraîchir la page du navigateur: `Ctrl + F5`

## 📝 Notes

- L'application envoie toutes les données au backend qui gère le feature engineering
- Aucun traitement ML n'est effectué côté frontend
- Les prédictions sont en temps réel
- L'interface est responsive et s'adapte à différentes tailles d'écran

## 🎯 Améliorations Futures Possibles

- [ ] Ajout d'un mode batch pour prédictions multiples
- [ ] Export des résultats en PDF
- [ ] Historique des prédictions
- [ ] Graphiques de visualisation
- [ ] Mode sombre/clair
- [ ] Authentification utilisateur
