# Backend FastAPI - Bank Churn Prediction

API REST pour la prédiction de churn bancaire utilisant MLflow pour charger automatiquement le meilleur modèle entraîné.

## 📋 Prérequis

- Python 3.8+
- MLflow tracking server en cours d'exécution sur `http://127.0.0.1:5000`
- Modèles entraînés disponibles dans MLflow

## 🚀 Installation

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## ▶️ Lancement de l'API

### Méthode 1 : Avec uvicorn (recommandé pour le développement)

```bash
uvicorn backend.api:app --reload
```

L'API sera accessible sur : `http://127.0.0.1:8000`

### Méthode 2 : Exécution directe

```bash
python backend/api.py
```

## 📡 Endpoints disponibles

### 1. **GET /** - Informations sur l'API
Retourne les informations générales sur l'API.

```bash
curl http://127.0.0.1:8000/
```

### 2. **GET /health** - Vérification de santé
Vérifie que l'API et le modèle sont opérationnels.

```bash
curl http://127.0.0.1:8000/health
```

### 3. **GET /model-info** - Informations sur le modèle
Retourne les détails du modèle chargé (run_id, accuracy, etc.).

```bash
curl http://127.0.0.1:8000/model-info
```

### 4. **POST /predict** - Prédiction de churn
Endpoint principal pour faire des prédictions.

#### Format de la requête

```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```

#### Exemple avec curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

#### Exemple avec PowerShell

```powershell
$body = @{
    CreditScore = 619
    Geography = "France"
    Gender = "Female"
    Age = 42
    Tenure = 2
    Balance = 0.0
    NumOfProducts = 1
    HasCrCard = 1
    IsActiveMember = 1
    EstimatedSalary = 101348.88
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

#### Format de la réponse

```json
{
  "prediction": 1,
  "probability": 0.7234,
  "model_version": "abc123def456"
}
```

- `prediction` : 0 (client reste) ou 1 (client churn)
- `probability` : Probabilité de churn (entre 0 et 1)
- `model_version` : ID du run MLflow utilisé

## 📚 Documentation interactive

Une fois l'API lancée, accédez à la documentation Swagger interactive :

- **Swagger UI** : http://127.0.0.1:8000/docs
- **ReDoc** : http://127.0.0.1:8000/redoc

## 🔧 Configuration

L'API se configure automatiquement pour :
- Se connecter à MLflow sur `http://127.0.0.1:5000`
- Charger le meilleur modèle de l'expérience `churn_prediction`
- Trier les modèles par accuracy décroissante

Pour modifier ces paramètres, éditez les constantes dans `api.py` :

```python
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "churn_prediction"
```

## ⚠️ Dépannage

### Erreur : "Modèle non chargé"
- Vérifiez que MLflow tracking server est en cours d'exécution
- Vérifiez que des modèles existent dans l'expérience `churn_prediction`

### Erreur de connexion MLflow
- Assurez-vous que MLflow UI est accessible sur `http://127.0.0.1:5000`
- Lancez MLflow avec : `mlflow ui --port 5000`

## 📝 Notes

- Le preprocessing est automatiquement géré par le modèle MLflow (s'il a été sauvegardé avec)
- L'API charge automatiquement le modèle avec la meilleure accuracy au démarrage
- Les logs sont affichés dans la console pour faciliter le débogage
