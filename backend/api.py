"""
Backend FastAPI pour le projet MLOps Bank Churn
Charge automatiquement le meilleur modèle depuis les fichiers pickle et expose un endpoint de prédiction
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins vers les modèles et preprocessors
BASE_DIR = Path(__file__).parent.parent  # Racine du projet
MODELS_DIR = BASE_DIR / "notebooks"  # Dossier contenant les modèles .pkl
PROCESSORS_DIR = MODELS_DIR / "processors"  # Dossier contenant les preprocessors

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Liste des modèles disponibles avec leurs performances estimées
# (À ajuster selon vos résultats d'entraînement)
AVAILABLE_MODELS = {
    "Stacking_LR_ensemble.pkl": {"name": "Stacking Ensemble", "priority": 1},
    "Voting_Soft_ensemble.pkl": {"name": "Voting Ensemble", "priority": 2},
    "XGBoost_tuned.pkl": {"name": "XGBoost Tuned", "priority": 3},
    "CatBoost_tuned.pkl": {"name": "CatBoost Tuned", "priority": 4},
    "LightGBM_tuned.pkl": {"name": "LightGBM Tuned", "priority": 5},
    "Random_Forest_tuned.pkl": {"name": "Random Forest Tuned", "priority": 6},
}

# ============================================================================
# MODÈLES PYDANTIC POUR VALIDATION DES DONNÉES
# ============================================================================

class CustomerData(BaseModel):
    """
    Modèle de données pour une prédiction individuelle
    Accepte les features métier simples et reconstruit automatiquement
    toutes les features avancées côté backend
    """
    # Features de base du client
    CreditScore: int = Field(..., description="Score de crédit du client (300-850)")
    Geography: str = Field(..., description="Pays du client (France, Germany, Spain)")
    Gender: str = Field(..., description="Genre du client (Male, Female)")
    Age: int = Field(None, description="Âge du client (optionnel si DateOfBirth fourni)")
    DateOfBirth: Optional[str] = Field(None, description="Date de naissance (YYYY-MM-DD)")
    
    # Informations bancaires
    Balance: float = Field(..., description="Solde du compte")
    NumOfProducts: int = Field(..., description="Nombre de produits (1-4)")
    HasCrCard: int = Field(..., description="Possède une carte de crédit (0 ou 1)")
    IsActiveMember: int = Field(..., description="Membre actif (0 ou 1)")
    CustomerTenure: int = Field(..., description="Ancienneté en mois")
    
    # Informations financières
    Income: float = Field(..., description="Revenu annuel")
    OutstandingLoans: float = Field(..., description="Prêts en cours")
    EstimatedSalary: float = Field(..., description="Salaire estimé")
    
    # Informations personnelles
    NumberOfDependents: int = Field(..., description="Nombre de personnes à charge")
    Occupation: str = Field(..., description="Profession du client")
    MaritalStatus: str = Field(..., description="Statut marital")
    EducationLevel: str = Field(..., description="Niveau d'éducation")
    
    # Informations comportementales
    CustomerSegment: str = Field(..., description="Segment client")
    PreferredCommunicationChannel: str = Field(..., description="Canal de communication préféré")
    NumComplaints: int = Field(0, description="Nombre de plaintes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Balance": 125000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "CustomerTenure": 24,
                "Income": 85000.0,
                "OutstandingLoans": 15000.0,
                "EstimatedSalary": 85000.0,
                "NumberOfDependents": 2,
                "Occupation": "Engineer",
                "MaritalStatus": "Married",
                "EducationLevel": "Bachelor",
                "CustomerSegment": "Premium",
                "PreferredCommunicationChannel": "Email",
                "NumComplaints": 0
            }
        }

class PredictionResponse(BaseModel):
    """
    Modèle de réponse pour une prédiction
    """
    prediction: int = Field(..., description="Prédiction (0 = reste, 1 = churn)")
    probability: float = Field(..., description="Probabilité de churn")
    model_version: str = Field(..., description="Version du modèle utilisé")

# ============================================================================
# INITIALISATION DE L'APPLICATION FASTAPI
# ============================================================================

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API de prédiction de churn bancaire utilisant MLflow",
    version="1.0.0"
)

# Variables globales pour stocker le modèle, les preprocessors et les informations
model = None
label_encoders = None
scaler = None
feature_names = None
model_info = {}

# ============================================================================
# FONCTIONS DE CHARGEMENT DU MODÈLE ET PREPROCESSORS
# ============================================================================

def load_preprocessors():
    """
    Charge les preprocessors (label encoders, scaler, feature names) depuis le dossier processors
    """
    global label_encoders, scaler, feature_names
    
    try:
        # Charger les label encoders
        label_encoders_path = PROCESSORS_DIR / "label_encoders.pkl"
        if label_encoders_path.exists():
            with open(label_encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
            print(f"✅ Label encoders chargés depuis {label_encoders_path}")
        else:
            print(f"⚠️  Label encoders non trouvés: {label_encoders_path}")
        
        # Charger le scaler
        scaler_path = PROCESSORS_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler chargé depuis {scaler_path}")
        else:
            print(f"⚠️  Scaler non trouvé: {scaler_path}")
        
        # Charger les noms de features
        feature_names_path = PROCESSORS_DIR / "feature_names.pkl"
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"✅ Feature names chargés depuis {feature_names_path}")
        else:
            print(f"⚠️  Feature names non trouvés: {feature_names_path}")
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement des preprocessors: {str(e)}")
        raise

def load_best_model():
    """
    Charge le meilleur modèle disponible depuis les fichiers pickle
    Essaie de charger les modèles dans l'ordre de priorité défini
    """
    global model, model_info
    
    try:
        print(f"🔍 Recherche du meilleur modèle dans: {MODELS_DIR}")
        
        # Charger d'abord les preprocessors
        load_preprocessors()
        
        # Essayer de charger les modèles dans l'ordre de priorité
        models_by_priority = sorted(AVAILABLE_MODELS.items(), key=lambda x: x[1]["priority"])
        
        model_loaded = False
        for model_filename, model_metadata in models_by_priority:
            model_path = MODELS_DIR / model_filename
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    model_info = {
                        "model_name": model_metadata["name"],
                        "model_file": model_filename,
                        "model_path": str(model_path),
                        "priority": model_metadata["priority"]
                    }
                    
                    print(f"🏆 Modèle chargé avec succès!")
                    print(f"📦 Nom: {model_info['model_name']}")
                    print(f"📁 Fichier: {model_info['model_file']}")
                    
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"⚠️  Erreur lors du chargement de {model_filename}: {str(e)}")
                    continue
        
        if not model_loaded:
            raise ValueError("Aucun modèle n'a pu être chargé. Vérifiez que les fichiers .pkl existent dans le dossier notebooks/")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        raise

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le feature engineering complet pour recréer toutes les features
    utilisées lors de l'entraînement du modèle
    
    Args:
        data: DataFrame avec les données brutes du client
    
    Returns:
        DataFrame avec toutes les features engineered
    """
    try:
        df = data.copy()
        
        # ============================================================================
        # 1. CALCUL DE L'ÂGE SI NÉCESSAIRE
        # ============================================================================
        if 'DateOfBirth' in df.columns and df['DateOfBirth'].notna().any():
            reference_date = pd.Timestamp.now()
            df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
            df['Age'] = (reference_date - df['DateOfBirth']).dt.days / 365.25
            df['Age'] = df['Age'].round(0).astype(int)
            df = df.drop(columns=['DateOfBirth'])
        
        # ============================================================================
        # 2. FEATURE ENGINEERING - Créer les features dérivées
        # ============================================================================
        
        # Income per dependent
        df['Income_Per_Dependent'] = df['Income'] / (df['NumberOfDependents'] + 1)
        
        # Balance per product
        df['Balance_Per_Product'] = df['Balance'] / df['NumOfProducts']
        
        # Credit utilization
        df['Credit_Utilization'] = df['OutstandingLoans'] / df['Income']
        
        # Loan to balance ratio
        df['Loan_To_Balance_Ratio'] = df['OutstandingLoans'] / (df['Balance'] + 1)
        
        # Tenure groups
        df['Tenure_Group'] = pd.cut(df['CustomerTenure'],
                                     bins=[0, 6, 12, 24, 30],
                                     labels=['0-6m', '6-12m', '1-2y', '2y+'])
        
        # Credit score categories
        df['Credit_Category'] = pd.cut(df['CreditScore'],
                                        bins=[0, 579, 669, 739, 799, 850],
                                        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        # Products per year (engagement metric)
        df['Products_Per_Year'] = df['NumOfProducts'] / (df['CustomerTenure'] / 12 + 0.1)
        
        # Complaints per year
        df['Complaints_Per_Year'] = df['NumComplaints'] / (df['CustomerTenure'] / 12 + 0.1)
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'],
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        # High value customer flag (utilise des quantiles fixes basés sur l'entraînement)
        # Note: Idéalement, ces seuils devraient être sauvegardés lors de l'entraînement
        balance_threshold = 100000  # Approximation du 75e percentile
        df['High_Value_Customer'] = ((df['Balance'] > balance_threshold) & 
                                      (df['NumOfProducts'] >= 3)).astype(int)
        
        # At-risk flag (utilise des médianes fixes basées sur l'entraînement)
        complaints_median = 1  # Approximation
        balance_median = 50000  # Approximation
        df['At_Risk'] = ((df['NumComplaints'] > complaints_median) & 
                         (df['Balance'] < balance_median)).astype(int)
        
        print(f"✅ Feature engineering terminé: {df.shape[1]} colonnes créées")
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def encode_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'encodage des variables catégorielles
    
    Args:
        data: DataFrame avec les features engineered
    
    Returns:
        DataFrame avec les features encodées
    """
    try:
        df = data.copy()
        
        # ============================================================================
        # 1. BINARY ENCODING - Gender
        # ============================================================================
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        
        # ============================================================================
        # 2. ONE-HOT ENCODING - Variables catégorielles
        # ============================================================================
        categorical_to_encode = [
            'MaritalStatus', 'EducationLevel', 'CustomerSegment',
            'PreferredCommunicationChannel', 'Age_Group', 'Tenure_Group',
            'Credit_Category'
        ]
        
        df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True, dtype=int)
        
        # ============================================================================
        # 3. LABEL ENCODING - Occupation (haute cardinalité)
        # ============================================================================
        if 'Occupation' in df.columns and label_encoders is not None:
            if 'Occupation' in label_encoders:
                encoder = label_encoders['Occupation']
                try:
                    df['Occupation_Encoded'] = encoder.transform(df['Occupation'])
                except ValueError:
                    # Si une valeur inconnue, utiliser la première classe
                    print(f"⚠️  Occupation inconnue, utilisation de la valeur par défaut")
                    df['Occupation_Encoded'] = encoder.transform([encoder.classes_[0]])[0]
                df = df.drop(columns=['Occupation'])
        
        print(f"✅ Encodage terminé: {df.shape[1]} colonnes")
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors de l'encodage: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de preprocessing:
    1. Feature engineering
    2. Encodage des variables catégorielles
    3. Scaling
    
    Args:
        data: DataFrame avec les données brutes
    
    Returns:
        DataFrame avec les données preprocessées prêtes pour la prédiction
    """
    try:
        # Étape 1: Feature engineering
        df = engineer_features(data)
        
        # Étape 2: Encodage
        df = encode_features(df)
        
        # Étape 3: Scaling
        if scaler is not None:
            # Récupérer l'ordre des colonnes depuis feature_names
            if feature_names is not None and isinstance(feature_names, dict):
                numerical_cols = feature_names.get('numerical_features', [])
                all_features_expected = feature_names.get('all_features', [])
                
                # Vérifier quelles colonnes sont présentes
                missing_cols = [col for col in all_features_expected if col not in df.columns]
                if missing_cols:
                    print(f"⚠️  Colonnes manquantes: {missing_cols[:5]}...")  # Afficher les 5 premières
                    # Ajouter les colonnes manquantes avec des zéros
                    for col in missing_cols:
                        df[col] = 0
                
                # Réorganiser les colonnes dans le bon ordre
                df = df[all_features_expected]
                
                # Appliquer le scaling uniquement sur les colonnes numériques
                df_scaled = df.copy()
                if numerical_cols:
                    df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
                    df = df_scaled
            else:
                # Fallback: scaler toutes les colonnes
                df_scaled = scaler.transform(df)
                df = pd.DataFrame(df_scaled, columns=df.columns)
        
        print(f"✅ Preprocessing complet: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# ÉVÉNEMENTS DE DÉMARRAGE/ARRÊT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Événement exécuté au démarrage de l'application
    Charge le meilleur modèle depuis MLflow
    """
    print("🚀 Démarrage de l'API Bank Churn Prediction")
    load_best_model()
    print("✅ API prête à recevoir des requêtes")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Événement exécuté à l'arrêt de l'application
    """
    print("🛑 Arrêt de l'API Bank Churn Prediction")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Endpoint racine - Informations sur l'API
    """
    return {
        "message": "Bank Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "model_info": "/model-info"
        }
    }

@app.get("/health")
async def health_check():
    """
    Endpoint de vérification de santé
    Vérifie que le modèle est chargé et prêt
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "mlflow_uri": MLFLOW_TRACKING_URI
    }

@app.get("/model-info")
async def get_model_info():
    """
    Endpoint pour obtenir les informations sur le modèle chargé
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_info": model_info,
        "preprocessors_loaded": {
            "label_encoders": label_encoders is not None,
            "scaler": scaler is not None,
            "feature_names": feature_names is not None
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Endpoint de prédiction
    Accepte les données d'un client et retourne la prédiction de churn
    
    Args:
        customer: Données du client (CustomerData)
    
    Returns:
        PredictionResponse: Prédiction (0/1) et probabilité de churn
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non disponible. Veuillez vérifier que les modèles sont chargés."
        )
    
    try:
        # Convertir les données Pydantic en DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        print(f"📥 Requête de prédiction reçue: {customer_dict}")
        
        # Appliquer le preprocessing
        df_preprocessed = preprocess_input(df)
        
        print(f"🔄 Données preprocessées: {df_preprocessed.values[0][:5]}...")  # Afficher les 5 premières valeurs
        
        # Faire la prédiction
        prediction = model.predict(df_preprocessed)
        
        # Obtenir les probabilités si disponibles
        try:
            probabilities = model.predict_proba(df_preprocessed)
            # Probabilité de la classe 1 (churn)
            churn_probability = float(probabilities[0][1])
        except AttributeError:
            # Si predict_proba n'est pas disponible, utiliser la prédiction binaire
            churn_probability = float(prediction[0])
        
        # Convertir la prédiction en entier (0 ou 1)
        prediction_value = int(prediction[0])
        
        print(f"✅ Prédiction: {prediction_value}, Probabilité: {churn_probability:.4f}")
        
        return PredictionResponse(
            prediction=prediction_value,
            probability=round(churn_probability, 4),
            model_version=model_info.get("model_file", "unknown")
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    # Lancer le serveur uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
