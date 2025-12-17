"""
Application Streamlit pour la prédiction de churn bancaire
Interface utilisateur moderne pour interagir avec l'API FastAPI
"""

import streamlit as st
import requests
import pandas as pd
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# URL de l'API backend
API_URL = "http://127.0.0.1:8000"

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLES CSS PERSONNALISÉS
# ============================================================================

st.markdown("""
<style>
    /* Style général */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Titre principal */
    .title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Sous-titre */
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Carte de résultat */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Carte de succès */
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Carte d'alerte */
    .warning-card {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Bouton personnalisé */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Section info */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def check_api_health():
    """
    Vérifie si l'API backend est accessible
    
    Returns:
        bool: True si l'API est accessible, False sinon
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def make_prediction(customer_data):
    """
    Envoie une requête de prédiction à l'API backend
    
    Args:
        customer_data (dict): Données du client
    
    Returns:
        dict: Réponse de l'API avec la prédiction
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"Erreur API: {response.status_code} - {response.text}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Impossible de se connecter à l'API. Vérifiez qu'elle est lancée sur http://127.0.0.1:8000"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Erreur lors de la requête: {str(e)}"
        }

# ============================================================================
# INTERFACE UTILISATEUR
# ============================================================================

# En-tête de l'application
st.markdown('<div class="title">🏦 Bank Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prédisez le risque de départ d\'un client bancaire</div>', unsafe_allow_html=True)

# Vérification de l'état de l'API
api_status = check_api_health()
if api_status:
    st.success("✅ API Backend connectée")
else:
    st.error(" ")

# Séparateur
st.markdown("---")

# ============================================================================
# FORMULAIRE DE SAISIE
# ============================================================================

st.markdown("### 📝 Informations du Client")

# Organisation en colonnes pour un meilleur layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Informations Personnelles")
    gender = st.selectbox("Genre", ["Male", "Female"], help="Genre du client")
    age = st.number_input("Âge", min_value=18, max_value=100, value=42, help="Âge du client en années")
    marital_status = st.selectbox(
        "Statut Marital",
        ["Single", "Married", "Divorced", "Widowed"],
        help="Statut marital du client"
    )
    education_level = st.selectbox(
        "Niveau d'Éducation",
        ["High School", "Bachelor", "Master", "PhD"],
        help="Niveau d'éducation du client"
    )
    occupation = st.text_input("Profession", value="Engineer", help="Profession du client")
    number_of_dependents = st.number_input(
        "Nombre de Personnes à Charge",
        min_value=0,
        max_value=10,
        value=2,
        help="Nombre de personnes à charge"
    )

with col2:
    st.markdown("#### 💰 Informations Financières")
    credit_score = st.slider(
        "Score de Crédit",
        min_value=300,
        max_value=850,
        value=650,
        help="Score de crédit du client (300-850)"
    )
    income = st.number_input(
        "Revenu Annuel (€)",
        min_value=0.0,
        value=85000.0,
        step=1000.0,
        help="Revenu annuel du client"
    )
    balance = st.number_input(
        "Solde du Compte (€)",
        min_value=0.0,
        value=125000.0,
        step=1000.0,
        help="Solde actuel du compte"
    )
    outstanding_loans = st.number_input(
        "Prêts en Cours (€)",
        min_value=0.0,
        value=15000.0,
        step=1000.0,
        help="Montant total des prêts en cours"
    )
    estimated_salary = st.number_input(
        "Salaire Estimé (€)",
        min_value=0.0,
        value=85000.0,
        step=1000.0,
        help="Salaire estimé du client"
    )

with col3:
    st.markdown("#### 🏦 Informations Bancaires")
    geography = st.selectbox(
        "Pays",
        ["France", "Germany", "Spain"],
        help="Pays de résidence du client"
    )
    customer_tenure = st.number_input(
        "Ancienneté (mois)",
        min_value=0,
        max_value=360,
        value=24,
        help="Nombre de mois en tant que client"
    )
    num_of_products = st.slider(
        "Nombre de Produits",
        min_value=1,
        max_value=4,
        value=2,
        help="Nombre de produits bancaires détenus"
    )
    has_cr_card = st.selectbox(
        "Carte de Crédit",
        [1, 0],
        format_func=lambda x: "Oui" if x == 1 else "Non",
        help="Possède une carte de crédit"
    )
    is_active_member = st.selectbox(
        "Membre Actif",
        [1, 0],
        format_func=lambda x: "Oui" if x == 1 else "Non",
        help="Client actif"
    )
    customer_segment = st.selectbox(
        "Segment Client",
        ["Standard", "Premium", "VIP"],
        help="Segment du client"
    )
    preferred_communication_channel = st.selectbox(
        "Canal de Communication Préféré",
        ["Email", "Phone", "SMS", "App"],
        help="Canal de communication préféré"
    )
    num_complaints = st.number_input(
        "Nombre de Plaintes",
        min_value=0,
        max_value=20,
        value=0,
        help="Nombre de plaintes enregistrées"
    )

# ============================================================================
# BOUTON DE PRÉDICTION
# ============================================================================

st.markdown("---")

if st.button("🔮 Prédire le Risque de Churn", use_container_width=True):
    # Préparer les données du client
    customer_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "CustomerTenure": customer_tenure,
        "Income": income,
        "OutstandingLoans": outstanding_loans,
        "EstimatedSalary": estimated_salary,
        "NumberOfDependents": number_of_dependents,
        "Occupation": occupation,
        "MaritalStatus": marital_status,
        "EducationLevel": education_level,
        "CustomerSegment": customer_segment,
        "PreferredCommunicationChannel": preferred_communication_channel,
        "NumComplaints": num_complaints
    }
    
    # Afficher un spinner pendant la prédiction
    with st.spinner("🔄 Analyse en cours..."):
        result = make_prediction(customer_data)
    
    # Afficher les résultats
    if result["success"]:
        prediction_data = result["data"]
        prediction = prediction_data["prediction"]
        probability = prediction_data["probability"]
        
        # Affichage conditionnel selon la prédiction
        if prediction == 1:
            # Client à risque de churn
            st.markdown(f"""
            <div class="warning-card">
                <h2>⚠️ RISQUE DE CHURN ÉLEVÉ</h2>
                <h3>Probabilité: {probability*100:.2f}%</h3>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Ce client présente un risque élevé de quitter la banque.
                    <br>Actions recommandées: Contact proactif, offres personnalisées, amélioration du service.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommandations
            st.markdown("### 💡 Recommandations")
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.info("""
                **Actions Immédiates:**
                - Contacter le client dans les 48h
                - Proposer un rendez-vous personnalisé
                - Analyser les plaintes récentes
                """)
            
            with col_rec2:
                st.info("""
                **Actions Préventives:**
                - Offrir des avantages exclusifs
                - Améliorer la qualité de service
                - Proposer des produits adaptés
                """)
        else:
            # Client fidèle
            st.markdown(f"""
            <div class="success-card">
                <h2>✅ CLIENT FIDÈLE</h2>
                <h3>Probabilité de churn: {probability*100:.2f}%</h3>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Ce client présente un faible risque de départ.
                    <br>Continuez à maintenir une relation de qualité.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommandations
            st.markdown("### 💡 Recommandations")
            st.success("""
            **Maintien de la Relation:**
            - Continuer le service de qualité actuel
            - Proposer des programmes de fidélité
            - Solliciter des retours d'expérience
            - Envisager des opportunités de cross-selling
            """)
        
        # Détails techniques (dans un expander)
        with st.expander("📊 Détails Techniques"):
            st.json(prediction_data)
            
    else:
        # Erreur lors de la prédiction
        st.error(f"❌ Erreur: {result['error']}")
        st.info("""
        **Vérifications:**
        - L'API backend est-elle lancée ? (`uvicorn backend.api:app --reload`)
        - L'URL de l'API est-elle correcte ? (http://127.0.0.1:8000)
        """)

# ============================================================================
# PIED DE PAGE
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p>🏦 Bank Churn Prediction System | MLOps Project</p>
    <p style="font-size: 0.9rem;">Powered by FastAPI + Streamlit + Machine Learning</p>
</div>
""", unsafe_allow_html=True)
