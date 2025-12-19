"""
Script de preprocessing pour les nouvelles données de production
Applique les mêmes transformations que dans preprocessing.ipynb
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class ProductionDataPreprocessor:
    def __init__(self, reference_date='2024-12-11'):
        self.reference_date = reference_date
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = {}
        
    def load_processors(self, processor_dir='notebooks/processors'):
        """Load existing processors from training"""
        try:
            with open(f"{processor_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f"{processor_dir}/label_encoders.pkl", 'rb') as f:
                self.label_encoders = pickle.load(f)
            with open(f"{processor_dir}/feature_names.pkl", 'rb') as f:
                self.feature_names = pickle.load(f)
            print("✅ Processors chargés depuis l'entraînement initial")
            return True
        except Exception as e:
            print(f"⚠️  Impossible de charger les processors: {e}")
            return False
    
    def transform(self, df):
        """Alias for preprocess to maintain sklearn compatibility"""
        return self.preprocess(df)
    
    def preprocess(self, df):
        """Apply full preprocessing pipeline"""
        print("\n" + "="*80)
        print("PREPROCESSING DES DONNÉES DE PRODUCTION")
        print("="*80)
        
        df_clean = df.copy()
        
        # 1. Drop unnecessary columns
        columns_to_drop = [
            'RowNumber', 'CustomerId', 'Surname', 'First Name', 'Address',
            'Contact Information', 'Churn Reason', 'Churn Date'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        df_clean = df_clean.drop(columns=columns_to_drop)
        print(f"✓ Colonnes supprimées: {len(columns_to_drop)}")
        
        # 2. Date features
        if 'Date of Birth' in df_clean.columns:
            df_clean['Date of Birth'] = pd.to_datetime(df_clean['Date of Birth'], errors='coerce')
            reference_date = pd.Timestamp(self.reference_date)
            df_clean['Age'] = (reference_date - df_clean['Date of Birth']).dt.days / 365.25
            df_clean['Age'] = df_clean['Age'].round(0).astype(int)
            df_clean['Age_Group'] = pd.cut(
                df_clean['Age'], 
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
            df_clean = df_clean.drop(columns=['Date of Birth'])
            print("✓ Features d'âge créées")
        
        # 3. Feature engineering
        df_clean['Income_Per_Dependent'] = df_clean['Income'] / (df_clean['Number of Dependents'] + 1)
        df_clean['Balance_Per_Product'] = df_clean['Balance'] / df_clean['NumOfProducts']
        df_clean['Credit_Utilization'] = df_clean['Outstanding Loans'] / df_clean['Income']
        df_clean['Loan_To_Balance_Ratio'] = df_clean['Outstanding Loans'] / (df_clean['Balance'] + 1)
        
        df_clean['Tenure_Group'] = pd.cut(
            df_clean['Customer Tenure'],
            bins=[0, 6, 12, 24, 30],
            labels=['0-6m', '6-12m', '1-2y', '2y+']
        )
        
        df_clean['Credit_Category'] = pd.cut(
            df_clean['Credit Score'],
            bins=[0, 579, 669, 739, 799, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        df_clean['Products_Per_Year'] = df_clean['NumOfProducts'] / (df_clean['Customer Tenure'] / 12 + 0.1)
        df_clean['Complaints_Per_Year'] = df_clean['NumComplaints'] / (df_clean['Customer Tenure'] / 12 + 0.1)
        
        df_clean['High_Value_Customer'] = (
            (df_clean['Balance'] > df_clean['Balance'].quantile(0.75)) & 
            (df_clean['NumOfProducts'] >= 3)
        ).astype(int)
        
        df_clean['At_Risk'] = (
            (df_clean['NumComplaints'] > df_clean['NumComplaints'].median()) & 
            (df_clean['Balance'] < df_clean['Balance'].median())
        ).astype(int)
        
        print("✓ Features engineered créées")
        
        # 4. Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print("✓ Valeurs manquantes traitées")
        
        # 5. Encode categorical variables
        if 'Gender' in df_clean.columns:
            df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
        
        categorical_to_encode = [
            'Marital Status', 'Education Level', 'Customer Segment',
            'Age_Group', 'Tenure_Group', 'Credit_Category'
        ]
        categorical_to_encode = [col for col in categorical_to_encode if col in df_clean.columns]
        
        df_encoded = pd.get_dummies(df_clean, columns=categorical_to_encode, drop_first=True, dtype=int)
        
        if 'Occupation' in df_encoded.columns:
            if 'Occupation' in self.label_encoders:
                # Use existing encoder
                df_encoded['Occupation_Encoded'] = self.label_encoders['Occupation'].transform(df_encoded['Occupation'])
            else:
                # Create new encoder
                le = LabelEncoder()
                df_encoded['Occupation_Encoded'] = le.fit_transform(df_encoded['Occupation'])
                self.label_encoders['Occupation'] = le
            df_encoded = df_encoded.drop(columns=['Occupation'])
        
        print("✓ Variables catégorielles encodées")
        
        # 6. Cap outliers
        outlier_cols = [
            'Income', 'Outstanding Loans', 'Balance', 'Income_Per_Dependent',
            'Balance_Per_Product', 'Credit_Utilization'
        ]
        for col in outlier_cols:
            if col in df_encoded.columns:
                lower_cap = df_encoded[col].quantile(0.01)
                upper_cap = df_encoded[col].quantile(0.99)
                df_encoded[col] = df_encoded[col].clip(lower=lower_cap, upper=upper_cap)
        
        print("✓ Outliers traités")
        
        # 7. Align columns with training data
        if 'all_features' in self.feature_names:
            expected_cols = self.feature_names['all_features']
            # Add missing columns with 0
            for col in expected_cols:
                if col not in df_encoded.columns and col != 'Churn Flag':
                    df_encoded[col] = 0
            # Keep only expected columns (except target)
            cols_to_keep = [col for col in expected_cols if col in df_encoded.columns]
            df_encoded = df_encoded[cols_to_keep]
            print(f"✓ Colonnes alignées: {len(cols_to_keep)} features")
        
        print(f"\n✅ Preprocessing terminé: {df_encoded.shape}")
        return df_encoded

def preprocess_production_data(input_file, output_file, processor_dir='notebooks/processors'):
    """Main function to preprocess production data"""
    print("="*80)
    print("PREPROCESSING PRODUCTION DATA")
    print("="*80)
    
    # Load data
    print(f"\n📂 Chargement: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Chargé: {df.shape}")
    
    # Preprocess
    preprocessor = ProductionDataPreprocessor()
    preprocessor.load_processors(processor_dir)
    df_processed = preprocessor.preprocess(df)
    
    # Save
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    print(f"\n✅ Sauvegardé: {output_file}")
    
    return df_processed

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/production/bank_churn_prod.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/production/bank_churn_prod_processed.csv"
    
    preprocess_production_data(input_file, output_file)
