"""
Churn Data Preprocessing Class
================================
A comprehensive preprocessing pipeline for bank customer churn prediction.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
import gc
import os
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


class ChurnPreprocessor:
    """
    A comprehensive preprocessing class for bank customer churn data.
    
    This class handles:
    - Data loading and extraction
    - Feature engineering
    - Missing value imputation
    - Categorical encoding
    - Outlier handling
    - Feature scaling
    - Class imbalance handling with SMOTE
    - Model-ready data preparation
    """
    
    def __init__(self, input_file='../data/bank_customer_churn.csv', 
                 n_rows=30000, 
                 test_size=0.2, 
                 random_state=42):
        """
        Initialize the ChurnPreprocessor.
        
        Parameters:
        -----------
        input_file : str
            Path to the input CSV file
        n_rows : int
            Number of rows to extract from the dataset
        test_size : float
            Proportion of data to use for testing (0.0 to 1.0)
        random_state : int
            Random seed for reproducibility
        """
        self.input_file = input_file
        self.n_rows = n_rows
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize storage for processors
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = {}
        self.smote_config = {}
        
        # Data containers
        self.df = None
        self.df_clean = None
        self.df_encoded = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        
        # Configure plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        plt.rcParams['figure.figsize'] = (15, 6)
        plt.rcParams['font.size'] = 10
        
    def load_data(self):
        """Load and extract the specified number of rows from the dataset."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        try:
            df_full = pd.read_csv(self.input_file)
            print(f"✅ Fichier bank_customer_churn.csv chargé avec succès!")
            print(f"   📊 Dimensions totales: {df_full.shape[0]:,} lignes × {df_full.shape[1]} colonnes")
            print(f"   💾 Taille mémoire: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except FileNotFoundError:
            print(f"❌ Erreur: Le fichier '{self.input_file}' n'a pas été trouvé!")
            print("   Veuillez vérifier que le fichier existe dans le répertoire de travail.")
            raise
        
        # Extract specified number of rows
        if len(df_full) >= self.n_rows:
            self.df = df_full.head(self.n_rows).copy()
            print(f"\n✅ Extraction réussie de {self.n_rows:,} lignes ({(self.n_rows/len(df_full)*100):.1f}% du total)")
        else:
            self.df = df_full.copy()
            print(f"\n⚠️  Le fichier contient seulement {len(df_full):,} lignes")
            print(f"   Toutes les lignes seront utilisées")
            self.n_rows = len(df_full)
        
        print(f"📊 Dimensions extraites: {self.df.shape[0]:,} lignes × {self.df.shape[1]} colonnes")
        
        # Verify churn distribution
        if 'Churn Flag' in self.df.columns:
            original_churn_rate = df_full['Churn Flag'].mean() * 100
            extracted_churn_rate = self.df['Churn Flag'].mean() * 100
            print(f"\n   Vérification de la distribution de churn:")
            print(f"   • Taux de churn original: {original_churn_rate:.3f}%")
            print(f"   • Taux de churn extrait: {extracted_churn_rate:.3f}%")
            if abs(original_churn_rate - extracted_churn_rate) < 0.1:
                print(f"✅ Distribution similaire maintenue")
            else:
                print(f"⚠️  Différence de distribution: {abs(original_churn_rate - extracted_churn_rate):.3f}%")
        
        # Free memory
        del df_full
        gc.collect()
        print("🗑️  Mémoire libérée (DataFrame original supprimé)")
        
        return self
    
    def drop_unnecessary_columns(self):
        """Drop columns that won't help with prediction."""
        print("\n" + "="*80)
        print("1. DROP UNNECESSARY COLUMNS")
        print("="*80)
        
        columns_to_drop = [
            'RowNumber',           # Just an index
            'CustomerId',          # Unique identifier (no predictive value)
            'Surname',             # Personal identifier (no predictive value)
            'First Name',          # Personal identifier (no predictive value)
            'Address',             # Unique for each customer
            'Contact Information', # Unique for each customer
            'Churn Reason',        # This is a result of churn, not a predictor
            'Churn Date'           # This is a result of churn, not a predictor
        ]
        
        # Only drop columns that exist
        columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        
        self.df_clean = self.df.drop(columns=columns_to_drop)
        print(f"\nAfter dropping unnecessary columns: {self.df_clean.shape}")
        
        return self
    
    def engineer_date_features(self, reference_date='2024-12-11'):
        """Engineer age-related features from Date of Birth."""
        print("\n" + "="*80)
        print("2. DATE FEATURE ENGINEERING")
        print("="*80)
        
        # Convert Date of Birth to datetime
        self.df_clean['Date of Birth'] = pd.to_datetime(
            self.df_clean['Date of Birth'], 
            format='%Y-%m-%d', 
            errors='coerce'
        )
        
        # Create age feature
        reference_date = pd.Timestamp(reference_date)
        self.df_clean['Age'] = (reference_date - self.df_clean['Date of Birth']).dt.days / 365.25
        self.df_clean['Age'] = self.df_clean['Age'].round(0).astype(int)
        
        # Create age groups
        self.df_clean['Age_Group'] = pd.cut(
            self.df_clean['Age'], 
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        # Drop original date column
        self.df_clean = self.df_clean.drop(columns=['Date of Birth'])
        
        print(f"\nAge statistics:")
        print(self.df_clean['Age'].describe())
        
        return self
    
    def engineer_features(self):
        """Create derived features for better prediction."""
        print("\n" + "="*80)
        print("3. FEATURE ENGINEERING")
        print("="*80)
        
        # Income per dependent
        self.df_clean['Income_Per_Dependent'] = (
            self.df_clean['Income'] / (self.df_clean['Number of Dependents'] + 1)
        )
        
        # Balance per product
        self.df_clean['Balance_Per_Product'] = (
            self.df_clean['Balance'] / self.df_clean['NumOfProducts']
        )
        
        # Credit utilization
        self.df_clean['Credit_Utilization'] = (
            self.df_clean['Outstanding Loans'] / self.df_clean['Income']
        )
        
        # Loan to Balance ratio
        self.df_clean['Loan_To_Balance_Ratio'] = (
            self.df_clean['Outstanding Loans'] / (self.df_clean['Balance'] + 1)
        )
        
        # Tenure groups
        self.df_clean['Tenure_Group'] = pd.cut(
            self.df_clean['Customer Tenure'],
            bins=[0, 6, 12, 24, 30],
            labels=['0-6m', '6-12m', '1-2y', '2y+']
        )
        
        # Credit score categories
        self.df_clean['Credit_Category'] = pd.cut(
            self.df_clean['Credit Score'],
            bins=[0, 579, 669, 739, 799, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # Products per tenure (engagement metric)
        self.df_clean['Products_Per_Year'] = (
            self.df_clean['NumOfProducts'] / (self.df_clean['Customer Tenure'] / 12 + 0.1)
        )
        
        # Complaints per year
        self.df_clean['Complaints_Per_Year'] = (
            self.df_clean['NumComplaints'] / (self.df_clean['Customer Tenure'] / 12 + 0.1)
        )
        
        # High value customer flag
        self.df_clean['High_Value_Customer'] = (
            (self.df_clean['Balance'] > self.df_clean['Balance'].quantile(0.75)) & 
            (self.df_clean['NumOfProducts'] >= 3)
        ).astype(int)
        
        # At-risk flag
        self.df_clean['At_Risk'] = (
            (self.df_clean['NumComplaints'] > self.df_clean['NumComplaints'].median()) & 
            (self.df_clean['Balance'] < self.df_clean['Balance'].median())
        ).astype(int)
        
        print(f"\nAfter feature engineering: {self.df_clean.shape}")
        
        return self
    
    def handle_missing_values(self):
        """Impute missing values in numerical and categorical columns."""
        print("\n" + "="*80)
        print("4. HANDLE MISSING VALUES")
        print("="*80)
        
        missing_before = self.df_clean.isnull().sum().sum()
        print(f"\nMissing values before imputation: {missing_before}")
        
        # Fill numerical values with median
        numerical_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df_clean[col].isnull().sum() > 0:
                self.df_clean[col].fillna(self.df_clean[col].median(), inplace=True)
        
        # Fill categorical values with mode
        categorical_cols = self.df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.df_clean[col].isnull().sum() > 0:
                self.df_clean[col].fillna(self.df_clean[col].mode()[0], inplace=True)
        
        missing_after = self.df_clean.isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")
        
        return self
    
    def encode_categorical_variables(self):
        """Encode categorical variables using appropriate methods."""
        print("\n" + "="*80)
        print("5. ENCODE CATEGORICAL VARIABLES")
        print("="*80)
        
        # Binary encoding for Gender
        if 'Gender' in self.df_clean.columns:
            self.df_clean['Gender'] = self.df_clean['Gender'].map({'Male': 1, 'Female': 0})
        
        # One-hot encoding for categorical variables
        categorical_to_encode = [
            'Marital Status', 'Education Level', 'Customer Segment', 
            'Preferred Communication Channel', 'Age_Group', 'Tenure_Group', 
            'Credit_Category'
        ]
        
        # Only encode columns that exist
        categorical_to_encode = [col for col in categorical_to_encode if col in self.df_clean.columns]
        
        self.df_encoded = pd.get_dummies(
            self.df_clean, 
            columns=categorical_to_encode, 
            drop_first=True, 
            dtype=int
        )
        
        # Label encoding for Occupation
        if 'Occupation' in self.df_encoded.columns:
            le = LabelEncoder()
            self.df_encoded['Occupation_Encoded'] = le.fit_transform(self.df_encoded['Occupation'])
            self.df_encoded = self.df_encoded.drop(columns=['Occupation'])
            self.label_encoders['Occupation'] = le
        
        print(f"\nAfter encoding: {self.df_encoded.shape}")
        print(f"Number of features: {self.df_encoded.shape[1] - 1}")  # Excluding target
        
        return self
    
    def cap_outliers(self, df, column, lower_percentile=0.01, upper_percentile=0.99):
        """Cap outliers at specified percentiles."""
        lower_cap = df[column].quantile(lower_percentile)
        upper_cap = df[column].quantile(upper_percentile)
        df[column] = df[column].clip(lower=lower_cap, upper=upper_cap)
        return df
    
    def handle_outliers(self):
        """Handle outliers in key numerical features."""
        print("\n" + "="*80)
        print("6. HANDLE OUTLIERS (Optional - using IQR capping)")
        print("="*80)
        
        outlier_cols = [
            'Income', 'Outstanding Loans', 'Balance', 'Income_Per_Dependent', 
            'Balance_Per_Product', 'Credit_Utilization'
        ]
        
        for col in outlier_cols:
            if col in self.df_encoded.columns:
                self.df_encoded = self.cap_outliers(self.df_encoded, col)
        
        print("\nOutliers capped for numerical features")
        
        return self
    
    def split_features_target(self):
        """Separate features and target variable."""
        print("\n" + "="*80)
        print("7. SEPARATE FEATURES AND TARGET")
        print("="*80)
        
        X = self.df_encoded.drop('Churn Flag', axis=1)
        y = self.df_encoded['Churn Flag']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nChurn distribution:\n{y.value_counts()}")
        print(f"Churn rate: {y.mean()*100:.2f}%")
        
        return X, y
    
    def train_test_split_data(self, X, y):
        """Perform stratified train-test split."""
        print("\n" + "="*80)
        print("8. TRAIN-TEST SPLIT (Stratified)")
        print("="*80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape}, Churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"Test set: {self.X_test.shape}, Churn rate: {self.y_test.mean()*100:.2f}%")
        
        return self
    
    def scale_features(self):
        """Scale numerical features using StandardScaler."""
        print("\n" + "="*80)
        print("9. FEATURE SCALING")
        print("="*80)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Get numerical columns (exclude binary encoded columns)
        numerical_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names['numerical_features'] = numerical_features
        self.feature_names['all_features'] = self.X_train.columns.tolist()
        
        # Scale features
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        self.X_train_scaled[numerical_features] = self.scaler.fit_transform(
            self.X_train[numerical_features]
        )
        self.X_test_scaled[numerical_features] = self.scaler.transform(
            self.X_test[numerical_features]
        )
        
        return self
    
    def handle_class_imbalance(self, apply_smote=True):
        """Handle class imbalance using SMOTE."""
        print("\n" + "="*80)
        print("10. HANDLE CLASS IMBALANCE (Multiple Options)")
        print("="*80)
        
        print("\n" + "="*80)
        print("GESTION DU DÉSÉQUILIBRE DES CLASSES (SMOTE)")
        print("="*80)
        
        print("\n📊 Distribution AVANT SMOTE:")
        print(f" Classe 0 (Non-churn) : {Counter(self.y_train)[0]:,}")
        print(f" Classe 1 (Churn)      : {Counter(self.y_train)[1]:,}")
        print(f" Ratio: {Counter(self.y_train)[0] / Counter(self.y_train)[1]:.2f}:1")
        
        if apply_smote and Counter(self.y_train)[1] > 1:
            smote = SMOTE(
                random_state=self.random_state,
                k_neighbors=min(5, Counter(self.y_train)[1] - 1)
            )
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train_scaled, 
                self.y_train
            )
            
            print("\n📊 Distribution APRÈS SMOTE:")
            print(f" Classe 0 : {Counter(self.y_train_resampled)[0]:,}")
            print(f" Classe 1 : {Counter(self.y_train_resampled)[1]:,}")
            print(" Ratio: 1:1 (équilibré)")
            
            self.smote_config = {"applied": True, "strategy": "SMOTE"}
            smote_applied = True
        else:
            print("\n⚠️ Impossible d'appliquer SMOTE (classe minoritaire absente)")
            self.X_train_resampled = self.X_train_scaled
            self.y_train_resampled = self.y_train
            self.smote_config = {"applied": False, "strategy": "SMOTE"}
            smote_applied = False
        
        return self
    
    def save_processors(self, output_dir='processors'):
        """Save all processors and preprocessed data."""
        print("\n" + "="*80)
        print("11. SAVE PROCESSED DATA")
        print("="*80)
        print("="*80)
        print("SAUVEGARDE DES PROCESSEURS")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1 — Save scaler
        with open(f"{output_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        print("✓ Scaler sauvegardé")
        
        # 2 — Save label encoders (occupation only)
        with open(f"{output_dir}/label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        print("✓ Label encoders sauvegardés")
        
        # 3 — Save feature names
        with open(f"{output_dir}/feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)
        print("✓ Noms des features sauvegardés")
        
        # 4 — Save SMOTE config
        with open(f"{output_dir}/smote_config.pkl", "wb") as f:
            pickle.dump(self.smote_config, f)
        print("✓ Configuration SMOTE sauvegardée")
        
        # 5 — Save final datasets
        datasets = {
            "X_train": self.X_train_resampled,
            "y_train": self.y_train_resampled,
            "X_test": self.X_test_scaled,
            "y_test": self.y_test
        }
        with open(f"{output_dir}/preprocessed_data.pkl", "wb") as f:
            pickle.dump(datasets, f)
        
        print("\n✅ Tous les processeurs et données preprocessées ont été sauvegardés.")
        
        return self
    
    def save_intermediate_csv(self, output_file='../data/churn.csv'):
        """Save intermediate cleaned dataset as CSV."""
        # Save the original extracted data, not the cleaned one (to match original behavior)
        self.df.to_csv(output_file, index=False)
        print(f"✅ Export réussi!")
        print(f"📁 Fichier créé: churn.csv ({self.n_rows:,} lignes)")
        print(f"💾 Taille du fichier: ~{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self
    
    def run_full_pipeline(self, save_csv=True, apply_smote=True):
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        save_csv : bool
            Whether to save intermediate CSV file
        apply_smote : bool
            Whether to apply SMOTE for class imbalance
        
        Returns:
        --------
        self : ChurnPreprocessor
            Returns self for method chaining
        """
        # Run all preprocessing steps
        self.load_data()
        
        if save_csv:
            self.save_intermediate_csv()
        
        self.drop_unnecessary_columns()
        self.engineer_date_features()
        self.engineer_features()
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.handle_outliers()
        
        X, y = self.split_features_target()
        
        self.train_test_split_data(X, y)
        self.scale_features()
        self.handle_class_imbalance(apply_smote=apply_smote)
        self.save_processors()
        
        return self
    
    def get_preprocessed_data(self):
        """
        Get the preprocessed training and testing data.
        
        Returns:
        --------
        dict : Dictionary containing all preprocessed data
        """
        return {
            'X_train': self.X_train_resampled,
            'y_train': self.y_train_resampled,
            'X_test': self.X_test_scaled,
            'y_test': self.y_test,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize preprocessor with configuration matching original code
    preprocessor = ChurnPreprocessor(
        input_file='../data/bank_customer_churn.csv',
        n_rows=30000,
        test_size=0.2,
        random_state=42
    )
    
    # Run full pipeline
    preprocessor.run_full_pipeline(save_csv=True, apply_smote=True)
    
    # Get preprocessed data
    data = preprocessor.get_preprocessed_data()
    
    print(f"\nX_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"y_test shape: {data['y_test'].shape}")