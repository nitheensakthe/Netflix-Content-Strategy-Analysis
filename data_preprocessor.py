"""
Data Preprocessing Module for Real Estate Price Prediction
Handles data cleaning, normalization, and train-test splitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import config


class DataPreprocessor:
    """Preprocess housing data for machine learning models"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = ['location_type', 'property_type']
        self.numerical_features = []
        
    def load_data(self, filepath):
        """Load housing data from CSV"""
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Found {missing.sum()} missing values")
            
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using Label Encoding"""
        df = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def create_temporal_features(self, df):
        """Create additional temporal features"""
        if 'date' in df.columns:
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            df['year_sin'] = np.sin(2 * np.pi * df['year'] / 12)
            df['year_cos'] = np.cos(2 * np.pi * df['year'] / 12)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        # Area per bedroom
        if 'area' in df.columns and 'bedrooms' in df.columns:
            df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
        
        # Price sensitivity to interest rate (for prediction)
        if 'area' in df.columns and 'interest_rate' in df.columns:
            df['area_interest_interaction'] = df['area'] * df['interest_rate']
        
        # Location quality score
        if 'infrastructure_score' in df.columns and 'distance_to_city' in df.columns:
            df['location_quality'] = df['infrastructure_score'] / (df['distance_to_city'] + 1)
        
        # Economic index
        if all(col in df.columns for col in ['interest_rate', 'inflation_rate', 'population_growth']):
            df['economic_index'] = (df['population_growth'] + df['inflation_rate']) / (df['interest_rate'] + 1)
        
        return df
    
    def scale_features(self, df, target_col='price', fit=True):
        """Scale numerical features"""
        # Identify numerical features (excluding target and date)
        exclude_cols = [target_col, 'date'] if 'date' in df.columns else [target_col]
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        self.numerical_features = numerical_cols
        
        if fit:
            self.scalers['features'] = StandardScaler()
            df[numerical_cols] = self.scalers['features'].fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scalers['features'].transform(df[numerical_cols])
        
        return df
    
    def prepare_features_target(self, df, target_col='price'):
        """Separate features and target variable"""
        # Columns to exclude from features
        exclude_cols = [target_col, 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        
        self.feature_names = feature_cols
        
        return X, y
    
    def split_data(self, X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess(self, df, fit=True, target_col='price'):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create additional features
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale features
        df = self.scale_features(df, target_col=target_col, fit=fit)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df, target_col=target_col)
        
        print(f"Preprocessing complete. Features: {X.shape[1]}")
        
        return X, y, df
    
    def save_preprocessor(self, filename='preprocessor.pkl'):
        """Save preprocessor objects"""
        filepath = os.path.join(config.MODELS_DIR, filename)
        preprocessor_data = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filename='preprocessor.pkl'):
        """Load preprocessor objects"""
        filepath = os.path.join(config.MODELS_DIR, filename)
        preprocessor_data = joblib.load(filepath)
        self.scalers = preprocessor_data['scalers']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        self.categorical_features = preprocessor_data['categorical_features']
        self.numerical_features = preprocessor_data['numerical_features']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load data
    data_path = os.path.join(config.DATA_DIR, 'housing_data.csv')
    if os.path.exists(data_path):
        df = preprocessor.load_data(data_path)
        
        # Preprocess data
        X, y, processed_df = preprocessor.preprocess(df)
        
        print("\n=== Processed Features ===")
        print(f"Feature names: {preprocessor.feature_names}")
        print(f"\nSample features:\n{X.head()}")
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Save preprocessor
        preprocessor.save_preprocessor()
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data_generator.py first")
