"""
Machine Learning Models for Real Estate Price Prediction
Includes: Linear Regression, Random Forest, XGBoost, and Neural Network
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import config

# Try to import TensorFlow (optional for Python 3.14+)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not available. Neural network model will be disabled.")


class HousingPriceModels:
    """Collection of ML models for housing price prediction"""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.feature_importance = {}
        
    def train_linear_regression(self, X_train, y_train, model_type='ridge'):
        """Train Linear Regression model"""
        print(f"\nTraining {model_type.upper()} Linear Regression...")
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        self.models[f'linear_{model_type}'] = model
        
        # Feature importance from coefficients
        self.feature_importance[f'linear_{model_type}'] = np.abs(model.coef_)
        
        print(f"{model_type.upper()} trained successfully")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest Regressor"""
        print("\nTraining Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        # Feature importance
        self.feature_importance['random_forest'] = model.feature_importances_
        
        print("Random Forest trained successfully")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost Regressor"""
        print("\nTraining XGBoost...")
        
        model = XGBRegressor(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        # Feature importance
        self.feature_importance['xgboost'] = model.feature_importances_
        
        print("XGBoost trained successfully")
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting Regressor"""
        print("\nTraining Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=config.RANDOM_SEED,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        
        # Feature importance
        self.feature_importance['gradient_boosting'] = model.feature_importances_
        
        print("Gradient Boosting trained successfully")
        return model
    
    def build_neural_network(self, input_dim):
        """Build Neural Network architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is not available. Cannot build neural network.")
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.NN_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val=None, y_val=None):
        """Train Neural Network"""
        if not HAS_TENSORFLOW:
            print("\nSkipping Neural Network (TensorFlow not available)...")
            return None
        
        print("\nTraining Neural Network...")
        
        model = self.build_neural_network(X_train.shape[1])
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=config.NN_EPOCHS,
            batch_size=config.NN_BATCH_SIZE,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['neural_network'] = model
        self.models['nn_history'] = history
        
        print("Neural Network trained successfully")
        return model, history
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model performance"""
        # Make predictions
        if model_name == 'neural_network':
            y_pred = model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        scores = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        self.model_scores[model_name] = scores
        
        return scores
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all available models"""
        print("=" * 70)
        print("TRAINING ALL MODELS")
        print("=" * 70)
        
        # Linear models
        self.train_linear_regression(X_train, y_train, 'ridge')
        self.train_linear_regression(X_train, y_train, 'lasso')
        
        # Tree-based models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Neural network
        self.train_neural_network(X_train, y_train, X_val, y_val)
        
        print("\n" + "=" * 70)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("=" * 70)
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "=" * 70)
        print("EVALUATING ALL MODELS")
        print("=" * 70)
        
        results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'nn_history' or model is None:
                continue
                
            print(f"\nEvaluating {model_name}...")
            scores = self.evaluate_model(model_name, model, X_test, y_test)
            results[model_name] = scores
            
            print(f"  RMSE: ${scores['rmse']:,.2f}")
            print(f"  MAE:  ${scores['mae']:,.2f}")
            print(f"  R²:   {scores['r2']:.4f}")
            print(f"  MAPE: {scores['mape']:.2f}%")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best_model[0].upper()}")
        print(f"RMSE: ${best_model[1]['rmse']:,.2f}")
        print("=" * 70)
        
        return results
    
    def get_feature_importance(self, model_name, feature_names, top_n=10):
        """Get top feature importances for a model"""
        if model_name not in self.feature_importance:
            return None
        
        importance = self.feature_importance[model_name]
        indices = np.argsort(importance)[::-1][:top_n]
        
        feature_importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importance[indices]
        })
        
        return feature_importance_df
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        for model_name, model in self.models.items():
            if model_name == 'nn_history' or model is None:
                continue
            
            filepath = os.path.join(config.MODELS_DIR, f'{model_name}_model.pkl')
            
            if model_name == 'neural_network' and HAS_TENSORFLOW:
                # Save Keras model
                filepath = os.path.join(config.MODELS_DIR, f'{model_name}_model.keras')
                model.save(filepath)
            else:
                # Save sklearn models
                joblib.dump(model, filepath)
            
            print(f"  Saved {model_name} to {filepath}")
        
        # Save scores and feature importance
        scores_path = os.path.join(config.MODELS_DIR, 'model_scores.pkl')
        joblib.dump(self.model_scores, scores_path)
        
        importance_path = os.path.join(config.MODELS_DIR, 'feature_importance.pkl')
        joblib.dump(self.feature_importance, importance_path)
        
        print("All models saved successfully")
    
    def load_model(self, model_name):
        """Load a specific model"""
        if model_name == 'neural_network':
            if not HAS_TENSORFLOW:
                print(f"Warning: Cannot load {model_name} - TensorFlow not available")
                return None
            filepath = os.path.join(config.MODELS_DIR, f'{model_name}_model.keras')
            model = keras.models.load_model(filepath)
        else:
            filepath = os.path.join(config.MODELS_DIR, f'{model_name}_model.pkl')
            model = joblib.load(filepath)
        
        self.models[model_name] = model
        print(f"Loaded {model_name} from {filepath}")
        return model
    
    def predict(self, model_name, X):
        """Make predictions using a specific model"""
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        if model_name == 'neural_network':
            predictions = model.predict(X, verbose=0).flatten()
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def ensemble_predict(self, X, method='average'):
        """Make ensemble predictions using multiple models"""
        predictions = []
        
        for model_name, model in self.models.items():
            if model_name == 'nn_history' or model is None:
                continue
            
            if model_name == 'neural_network' and HAS_TENSORFLOW:
                pred = model.predict(X, verbose=0).flatten()
            else:
                pred = model.predict(X)
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        elif method == 'weighted':
            # Weight by R² scores
            weights = [self.model_scores[name]['r2'] 
                      for name in self.models.keys() 
                      if name != 'nn_history' and self.models[name] is not None]
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred


if __name__ == "__main__":
    from data_preprocessor import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data_path = os.path.join(config.DATA_DIR, 'housing_data.csv')
    
    if os.path.exists(data_path):
        df = preprocessor.load_data(data_path)
        X, y, _ = preprocessor.preprocess(df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=config.RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_SEED
        )
        
        # Initialize and train models
        models = HousingPriceModels()
        models.train_all_models(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        results = models.evaluate_all_models(X_test, y_test)
        
        # Display feature importance
        print("\n=== Top 10 Important Features (Random Forest) ===")
        importance_df = models.get_feature_importance('random_forest', 
                                                       preprocessor.feature_names)
        print(importance_df)
        
        # Save models
        models.save_models()
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data_generator.py first")
