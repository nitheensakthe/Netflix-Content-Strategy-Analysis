"""
Main Application for Real Estate Price Prediction System
Orchestrates data generation, model training, and predictions
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
import config
from data_generator import HousingDataGenerator
from data_preprocessor import DataPreprocessor
from models import HousingPriceModels
from predictor import HousingPricePredictor
from visualizer import HousingDataVisualizer


class RealEstatePricePredictionSystem:
    """Main system for real estate price prediction"""
    
    def __init__(self):
        self.data_generator = HousingDataGenerator()
        self.preprocessor = DataPreprocessor()
        self.models = HousingPriceModels()
        self.predictor = HousingPricePredictor()
        self.visualizer = HousingDataVisualizer()
        self.data = None
        
    def generate_data(self, n_samples=config.SAMPLE_SIZE):
        """Generate synthetic housing data"""
        print("\n" + "="*70)
        print("STEP 1: GENERATING HOUSING DATA")
        print("="*70)
        
        self.data_generator = HousingDataGenerator(n_samples=n_samples)
        self.data = self.data_generator.generate_dataset()
        
        # Save dataset
        self.data_generator.save_dataset(self.data)
        
        print(f"\nGenerated {len(self.data)} housing records")
        print(f"Price range: ${self.data['price'].min():,.2f} - ${self.data['price'].max():,.2f}")
        print(f"Average price: ${self.data['price'].mean():,.2f}")
        
        return self.data
    
    def preprocess_data(self, data=None):
        """Preprocess housing data"""
        print("\n" + "="*70)
        print("STEP 2: PREPROCESSING DATA")
        print("="*70)
        
        if data is None:
            # Load existing data
            data_path = os.path.join(config.DATA_DIR, 'housing_data.csv')
            self.data = self.preprocessor.load_data(data_path)
        else:
            self.data = data
        
        # Preprocess
        X, y, processed_df = self.preprocessor.preprocess(self.data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=config.RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_SEED
        )
        
        # Save preprocessor
        self.preprocessor.save_preprocessor()
        
        print(f"\nData split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Testing:    {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train all ML models"""
        print("\n" + "="*70)
        print("STEP 3: TRAINING MODELS")
        print("="*70)
        
        self.models.train_all_models(X_train, y_train, X_val, y_val)
        self.models.save_models()
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("STEP 4: EVALUATING MODELS")
        print("="*70)
        
        results = self.models.evaluate_all_models(X_test, y_test)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('rmse')
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        print(comparison_df.to_string())
        
        # Save results
        results_path = os.path.join(config.RESULTS_DIR, 'model_comparison.csv')
        comparison_df.to_csv(results_path)
        print(f"\nResults saved to {results_path}")
        
        return results
    
    def create_visualizations(self, X_test, y_test, results):
        """Create visualizations"""
        print("\n" + "="*70)
        print("STEP 5: CREATING VISUALIZATIONS")
        print("="*70)
        
        viz_dir = os.path.join(config.RESULTS_DIR, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Model comparison
        print("Creating model comparison chart...")
        self.visualizer.plot_model_comparison(
            results, show=False,
            save_path=os.path.join(viz_dir, 'model_comparison.png')
        )
        
        # Actual vs Predicted for best model
        best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
        print(f"Creating actual vs predicted plot for {best_model_name}...")
        
        if best_model_name == 'neural_network':
            y_pred = self.models.models[best_model_name].predict(X_test, verbose=0).flatten()
        else:
            y_pred = self.models.models[best_model_name].predict(X_test)
        
        self.visualizer.plot_actual_vs_predicted(
            y_test, y_pred, model_name=best_model_name,
            show=False, save_path=os.path.join(viz_dir, 'actual_vs_predicted.png')
        )
        
        # Prediction errors
        print("Creating prediction error plots...")
        self.visualizer.plot_prediction_errors(
            y_test, y_pred, show=False,
            save_path=os.path.join(viz_dir, 'prediction_errors.png')
        )
        
        # Feature importance
        print("Creating feature importance plot...")
        importance_df = self.models.get_feature_importance(
            'random_forest', self.preprocessor.feature_names, top_n=15
        )
        self.visualizer.plot_feature_importance(
            importance_df, show=False,
            save_path=os.path.join(viz_dir, 'feature_importance.png')
        )
        
        # Data visualizations
        if self.data is not None:
            print("Creating data distribution plots...")
            
            self.visualizer.plot_price_distribution(
                self.data, show=False,
                save_path=os.path.join(viz_dir, 'price_distribution.png')
            )
            
            self.visualizer.plot_location_price_comparison(
                self.data, show=False,
                save_path=os.path.join(viz_dir, 'location_comparison.png')
            )
            
            self.visualizer.plot_price_trends(
                self.data, show=False,
                save_path=os.path.join(viz_dir, 'price_trends.png')
            )
        
        print(f"\nAll visualizations saved to {viz_dir}")
    
    def demo_predictions(self):
        """Demonstrate prediction capabilities"""
        print("\n" + "="*70)
        print("STEP 6: DEMONSTRATION - PREDICTIONS")
        print("="*70)
        
        # Load models and preprocessor
        self.predictor.load_trained_models(['xgboost', 'random_forest'])
        self.predictor.load_historical_data()
        
        # Example 1: Single property prediction
        print("\n--- Example 1: Single Property Price Prediction ---")
        sample_property = {
            'location_type': 'Urban',
            'property_type': 'House',
            'area': 2200,
            'bedrooms': 4,
            'bathrooms': 3,
            'age': 3,
            'distance_to_city': 8,
            'infrastructure_score': 8.5,
            'interest_rate': 4.2,
            'inflation_rate': 2.3,
            'population_growth': 1.9,
            'year': 2023,
            'month': 6,
            'quarter': 2
        }
        
        print("\nProperty Details:")
        for key, value in sample_property.items():
            if key not in ['year', 'month', 'quarter']:
                print(f"  {key}: {value}")
        
        try:
            predicted_price = self.predictor.predict_single_property(
                sample_property, model_name='xgboost'
            )
            print(f"\nPredicted Price: ${predicted_price:,.2f}")
        except Exception as e:
            print(f"Error in prediction: {e}")
        
        # Example 2: Price forecast
        print("\n--- Example 2: 12-Month Price Forecast ---")
        try:
            forecast = self.predictor.forecast_property_price(
                sample_property, months_ahead=12, model_name='xgboost'
            )
            print("\nForecast Summary:")
            print(f"  Current Price:   ${forecast['predicted_price'].iloc[0]:,.2f}")
            print(f"  12-Month Price:  ${forecast['predicted_price'].iloc[-1]:,.2f}")
            print(f"  Expected Change: ${forecast['predicted_price'].iloc[-1] - forecast['predicted_price'].iloc[0]:,.2f}")
            print(f"  Change %:        {((forecast['predicted_price'].iloc[-1] / forecast['predicted_price'].iloc[0]) - 1) * 100:.2f}%")
            
            # Save forecast
            forecast_path = os.path.join(config.RESULTS_DIR, 'sample_forecast.csv')
            forecast.to_csv(forecast_path, index=False)
            print(f"\nDetailed forecast saved to {forecast_path}")
            
            # Create forecast visualization
            fig = self.visualizer.plot_interactive_forecast(
                forecast, title='12-Month Price Forecast'
            )
            forecast_html = os.path.join(config.RESULTS_DIR, 'forecast_visualization.html')
            fig.write_html(forecast_html)
            print(f"Interactive forecast chart saved to {forecast_html}")
            
        except Exception as e:
            print(f"Error in forecasting: {e}")
        
        # Example 3: Market trend analysis
        print("\n--- Example 3: Market Trend Analysis ---")
        try:
            trend = self.predictor.predict_market_trend(
                'Urban', 'House', months_ahead=12
            )
            print(f"\nMarket: {trend['location_type']} {trend['property_type']}")
            print(f"Current Avg Price:  ${trend['current_price']:,.2f}")
            print(f"12-Month Forecast:  ${trend['forecast_price']:,.2f}")
            print(f"Expected Change:    ${trend['total_change']:,.2f} ({trend['total_change_pct']:.2f}%)")
            print(f"Trend Direction:    {trend['trend']}")
            print(f"Avg Monthly Change: {trend['avg_monthly_change']:.2f}%")
        except Exception as e:
            print(f"Error in trend analysis: {e}")
        
        # Example 4: Scenario comparison
        print("\n--- Example 4: Economic Scenario Comparison ---")
        scenarios = {
            'Current': {
                'interest_rate': 4.2,
                'inflation_rate': 2.3,
                'population_growth': 1.9
            },
            'Low Interest': {
                'interest_rate': 3.0,
                'inflation_rate': 2.3,
                'population_growth': 1.9
            },
            'High Interest': {
                'interest_rate': 6.0,
                'inflation_rate': 2.3,
                'population_growth': 1.9
            },
            'High Growth': {
                'interest_rate': 4.2,
                'inflation_rate': 3.5,
                'population_growth': 2.5
            }
        }
        
        try:
            comparison = self.predictor.compare_scenarios(
                sample_property, scenarios, model_name='xgboost'
            )
            print("\nScenario Comparison:")
            print(comparison[['scenario', 'predicted_price', 'price_difference', 
                            'price_difference_pct']].to_string(index=False))
            
            # Save comparison
            comparison_path = os.path.join(config.RESULTS_DIR, 'scenario_comparison.csv')
            comparison.to_csv(comparison_path, index=False)
            print(f"\nScenario comparison saved to {comparison_path}")
        except Exception as e:
            print(f"Error in scenario comparison: {e}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*70)
        print("REAL ESTATE PRICE PREDICTION SYSTEM")
        print("AI-Based Predictive System for Housing Market Analysis")
        print("="*70)
        
        try:
            # Step 1: Generate data
            self.generate_data()
            
            # Step 2: Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data()
            
            # Step 3: Train models
            self.train_models(X_train, y_train, X_val, y_val)
            
            # Step 4: Evaluate models
            results = self.evaluate_models(X_test, y_test)
            
            # Step 5: Create visualizations
            self.create_visualizations(X_test, y_test, results)
            
            # Step 6: Demo predictions
            self.demo_predictions()
            
            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nResults saved to: {config.RESULTS_DIR}")
            print(f"Models saved to: {config.MODELS_DIR}")
            print(f"Data saved to: {config.DATA_DIR}")
            
        except Exception as e:
            print(f"\nError in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Real Estate Price Prediction System'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'generate', 'train', 'predict', 'visualize'],
        default='full',
        help='Operation mode'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=config.SAMPLE_SIZE,
        help='Number of samples to generate'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = RealEstatePricePredictionSystem()
    
    if args.mode == 'full':
        # Run complete pipeline
        system.run_full_pipeline()
    
    elif args.mode == 'generate':
        # Only generate data
        system.generate_data(n_samples=args.samples)
    
    elif args.mode == 'train':
        # Load data and train models
        X_train, X_val, X_test, y_train, y_val, y_test = system.preprocess_data()
        system.train_models(X_train, y_train, X_val, y_val)
        results = system.evaluate_models(X_test, y_test)
    
    elif args.mode == 'predict':
        # Demo predictions
        system.demo_predictions()
    
    elif args.mode == 'visualize':
        # Create visualizations
        data_path = os.path.join(config.DATA_DIR, 'housing_data.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            system.data = data
            # Create visualizations (will need model results)
            print("Loading models for visualization...")
            print("Run training mode first to generate model results")


if __name__ == "__main__":
    main()
