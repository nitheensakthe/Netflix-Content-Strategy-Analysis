"""
Prediction and Forecasting Module for Real Estate Prices
Handles future price predictions and market trend forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import config
from data_preprocessor import DataPreprocessor
from models import HousingPriceModels


class HousingPricePredictor:
    """Make predictions and forecasts for housing prices"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = HousingPriceModels()
        self.historical_data = None
        
    def load_trained_models(self, model_names=None):
        """Load pre-trained models"""
        if model_names is None:
            model_names = ['xgboost', 'random_forest']
        
        for model_name in model_names:
            try:
                self.models.load_model(model_name)
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
        
        # Load preprocessor
        try:
            self.preprocessor.load_preprocessor()
            print("Loaded preprocessor")
        except Exception as e:
            print(f"Warning: Could not load preprocessor: {e}")
    
    def load_historical_data(self, filepath=None):
        """Load historical housing data"""
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, 'housing_data.csv')
        
        self.historical_data = pd.read_csv(filepath)
        if 'date' in self.historical_data.columns:
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        
        print(f"Loaded {len(self.historical_data)} historical records")
        return self.historical_data
    
    def predict_single_property(self, property_data, model_name='xgboost'):
        """
        Predict price for a single property
        
        Parameters:
        -----------
        property_data : dict
            Dictionary with property features
        model_name : str
            Name of the model to use for prediction
        
        Returns:
        --------
        predicted_price : float
        """
        # Create DataFrame from property data
        df = pd.DataFrame([property_data])
        
        # Add date if missing (for temporal features)
        if 'date' not in df.columns:
            df['date'] = datetime.now()
        
        # Preprocess
        X, _, _ = self.preprocessor.preprocess(df, fit=False, target_col='price')
        
        # Predict
        prediction = self.models.predict(model_name, X)[0]
        
        return prediction
    
    def predict_batch(self, properties_df, model_name='xgboost'):
        """
        Predict prices for multiple properties
        
        Parameters:
        -----------
        properties_df : DataFrame
            DataFrame with property features
        model_name : str
            Name of the model to use
        
        Returns:
        --------
        predictions : array
        """
        # Add date if missing (for temporal features)
        if 'date' not in properties_df.columns:
            properties_df = properties_df.copy()
            properties_df['date'] = datetime.now()
        
        # Preprocess
        X, _, _ = self.preprocessor.preprocess(properties_df, fit=False, target_col='price')
        
        # Predict
        predictions = self.models.predict(model_name, X)
        
        return predictions
    
    def forecast_economic_factors(self, start_date, periods, base_factors=None):
        """
        Forecast future economic factors
        
        Parameters:
        -----------
        start_date : datetime
            Starting date for forecast
        periods : int
            Number of periods to forecast
        base_factors : dict
            Current economic factors as baseline
        
        Returns:
        --------
        forecasted_factors : DataFrame
        """
        if base_factors is None:
            # Use recent averages from historical data
            if self.historical_data is not None:
                recent_data = self.historical_data.tail(30)
                base_factors = {
                    'interest_rate': recent_data['interest_rate'].mean(),
                    'inflation_rate': recent_data['inflation_rate'].mean(),
                    'population_growth': recent_data['population_growth'].mean()
                }
            else:
                # Default values
                base_factors = {
                    'interest_rate': 4.5,
                    'inflation_rate': 2.5,
                    'population_growth': 1.5
                }
        
        # Generate future dates
        dates = [start_date + timedelta(days=30*i) for i in range(periods)]
        
        # Simple trend-based forecasting with random walk
        np.random.seed(config.RANDOM_SEED)
        
        interest_rates = [base_factors['interest_rate']]
        inflation_rates = [base_factors['inflation_rate']]
        population_growth = [base_factors['population_growth']]
        
        for i in range(1, periods):
            # Interest rate random walk with mean reversion
            ir_change = np.random.normal(0, 0.1) - 0.05 * (interest_rates[-1] - 4.5)
            new_ir = interest_rates[-1] + ir_change
            interest_rates.append(np.clip(new_ir, *config.INTEREST_RATE_RANGE))
            
            # Inflation rate (somewhat correlated with interest rate)
            inf_change = np.random.normal(0, 0.15) + 0.3 * ir_change
            new_inf = inflation_rates[-1] + inf_change
            inflation_rates.append(np.clip(new_inf, *config.INFLATION_RATE_RANGE))
            
            # Population growth (relatively stable)
            pop_change = np.random.normal(0, 0.05)
            new_pop = population_growth[-1] + pop_change
            population_growth.append(np.clip(new_pop, *config.POPULATION_GROWTH_RANGE))
        
        forecasted_df = pd.DataFrame({
            'date': dates,
            'interest_rate': interest_rates,
            'inflation_rate': inflation_rates,
            'population_growth': population_growth
        })
        
        return forecasted_df
    
    def forecast_property_price(self, property_data, months_ahead=12, model_name='xgboost'):
        """
        Forecast property price for future months
        
        Parameters:
        -----------
        property_data : dict
            Current property features
        months_ahead : int
            Number of months to forecast
        model_name : str
            Model to use for prediction
        
        Returns:
        --------
        forecast_df : DataFrame
            Forecasted prices with dates
        """
        # Get current date or use property date
        if 'date' in property_data:
            start_date = pd.to_datetime(property_data['date'])
        else:
            start_date = datetime.now()
        
        # Forecast economic factors
        economic_forecast = self.forecast_economic_factors(
            start_date, months_ahead
        )
        
        # Create property data for each future period
        forecast_data = []
        
        for _, row in economic_forecast.iterrows():
            prop_copy = property_data.copy()
            prop_copy['date'] = row['date']
            prop_copy['interest_rate'] = row['interest_rate']
            prop_copy['inflation_rate'] = row['inflation_rate']
            prop_copy['population_growth'] = row['population_growth']
            
            # Update year and month
            prop_copy['year'] = row['date'].year
            prop_copy['month'] = row['date'].month
            prop_copy['quarter'] = (row['date'].month - 1) // 3 + 1
            
            forecast_data.append(prop_copy)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast_data)
        
        # Make predictions
        predictions = self.predict_batch(forecast_df, model_name)
        
        # Add predictions to forecast
        forecast_df['predicted_price'] = predictions
        
        # Calculate price change
        forecast_df['price_change'] = forecast_df['predicted_price'].diff()
        forecast_df['price_change_pct'] = forecast_df['predicted_price'].pct_change() * 100
        
        return forecast_df[['date', 'predicted_price', 'price_change', 'price_change_pct',
                           'interest_rate', 'inflation_rate', 'population_growth']]
    
    def predict_market_trend(self, location_type, property_type, months_ahead=12):
        """
        Predict market trend for a specific location and property type
        
        Parameters:
        -----------
        location_type : str
            Type of location (Urban, Suburban, etc.)
        property_type : str
            Type of property (House, Apartment, etc.)
        months_ahead : int
            Number of months to forecast
        
        Returns:
        --------
        trend_analysis : dict
        """
        # Create representative property for this market segment
        if self.historical_data is not None:
            # Use median values from historical data for this segment
            segment_data = self.historical_data[
                (self.historical_data['location_type'] == location_type) &
                (self.historical_data['property_type'] == property_type)
            ]
            
            if len(segment_data) > 0:
                representative_property = {
                    'location_type': location_type,
                    'property_type': property_type,
                    'area': segment_data['area'].median(),
                    'bedrooms': int(segment_data['bedrooms'].median()),
                    'bathrooms': int(segment_data['bathrooms'].median()),
                    'age': segment_data['age'].median(),
                    'distance_to_city': segment_data['distance_to_city'].median(),
                    'infrastructure_score': segment_data['infrastructure_score'].median(),
                }
            else:
                # Use defaults if no historical data for this segment
                representative_property = self._get_default_property(location_type, property_type)
        else:
            representative_property = self._get_default_property(location_type, property_type)
        
        # Forecast prices
        forecast_df = self.forecast_property_price(
            representative_property, months_ahead
        )
        
        # Analyze trend
        trend_analysis = {
            'location_type': location_type,
            'property_type': property_type,
            'current_price': forecast_df['predicted_price'].iloc[0],
            'forecast_price': forecast_df['predicted_price'].iloc[-1],
            'total_change': forecast_df['predicted_price'].iloc[-1] - forecast_df['predicted_price'].iloc[0],
            'total_change_pct': ((forecast_df['predicted_price'].iloc[-1] / 
                                 forecast_df['predicted_price'].iloc[0]) - 1) * 100,
            'avg_monthly_change': forecast_df['price_change_pct'].mean(),
            'trend': 'Increasing' if forecast_df['predicted_price'].iloc[-1] > 
                    forecast_df['predicted_price'].iloc[0] else 'Decreasing',
            'forecast_data': forecast_df
        }
        
        return trend_analysis
    
    def _get_default_property(self, location_type, property_type):
        """Get default property features"""
        defaults = {
            'location_type': location_type,
            'property_type': property_type,
            'area': 1800,
            'bedrooms': 3,
            'bathrooms': 2,
            'age': 10,
            'distance_to_city': 15,
            'infrastructure_score': 7,
            'interest_rate': 4.5,
            'inflation_rate': 2.5,
            'population_growth': 1.5,
            'year': 2023,
            'month': 1,
            'quarter': 1
        }
        return defaults
    
    def compare_scenarios(self, property_data, scenarios, model_name='xgboost'):
        """
        Compare different economic scenarios
        
        Parameters:
        -----------
        property_data : dict
            Base property features
        scenarios : dict
            Different economic scenarios to compare
        
        Returns:
        --------
        comparison : DataFrame
        """
        results = []
        
        for scenario_name, economic_factors in scenarios.items():
            prop_copy = property_data.copy()
            prop_copy.update(economic_factors)
            
            predicted_price = self.predict_single_property(prop_copy, model_name)
            
            results.append({
                'scenario': scenario_name,
                'predicted_price': predicted_price,
                **economic_factors
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df['price_difference'] = comparison_df['predicted_price'] - \
                                            comparison_df['predicted_price'].iloc[0]
        comparison_df['price_difference_pct'] = (comparison_df['price_difference'] / 
                                                 comparison_df['predicted_price'].iloc[0]) * 100
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    predictor = HousingPricePredictor()
    
    # Load models and data
    predictor.load_trained_models()
    predictor.load_historical_data()
    
    # Example: Predict single property
    print("=" * 70)
    print("SINGLE PROPERTY PREDICTION")
    print("=" * 70)
    
    sample_property = {
        'location_type': 'Urban',
        'property_type': 'House',
        'area': 2000,
        'bedrooms': 3,
        'bathrooms': 2,
        'age': 5,
        'distance_to_city': 10,
        'infrastructure_score': 8,
        'interest_rate': 4.0,
        'inflation_rate': 2.0,
        'population_growth': 1.8,
        'year': 2023,
        'month': 6,
        'quarter': 2
    }
    
    try:
        predicted_price = predictor.predict_single_property(sample_property)
        print(f"\nPredicted Price: ${predicted_price:,.2f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example: Forecast property price
    print("\n" + "=" * 70)
    print("PRICE FORECAST (12 months)")
    print("=" * 70)
    
    try:
        forecast = predictor.forecast_property_price(sample_property, months_ahead=12)
        print(forecast.to_string())
    except Exception as e:
        print(f"Error: {e}")
    
    # Example: Market trend analysis
    print("\n" + "=" * 70)
    print("MARKET TREND ANALYSIS")
    print("=" * 70)
    
    try:
        trend = predictor.predict_market_trend('Urban', 'House', months_ahead=12)
        print(f"\nLocation: {trend['location_type']}")
        print(f"Property Type: {trend['property_type']}")
        print(f"Current Price: ${trend['current_price']:,.2f}")
        print(f"Forecast Price (12 months): ${trend['forecast_price']:,.2f}")
        print(f"Expected Change: ${trend['total_change']:,.2f} ({trend['total_change_pct']:.2f}%)")
        print(f"Trend: {trend['trend']}")
    except Exception as e:
        print(f"Error: {e}")
