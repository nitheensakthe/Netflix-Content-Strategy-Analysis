"""
Quick Example - Real Estate Price Prediction
A simple script to get started with the system
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import HousingPricePredictor


def quick_prediction_example():
    """Quick example of making a price prediction"""
    
    print("="*70)
    print("QUICK EXAMPLE: Real Estate Price Prediction")
    print("="*70)
    
    # Initialize predictor
    predictor = HousingPricePredictor()
    
    # Load models (make sure you've run main.py first)
    print("\nLoading trained models...")
    try:
        predictor.load_trained_models(['xgboost', 'random_forest'])
        predictor.load_historical_data()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"\n⚠ Error loading models: {e}")
        print("\nPlease run the following command first:")
        print("  python main.py --mode full")
        print("\nThis will generate data and train the models.")
        return
    
    # Define a sample property
    print("\n" + "-"*70)
    print("EXAMPLE PROPERTY")
    print("-"*70)
    
    sample_property = {
        'location_type': 'Urban',
        'property_type': 'House',
        'area': 2500,              # 2500 sq ft
        'bedrooms': 4,
        'bathrooms': 3,
        'age': 5,                  # 5 years old
        'distance_to_city': 12,    # 12 km from city center
        'infrastructure_score': 8.5,
        'interest_rate': 4.5,      # Current interest rate
        'inflation_rate': 2.8,     # Current inflation
        'population_growth': 2.0,  # Population growth rate
        'year': 2023,
        'month': 6,
        'quarter': 2
    }
    
    print("\nProperty Details:")
    print(f"  Type: {sample_property['property_type']}")
    print(f"  Location: {sample_property['location_type']}")
    print(f"  Size: {sample_property['area']} sq ft")
    print(f"  Bedrooms: {sample_property['bedrooms']}")
    print(f"  Bathrooms: {sample_property['bathrooms']}")
    print(f"  Age: {sample_property['age']} years")
    print(f"  Distance to city: {sample_property['distance_to_city']} km")
    print(f"  Infrastructure score: {sample_property['infrastructure_score']}/10")
    
    print("\nEconomic Conditions:")
    print(f"  Interest Rate: {sample_property['interest_rate']}%")
    print(f"  Inflation Rate: {sample_property['inflation_rate']}%")
    print(f"  Population Growth: {sample_property['population_growth']}%")
    
    # Make prediction
    print("\n" + "-"*70)
    print("PREDICTIONS")
    print("-"*70)
    
    try:
        # Predict with XGBoost
        price_xgb = predictor.predict_single_property(sample_property, model_name='xgboost')
        print(f"\nXGBoost Prediction:      ${price_xgb:,.2f}")
        
        # Predict with Random Forest
        price_rf = predictor.predict_single_property(sample_property, model_name='random_forest')
        print(f"Random Forest Prediction: ${price_rf:,.2f}")
        
        # Average prediction
        avg_price = (price_xgb + price_rf) / 2
        print(f"\nAverage Prediction:       ${avg_price:,.2f}")
        
    except Exception as e:
        print(f"\nError making prediction: {e}")
        return
    
    # 12-month forecast
    print("\n" + "-"*70)
    print("12-MONTH PRICE FORECAST")
    print("-"*70)
    
    try:
        forecast = predictor.forecast_property_price(
            sample_property, 
            months_ahead=12, 
            model_name='xgboost'
        )
        
        current_price = forecast['predicted_price'].iloc[0]
        future_price = forecast['predicted_price'].iloc[-1]
        change = future_price - current_price
        change_pct = (change / current_price) * 100
        
        print(f"\nCurrent Estimated Price: ${current_price:,.2f}")
        print(f"12-Month Forecast:       ${future_price:,.2f}")
        print(f"Expected Change:         ${change:,.2f} ({change_pct:+.2f}%)")
        
        if change_pct > 0:
            print(f"\n📈 Market Outlook: INCREASING (Good time to buy/hold)")
        else:
            print(f"\n📉 Market Outlook: DECREASING (Wait for better prices)")
        
    except Exception as e:
        print(f"\nError making forecast: {e}")
        return
    
    # Economic scenarios
    print("\n" + "-"*70)
    print("ECONOMIC SCENARIO ANALYSIS")
    print("-"*70)
    
    scenarios = {
        'Current Conditions': {
            'interest_rate': 4.5,
            'inflation_rate': 2.8,
            'population_growth': 2.0
        },
        'Low Interest Rate': {
            'interest_rate': 3.0,
            'inflation_rate': 2.8,
            'population_growth': 2.0
        },
        'High Interest Rate': {
            'interest_rate': 6.5,
            'inflation_rate': 2.8,
            'population_growth': 2.0
        },
        'Economic Boom': {
            'interest_rate': 4.5,
            'inflation_rate': 4.0,
            'population_growth': 3.5
        }
    }
    
    try:
        comparison = predictor.compare_scenarios(
            sample_property, 
            scenarios, 
            model_name='xgboost'
        )
        
        print("\nImpact of Different Economic Conditions:\n")
        for _, row in comparison.iterrows():
            scenario = row['scenario']
            price = row['predicted_price']
            diff = row['price_difference']
            diff_pct = row['price_difference_pct']
            
            if diff_pct > 0:
                direction = "↑"
            elif diff_pct < 0:
                direction = "↓"
            else:
                direction = "="
            
            print(f"{scenario:25s} ${price:>12,.2f}  {direction} ${diff:>+10,.2f} ({diff_pct:>+6.2f}%)")
        
    except Exception as e:
        print(f"\nError in scenario analysis: {e}")
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)
    print("\nTo explore more features:")
    print("  • Run: python main.py --mode full")
    print("  • Check visualizations in: results/visualizations/")
    print("  • Read the full documentation: README.md")
    print("="*70)


if __name__ == "__main__":
    quick_prediction_example()
