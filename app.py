"""
Flask Web Application for Real Estate Price Prediction System
Run on localhost to access via browser
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
from datetime import datetime
import json

# Import custom modules
from predictor import HousingPricePredictor
from models import HousingPriceModels
import config

app = Flask(__name__)

# Initialize predictor globally
predictor = HousingPricePredictor()

# Load models on startup
try:
    predictor.load_trained_models(['xgboost', 'random_forest', 'gradient_boosting'])
    predictor.load_historical_data()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Please run 'python main.py --mode train' first")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            # Get form data
            property_data = {
                'location_type': request.form.get('location_type'),
                'property_type': request.form.get('property_type'),
                'area': float(request.form.get('area')),
                'bedrooms': int(request.form.get('bedrooms')),
                'bathrooms': int(request.form.get('bathrooms')),
                'age': float(request.form.get('age')),
                'distance_to_city': float(request.form.get('distance_to_city')),
                'infrastructure_score': float(request.form.get('infrastructure_score')),
                'interest_rate': float(request.form.get('interest_rate')),
                'inflation_rate': float(request.form.get('inflation_rate')),
                'population_growth': float(request.form.get('population_growth')),
                'year': datetime.now().year,
                'month': datetime.now().month,
                'quarter': (datetime.now().month - 1) // 3 + 1
            }
            
            # Make predictions with different models
            model_name = request.form.get('model', 'xgboost')
            price = predictor.predict_single_property(property_data, model_name=model_name)
            
            # Get predictions from all models for comparison
            predictions = {}
            for model in ['xgboost', 'random_forest', 'gradient_boosting']:
                try:
                    predictions[model] = predictor.predict_single_property(property_data, model_name=model)
                except:
                    pass
            
            return render_template('predict.html', 
                                 prediction=price,
                                 predictions=predictions,
                                 property_data=property_data,
                                 model_used=model_name)
        
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')


@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """Forecast page"""
    if request.method == 'POST':
        try:
            # Get form data
            property_data = {
                'location_type': request.form.get('location_type'),
                'property_type': request.form.get('property_type'),
                'area': float(request.form.get('area')),
                'bedrooms': int(request.form.get('bedrooms')),
                'bathrooms': int(request.form.get('bathrooms')),
                'age': float(request.form.get('age')),
                'distance_to_city': float(request.form.get('distance_to_city')),
                'infrastructure_score': float(request.form.get('infrastructure_score')),
                'interest_rate': float(request.form.get('interest_rate')),
                'inflation_rate': float(request.form.get('inflation_rate')),
                'population_growth': float(request.form.get('population_growth')),
                'year': datetime.now().year,
                'month': datetime.now().month,
                'quarter': (datetime.now().month - 1) // 3 + 1
            }
            
            months = int(request.form.get('months', 12))
            model_name = request.form.get('model', 'xgboost')
            
            # Generate forecast
            forecast_df = predictor.forecast_property_price(
                property_data, 
                months_ahead=months,
                model_name=model_name
            )
            
            # Convert to JSON for chart
            forecast_data = {
                'dates': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': forecast_df['predicted_price'].tolist(),
                'changes': forecast_df['price_change_pct'].fillna(0).tolist(),
                'interest_rates': forecast_df['interest_rate'].tolist(),
                'inflation_rates': forecast_df['inflation_rate'].tolist()
            }
            
            # Calculate summary stats
            current_price = forecast_df['predicted_price'].iloc[0]
            future_price = forecast_df['predicted_price'].iloc[-1]
            total_change = future_price - current_price
            total_change_pct = (total_change / current_price) * 100
            
            summary = {
                'current_price': current_price,
                'future_price': future_price,
                'total_change': total_change,
                'total_change_pct': total_change_pct,
                'trend': 'Increasing' if total_change > 0 else 'Decreasing'
            }
            
            return render_template('forecast.html',
                                 forecast_data=json.dumps(forecast_data),
                                 summary=summary,
                                 property_data=property_data,
                                 months=months)
        
        except Exception as e:
            return render_template('forecast.html', error=str(e))
    
    return render_template('forecast.html')


@app.route('/scenarios', methods=['GET', 'POST'])
def scenarios():
    """Economic scenarios comparison page"""
    if request.method == 'POST':
        try:
            # Get base property data
            property_data = {
                'location_type': request.form.get('location_type'),
                'property_type': request.form.get('property_type'),
                'area': float(request.form.get('area')),
                'bedrooms': int(request.form.get('bedrooms')),
                'bathrooms': int(request.form.get('bathrooms')),
                'age': float(request.form.get('age')),
                'distance_to_city': float(request.form.get('distance_to_city')),
                'infrastructure_score': float(request.form.get('infrastructure_score')),
                'interest_rate': float(request.form.get('interest_rate')),
                'inflation_rate': float(request.form.get('inflation_rate')),
                'population_growth': float(request.form.get('population_growth')),
                'year': datetime.now().year,
                'month': datetime.now().month,
                'quarter': (datetime.now().month - 1) // 3 + 1
            }
            
            # Define scenarios
            scenarios_dict = {
                'Current': {
                    'interest_rate': property_data['interest_rate'],
                    'inflation_rate': property_data['inflation_rate'],
                    'population_growth': property_data['population_growth']
                },
                'Low Interest': {
                    'interest_rate': 3.0,
                    'inflation_rate': property_data['inflation_rate'],
                    'population_growth': property_data['population_growth']
                },
                'High Interest': {
                    'interest_rate': 6.5,
                    'inflation_rate': property_data['inflation_rate'],
                    'population_growth': property_data['population_growth']
                },
                'Economic Boom': {
                    'interest_rate': property_data['interest_rate'],
                    'inflation_rate': 4.0,
                    'population_growth': 3.0
                },
                'Recession': {
                    'interest_rate': 2.0,
                    'inflation_rate': 1.0,
                    'population_growth': 0.5
                }
            }
            
            model_name = request.form.get('model', 'xgboost')
            
            # Compare scenarios
            comparison = predictor.compare_scenarios(
                property_data,
                scenarios_dict,
                model_name=model_name
            )
            
            # Convert to dict for template
            scenarios_results = comparison.to_dict('records')
            
            return render_template('scenarios.html',
                                 scenarios_results=scenarios_results,
                                 property_data=property_data)
        
        except Exception as e:
            return render_template('scenarios.html', error=str(e))
    
    return render_template('scenarios.html')


@app.route('/analytics')
def analytics():
    """Analytics dashboard page"""
    try:
        # Load model scores
        models_path = os.path.join(config.MODELS_DIR, 'model_scores.pkl')
        import joblib
        model_scores = joblib.load(models_path)
        
        # Prepare data for charts
        model_names = list(model_scores.keys())
        rmse_values = [model_scores[m]['rmse'] for m in model_names]
        r2_values = [model_scores[m]['r2'] for m in model_names]
        mape_values = [model_scores[m]['mape'] for m in model_names]
        
        chart_data = {
            'models': model_names,
            'rmse': rmse_values,
            'r2': r2_values,
            'mape': mape_values
        }
        
        # Get best model
        best_model = min(model_scores.items(), key=lambda x: x[1]['rmse'])
        
        return render_template('analytics.html',
                             chart_data=json.dumps(chart_data),
                             model_scores=model_scores,
                             best_model=best_model[0])
    
    except Exception as e:
        return render_template('analytics.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        prediction = predictor.predict_single_property(data, model_name=data.get('model', 'xgboost'))
        return jsonify({'prediction': float(prediction), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏠 REAL ESTATE PRICE PREDICTION WEB APPLICATION")
    print("="*70)
    print("\n🌐 Starting Flask server...")
    print("\n📍 Access the application at: http://localhost:5000")
    print("📍 Alternative: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
