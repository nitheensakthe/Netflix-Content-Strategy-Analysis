"""
Configuration file for Real Estate Price Prediction System
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data generation parameters
SAMPLE_SIZE = 10000
RANDOM_SEED = 42

# Feature columns
LOCATION_TYPES = ['Urban', 'Suburban', 'Rural', 'Metropolitan']
PROPERTY_TYPES = ['Apartment', 'House', 'Villa', 'Condo', 'Townhouse']

# Economic factors ranges
INTEREST_RATE_RANGE = (2.5, 7.5)  # percentage
INFLATION_RATE_RANGE = (1.0, 6.0)  # percentage
POPULATION_GROWTH_RANGE = (-0.5, 3.0)  # percentage
INFRASTRUCTURE_SCORE_RANGE = (1, 10)  # score out of 10

# House features ranges
AREA_RANGE = (500, 5000)  # square feet
BEDROOMS_RANGE = (1, 6)
BATHROOMS_RANGE = (1, 5)
AGE_RANGE = (0, 50)  # years
DISTANCE_TO_CITY_RANGE = (0, 50)  # km

# Model parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
CV_FOLDS = 5

# Neural Network parameters
NN_EPOCHS = 100
NN_BATCH_SIZE = 32
NN_LEARNING_RATE = 0.001

# XGBoost parameters
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 7
XGB_LEARNING_RATE = 0.1

# Random Forest parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15

# Feature importance threshold
FEATURE_IMPORTANCE_THRESHOLD = 0.01
