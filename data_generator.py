"""
Data Generator for Real Estate Price Prediction
Generates synthetic housing data with economic factors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import config

class HousingDataGenerator:
    """Generate realistic synthetic housing data"""
    
    def __init__(self, n_samples=config.SAMPLE_SIZE, random_seed=config.RANDOM_SEED):
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_temporal_data(self):
        """Generate temporal data spanning multiple years"""
        start_date = datetime(2015, 1, 1)
        dates = [start_date + timedelta(days=int(x)) 
                 for x in np.random.uniform(0, 365*8, self.n_samples)]
        return sorted(dates)
    
    def generate_economic_factors(self, dates):
        """Generate realistic economic factors over time"""
        n = len(dates)
        
        # Interest rates (tend to cycle over time)
        base_interest = 4.5
        interest_rates = base_interest + 2 * np.sin(np.arange(n) / 365 * 2 * np.pi / 3) + \
                        np.random.normal(0, 0.3, n)
        interest_rates = np.clip(interest_rates, *config.INTEREST_RATE_RANGE)
        
        # Inflation rates (correlated with interest rates)
        inflation_rates = 2.5 + 0.5 * (interest_rates - 4.5) + np.random.normal(0, 0.5, n)
        inflation_rates = np.clip(inflation_rates, *config.INFLATION_RATE_RANGE)
        
        # Population growth (relatively stable with slight trend)
        population_growth = 1.5 + 0.1 * np.arange(n) / 365 + np.random.normal(0, 0.2, n)
        population_growth = np.clip(population_growth, *config.POPULATION_GROWTH_RANGE)
        
        return interest_rates, inflation_rates, population_growth
    
    def generate_property_features(self):
        """Generate property-specific features"""
        # Location type
        location_types = np.random.choice(config.LOCATION_TYPES, self.n_samples)
        
        # Property type
        property_types = np.random.choice(config.PROPERTY_TYPES, self.n_samples)
        
        # Area (log-normal distribution for realistic distribution)
        areas = np.random.lognormal(np.log(2000), 0.5, self.n_samples)
        areas = np.clip(areas, *config.AREA_RANGE)
        
        # Bedrooms (correlated with area)
        bedrooms = np.round(areas / 800 + np.random.normal(0, 0.5, self.n_samples))
        bedrooms = np.clip(bedrooms, *config.BEDROOMS_RANGE).astype(int)
        
        # Bathrooms (correlated with bedrooms)
        bathrooms = np.round(bedrooms * 0.75 + np.random.normal(0, 0.3, self.n_samples))
        bathrooms = np.clip(bathrooms, *config.BATHROOMS_RANGE).astype(int)
        
        # Age of property
        age = np.random.exponential(15, self.n_samples)
        age = np.clip(age, *config.AGE_RANGE)
        
        # Distance to city center
        distance_to_city = np.random.gamma(2, 5, self.n_samples)
        distance_to_city = np.clip(distance_to_city, *config.DISTANCE_TO_CITY_RANGE)
        
        # Infrastructure score (higher for urban, lower distance)
        infrastructure_score = 7 - 0.1 * distance_to_city + \
                              np.random.normal(0, 1, self.n_samples)
        infrastructure_score = np.clip(infrastructure_score, 
                                      *config.INFRASTRUCTURE_SCORE_RANGE)
        
        return (location_types, property_types, areas, bedrooms, bathrooms, 
                age, distance_to_city, infrastructure_score)
    
    def calculate_price(self, df):
        """Calculate realistic house prices based on features"""
        # Base price calculation
        base_price = 100000
        
        # Area factor (primary driver)
        area_factor = df['area'] * 150
        
        # Location multipliers
        location_multipliers = {
            'Metropolitan': 1.5,
            'Urban': 1.3,
            'Suburban': 1.0,
            'Rural': 0.7
        }
        location_factor = df['location_type'].map(location_multipliers)
        
        # Property type multipliers
        property_multipliers = {
            'Villa': 1.4,
            'House': 1.2,
            'Condo': 1.0,
            'Townhouse': 0.95,
            'Apartment': 0.85
        }
        property_factor = df['property_type'].map(property_multipliers)
        
        # Economic factors
        # Higher interest rates -> lower prices
        interest_impact = -10000 * (df['interest_rate'] - 4.5)
        
        # Higher inflation -> higher nominal prices
        inflation_impact = 5000 * (df['inflation_rate'] - 2.5)
        
        # Population growth -> higher demand -> higher prices
        population_impact = 15000 * df['population_growth']
        
        # Infrastructure impact
        infrastructure_impact = 8000 * df['infrastructure_score']
        
        # Age depreciation
        age_depreciation = -2000 * df['age']
        
        # Distance penalty
        distance_penalty = -3000 * df['distance_to_city']
        
        # Room bonuses
        bedroom_bonus = 15000 * df['bedrooms']
        bathroom_bonus = 10000 * df['bathrooms']
        
        # Calculate final price
        price = (base_price + area_factor + interest_impact + inflation_impact +
                population_impact + infrastructure_impact + age_depreciation +
                distance_penalty + bedroom_bonus + bathroom_bonus)
        
        # Apply location and property type multipliers
        price = price * location_factor * property_factor
        
        # Add random noise (market volatility)
        noise = np.random.normal(1, 0.1, len(df))
        price = price * noise
        
        # Ensure positive prices
        price = np.maximum(price, 50000)
        
        return price
    
    def generate_dataset(self):
        """Generate complete housing dataset"""
        print("Generating temporal data...")
        dates = self.generate_temporal_data()
        
        print("Generating economic factors...")
        interest_rates, inflation_rates, population_growth = \
            self.generate_economic_factors(dates)
        
        print("Generating property features...")
        (location_types, property_types, areas, bedrooms, bathrooms,
         age, distance_to_city, infrastructure_score) = self.generate_property_features()
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'location_type': location_types,
            'property_type': property_types,
            'area': areas,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'distance_to_city': distance_to_city,
            'infrastructure_score': infrastructure_score,
            'interest_rate': interest_rates,
            'inflation_rate': inflation_rates,
            'population_growth': population_growth
        })
        
        print("Calculating house prices...")
        df['price'] = self.calculate_price(df)
        
        # Add year and month for time series analysis
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Generated {len(df)} housing records")
        return df
    
    def save_dataset(self, df, filename='housing_data.csv'):
        """Save dataset to CSV file"""
        filepath = os.path.join(config.DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return filepath


if __name__ == "__main__":
    import os
    
    # Generate dataset
    generator = HousingDataGenerator()
    df = generator.generate_dataset()
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(df.describe())
    
    print("\n=== Sample Data ===")
    print(df.head(10))
    
    # Save dataset
    generator.save_dataset(df)
