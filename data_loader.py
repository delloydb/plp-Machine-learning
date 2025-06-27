"""
data_loader.py
Generates and preprocesses a synthetic dataset for crop yield prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import encode_soil_type

def generate_synthetic_data(seed=42, sample_size=500):
    np.random.seed(seed)
    data = pd.DataFrame({
        'rainfall_mm': np.random.normal(600, 150, sample_size),
        'temperature_c': np.random.normal(25, 3, sample_size),
        'soil_type': np.random.choice(['clay', 'sandy', 'loam'], sample_size),
        'ndvi': np.random.uniform(0.2, 0.8, sample_size),
        'past_yield_tph': np.random.normal(3.0, 0.7, sample_size)
    })
    
    # Add a synthetic target variable (crop yield)
    data['crop_yield_tph'] = (
        0.003 * data['rainfall_mm'] +
        -0.05 * data['temperature_c'] +
        4 * data['ndvi'] +
        0.7 * data['past_yield_tph'] +
        np.random.normal(0, 0.5, sample_size)
    )
    
    return data

def preprocess_data(df):
    df_encoded = encode_soil_type(df)
    X = df_encoded.drop(columns='crop_yield_tph')
    y = df_encoded['crop_yield_tph']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
