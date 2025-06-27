"""
model.py
Defines and trains the crop yield prediction model.
"""

from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest Regressor on the training data.
    
    Parameters:
        X_train: Feature training data
        y_train: Target training data
        n_estimators: Number of trees in the forest
        random_state: Seed for reproducibility
    
    Returns:
        Trained model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='crop_yield_model.pkl'):
    """
    Saves the trained model to a file using joblib.
    
    Parameters:
        model: Trained model
        filename: File path to save the model
    """
    joblib.dump(model, filename)
