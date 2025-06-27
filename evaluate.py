"""
evaluate.py
Evaluates the trained model and visualizes the results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using MAE and R² score.
    
    Parameters:
        model: Trained model
        X_test: Feature test data
        y_test: Target test data
    
    Returns:
        Dictionary with MAE and R² score
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "MAE": round(mae, 3),
        "R2 Score": round(r2, 3),
        "y_pred": y_pred  # for plotting
    }

def plot_predictions(y_test, y_pred):
    """
    Plots actual vs predicted crop yields.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='seagreen', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Crop Yield (tons/ha)")
    plt.ylabel("Predicted Crop Yield (tons/ha)")
    plt.title("Actual vs Predicted Crop Yield")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots feature importances from the trained model.
    """
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance for Crop Yield Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
