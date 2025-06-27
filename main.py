"""
main.py
Main script to run the AI crop yield prediction pipeline.
"""

from data_loader import generate_synthetic_data, preprocess_data
from model import train_model, save_model
from evaluate import evaluate_model, plot_predictions, plot_feature_importance
import pandas as pd

def main():
    print("🔄 Generating synthetic crop yield data...")
    data = generate_synthetic_data()
    
    print("📊 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    print("🤖 Training model...")
    model = train_model(X_train, y_train)

    print("💾 Saving model...")
    save_model(model)

    print("📈 Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    print(f"✅ Model Evaluation:\nMAE: {results['MAE']} tons/ha\nR² Score: {results['R2 Score']}")

    print("📉 Plotting predictions...")
    plot_predictions(y_test, results['y_pred'])

    print("🔍 Plotting feature importances...")
    feature_names = data.drop(columns='crop_yield_tph')
    feature_names = pd.get_dummies(feature_names, columns=['soil_type'], drop_first=True).columns
    plot_feature_importance(model, feature_names)

if __name__ == "__main__":
    main()
