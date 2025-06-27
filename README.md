# plp-Machine-learning Documentation 
AI for Sustainable Development Report
Project Title:
AI-Driven Crop Yield Prediction for Food Security in Sub-Saharan Africa

# SDG Addressed:
SDG 2 – Zero Hunger

# Problem Statement:
Millions of smallholder farmers in drought-prone regions of Sub-Saharan Africa face uncertainty in agricultural productivity due to climate variability and lack of predictive tools. This project aims to build a machine learning model that forecasts crop yield to support decision-making in food security and agricultural planning.

# Machine Learning Approach Used:
Supervised Learning — using a Random Forest Regressor.

# Dataset:
A synthetic dataset simulating real-world features such as:
Rainfall
Temperature
Soil Type
NDVI (Normalized Difference Vegetation Index)
Past Yield
These features were used to train a model that predicts crop yield in tons per hectare.

# Tools & Frameworks:
Python (Jupyter Notebook)
Scikit-learn, Seaborn, Matplotlib
Data preprocessing using StandardScaler

# Results:
Mean Absolute Error (MAE): 0.462 tons/ha
R² Score: 0.683
Top Features: NDVI, Past Yield, Rainfall
These metrics show that the model predicts yield with good accuracy and identifies key environmental factors influencing productivity.

# Ethical Considerations & Fairness:
Bias Awareness: Underrepresented regions in training data can skew model accuracy.
Fairness Promotion: The tool can provide equal access to yield predictions for marginalized farming communities.
Sustainability: Helps NGOs, governments, and farmers optimize agricultural resources and combat hunger.

# Impact Statement:
This project demonstrates how AI can be a bridge between innovation and sustainability by empowering communities to plan food production more effectively. With further enhancement and real-world data, the model can be deployed as a mobile or web application for use by farmers and policymakers alike.
