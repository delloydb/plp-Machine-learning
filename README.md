# Machine Learning plp assignment 
## 📘 Crop Yield Prediction Using AI
An AI-Driven Tool to Support SDG 2: Zero Hunger

# 📌 Project Overview
This project demonstrates how supervised machine learning can help tackle food insecurity by predicting crop yields using environmental and historical agricultural data. It contributes to the United Nations Sustainable Development Goal 2 (Zero Hunger) by enabling data-driven agricultural decision-making for farmers, NGOs, and policymakers.

#🎯 Goal
To build a predictive model that estimates crop yield (in tons per hectare) using a set of key features such as:
Rainfall
Temperature
Soil type
NDVI (vegetation health)
Past yield records

# 🧠 Machine Learning Approach
Type: Supervised Learning
Algorithm: Random Forest Regressor
Metric Evaluation:
Mean Absolute Error (MAE)
R² Score (Explained Variance)

# 🛠 Features
Synthetic dataset generation to simulate real-world crop factors
One-hot encoding of categorical features (soil type)
Feature scaling using StandardScaler
Model training, evaluation, and visualization
Feature importance analysis
Clean, modular Python codebase

# 🗂 File Structure

ai_crop_yield_prediction/
│
├── data_loader.py         # Data generation and preprocessing
├── model.py               # Model training and saving
├── evaluate.py            # Model evaluation and plotting
├── utils.py               # Helper utilities (encoding)
├── main.py                # Main script to run the full pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
# ⚙️ How to Run
Clone the Repository

git clone https://github.com/yourusername/ai_crop_yield_prediction.git
cd ai_crop_yield_prediction
Install Dependencies

pip install -r requirements.txt
Run the Project

python main.py
# 📊 Results
Mean Absolute Error (MAE): ~0.46 tons/ha
R² Score: ~0.68
The model correctly identifies key influencers like NDVI, past yield, and rainfall.

# 🔐 Ethical Reflection
Bias Awareness: Real-world deployment requires de-biasing across regions, crop types, and climate zones.
Fairness: This solution supports equal access to prediction tools for farmers in under-resourced regions.
Sustainability: Promotes data-driven agriculture, reducing waste and optimizing food production.

# 📎 Future Enhancements
Integrate real-world datasets from FAO, World Bank, or satellite APIs
Add regional customization per country or crop type
Deploy as a web app using Streamlit or Flask
Enable mobile access for rural farmers

# 🙌 License
This project is open-source and free to use under the MIT License.

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
