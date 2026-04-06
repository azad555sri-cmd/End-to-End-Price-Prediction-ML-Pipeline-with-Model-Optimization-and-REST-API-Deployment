# 🥑 End-to-End Price Prediction ML Pipeline

## 📌 Overview
This project is a **production-ready machine learning pipeline** designed to predict product prices using historical data. It covers the complete ML lifecycle including data preprocessing, feature engineering, model training, evaluation, optimization, and deployment via a REST API.

The system is built to simulate real-world scenarios where scalable and efficient ML solutions are required for business decision-making.

---

## 🚀 Features
- End-to-end machine learning pipeline
- Data preprocessing and feature engineering
- Multiple model training and comparison
- Hyperparameter tuning using GridSearchCV
- Model evaluation using MAE, MSE, and R² score
- Model persistence using Pickle
- REST API deployment using Flask
- Scalable and production-ready structure

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
- **Deployment:** Flask
- **Tools:** VS Code, Git

---

## 📂 Project Structure
price-prediction-ml-pipeline/
│
├── model.py # Main ML pipeline
├── app.py # Flask API for deployment
├── avocado.csv # Dataset
├── model.pkl # Trained model
├── scaler.pkl # Scaler object
├── features.pkl # Feature columns
├── README.md

📊 Model Evaluation

Models used:
- Linear Regression
- Random Forest Regressor
- Support Vector Regressor
- XGBoost Regressor

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

🎯 Key Highlights
- Designed a scalable ML pipeline for real-world use cases
- Implemented model comparison and selection strategy
- Applied hyperparameter tuning for performance optimization
- Built a REST API for real-time predictions

🚀 Future Improvements
- Add Streamlit UI for interactive predictions
- Deploy on cloud platforms (AWS / Render)
- Implement advanced feature engineering
- Integrate CI/CD pipeline
