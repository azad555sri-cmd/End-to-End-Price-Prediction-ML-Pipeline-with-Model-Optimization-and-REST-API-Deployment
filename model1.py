# ==============================
# 1. IMPORTS & SETUP
# ==============================
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)

# ==============================
# 2. LOAD DATA
# ==============================
def load_data():
    print("Current Folder:", os.getcwd())
    print("Files in folder:", os.listdir())
    
    return pd.read_csv("avocado.csv", index_col=0)

# ==============================
# 3. PREPROCESSING
# ==============================
def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)

    df['type'] = df['type'].map({'conventional': 0, 'organic': 1})

    df = pd.get_dummies(df, columns=['region', 'year', 'Month'], drop_first=True)

    X = df.drop("AveragePrice", axis=1)
    y = df["AveragePrice"]

    return X, y

# ==============================
# 4. SPLIT & SCALE
# ==============================
def split_and_scale(X, y):
    cols_to_scale = [
        'Total Volume', '4046', '4225', '4770',
        'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags'
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, y_train, y_test, scaler

# ==============================
# 5. TRAIN MODELS
# ==============================
def train_models(X_train, y_train):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(use_label_encoder=False, eval_metric='rmse')
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

# ==============================
# 6. EVALUATE MODELS
# ==============================
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        pred = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred)
        }

    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    return results_df

# ==============================
# 7. HYPERPARAMETER TUNING
# ==============================
def tune_random_forest(X_train, y_train):
    params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    }

    grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=3)
    grid.fit(X_train, y_train)

    return grid.best_estimator_

# ==============================
# 8. SAVE MODEL
# ==============================
def save_model(model, scaler, columns):
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(columns, open("features.pkl", "wb"))
    print("\nModel, scaler, and features saved successfully!")

# ==============================
# 9. MAIN PIPELINE
# ==============================
def main():
    print("Starting pipeline...\n")

    # Load data
    df = load_data()

    # Preprocess
    X, y = preprocess(df)

    # Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    print("\nModel Comparison:\n")
    print(results)

    # Hyperparameter tuning
    print("\nTuning Random Forest...")
    best_model = tune_random_forest(X_train, y_train)

    # Final evaluation
    pred = best_model.predict(X_test)

    print("\nFinal Tuned Model Performance:")
    print("MAE:", round(mean_absolute_error(y_test, pred), 4))
    print("MSE:", round(mean_squared_error(y_test, pred), 4))
    print("R2:", round(r2_score(y_test, pred), 4))

    # Save model
    save_model(best_model, scaler, X_train.columns.tolist())

    print("\nPipeline completed successfully!")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()