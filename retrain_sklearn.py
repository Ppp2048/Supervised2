#!/usr/bin/env python3
"""
Simple retraining script using only sklearn models
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import yaml

def retrain_sklearn_only():
    try:
        # Load config
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Load processed data
        data_path = config['data']['processed_path']
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✅ Data loaded from {data_path}")

        # Prepare features and target
        features = ['lag_1', 'lag_5', 'lag_10', 'MA20', 'MA50', 'RSI']
        target = 'Close'

        X = df[features]
        y = df[target]
        print("✅ Features and target prepared")

        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))

        # Get last split for final evaluation
        train_idx, test_idx = splits[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train Linear Regression
        lr_model = LinearRegression(fit_intercept=config['models']['linear_regression']['fit_intercept'])
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        print(f"✅ Linear Regression - MSE: {lr_mse}, R2: {lr_r2}")

        # Train XGBoost
        xgb_model = XGBRegressor(
            n_estimators=config['models']['xgboost']['n_estimators'],
            max_depth=config['models']['xgboost']['max_depth'],
            learning_rate=config['models']['xgboost']['learning_rate'],
            random_state=config['training']['random_state']
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        print(f"✅ XGBoost - MSE: {xgb_mse}, R2: {xgb_r2}")

        # Compare models and select best (lowest MSE)
        models = {
            'LinearRegression': (lr_model, lr_mse, lr_r2),
            'XGBoost': (xgb_model, xgb_mse, xgb_r2)
        }

        best_model_name = min(models, key=lambda x: models[x][1])
        best_model, best_mse, best_r2 = models[best_model_name]

        print(f"🏆 Best model: {best_model_name} with MSE: {best_mse}, R2: {best_r2}")

        # Save best model
        os.makedirs(os.path.dirname(config['artifacts']['model_path']), exist_ok=True)
        with open(config['artifacts']['model_path'], 'wb') as f:
            pickle.dump(best_model, f)

        # Save metrics
        metrics = {
            'best_model': best_model_name,
            'LinearRegression': {'mse': lr_mse, 'r2': lr_r2},
            'XGBoost': {'mse': xgb_mse, 'r2': xgb_r2}
        }

        with open(config['artifacts']['metrics_path'], 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"💾 Best model saved to {config['artifacts']['model_path']}")
        print(f"📊 Metrics saved to {config['artifacts']['metrics_path']}")

        return best_model_name, best_mse, best_r2

    except Exception as e:
        print(f"❌ Error in retraining: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    retrain_sklearn_only()