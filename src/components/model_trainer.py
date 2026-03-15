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
from src.logger import logging
from src.exceptions import CustomException
import yaml

class ModelTrainer:
    def __init__(self):
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def load_data(self):
        try:
            data_path = self.config['data']['processed_path']
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logging.info(f"Processed data loaded from {data_path}")
            return df
        except Exception as e:
            logging.error("Error loading processed data")
            raise CustomException(e, sys)

    def prepare_features_target(self, df):
        try:
            # Features: lagged prices and technical indicators
            features = ['lag_1', 'lag_5', 'lag_10', 'MA20', 'MA50', 'RSI']
            target = 'Close'

            X = df[features]
            y = df[target]

            logging.info("Features and target prepared")
            return X, y
        except Exception as e:
            logging.error("Error preparing features and target")
            raise CustomException(e, sys)

    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        try:
            model = LinearRegression(fit_intercept=self.config['models']['linear_regression']['fit_intercept'])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Linear Regression - MSE: {mse}, R2: {r2}")
            return model, mse, r2
        except Exception as e:
            logging.error("Error training Linear Regression")
            raise CustomException(e, sys)

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        try:
            model = XGBRegressor(
                n_estimators=self.config['models']['xgboost']['n_estimators'],
                max_depth=self.config['models']['xgboost']['max_depth'],
                learning_rate=self.config['models']['xgboost']['learning_rate'],
                random_state=self.config['training']['random_state']
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"XGBoost - MSE: {mse}, R2: {r2}")
            return model, mse, r2
        except Exception as e:
            logging.error("Error training XGBoost")
            raise CustomException(e, sys)

    def train_models(self):
        try:
            logging.info("Starting model training")

            # Load data
            df = self.load_data()
            X, y = self.prepare_features_target(df)

            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            splits = list(tscv.split(X))

            # Get last split for final evaluation
            train_idx, test_idx = splits[-1]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train models
            lr_model, lr_mse, lr_r2 = self.train_linear_regression(X_train, y_train, X_test, y_test)
            xgb_model, xgb_mse, xgb_r2 = self.train_xgboost(X_train, y_train, X_test, y_test)

            # Compare models and select best (lowest MSE)
            models = {
                'LinearRegression': (lr_model, lr_mse, lr_r2),
                'XGBoost': (xgb_model, xgb_mse, xgb_r2)
            }

            best_model_name = min(models, key=lambda x: models[x][1])
            best_model, best_mse, best_r2 = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with MSE: {best_mse}, R2: {best_r2}")

            # Save best model
            os.makedirs(os.path.dirname(self.config['artifacts']['model_path']), exist_ok=True)
            with open(self.config['artifacts']['model_path'], 'wb') as f:
                pickle.dump(best_model, f)

            # Save metrics
            metrics = {
                'best_model': best_model_name,
                'LinearRegression': {'mse': lr_mse, 'r2': lr_r2},
                'XGBoost': {'mse': xgb_mse, 'r2': xgb_r2}
            }

            with open(self.config['artifacts']['metrics_path'], 'w') as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Best model saved to {self.config['artifacts']['model_path']}")
            logging.info(f"Metrics saved to {self.config['artifacts']['metrics_path']}")

            return self.config['artifacts']['model_path'], metrics

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_models()
