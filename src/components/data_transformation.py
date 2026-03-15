import os
import sys
import pandas as pd
import numpy as np
import ta
from src.logger import logging
from src.exceptions import CustomException
import yaml

class DataTransformation:
    def __init__(self):
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def load_data(self):
        try:
            data_path = self.config['data']['raw_path']
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logging.info(f"Data loaded from {data_path}")
            return df
        except Exception as e:
            logging.error("Error loading data")
            raise CustomException(e, sys)

    def handle_missing_values(self, df):
        try:
            # Forward fill missing values using the new pandas syntax
            df = df.ffill()
            # Drop any remaining NaN values
            df = df.dropna()
            logging.info("Missing values handled")
            return df
        except Exception as e:
            logging.error("Error handling missing values")
            raise CustomException(e, sys)

    def create_lagged_features(self, df):
        try:
            # Create lagged features
            df['lag_1'] = df['Close'].shift(1)
            df['lag_5'] = df['Close'].shift(5)
            df['lag_10'] = df['Close'].shift(10)
            logging.info("Lagged features created")
            return df
        except Exception as e:
            logging.error("Error creating lagged features")
            raise CustomException(e, sys)

    def add_technical_indicators(self, df):
        try:
            # Moving Averages
            df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)

            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            logging.info("Technical indicators added")
            return df
        except Exception as e:
            logging.error("Error adding technical indicators")
            raise CustomException(e, sys)

    def transform_data(self):
        try:
            logging.info("Starting data transformation")

            # Load data
            df = self.load_data()

            # Handle missing values
            df = self.handle_missing_values(df)

            # Create lagged features
            df = self.create_lagged_features(df)

            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Drop rows with NaN values created by lagging and indicators
            df = df.dropna()

            # Ensure data/processed directory exists
            os.makedirs(os.path.dirname(self.config['data']['processed_path']), exist_ok=True)

            # Save processed data
            df.to_csv(self.config['data']['processed_path'])
            logging.info(f"Processed data saved to {self.config['data']['processed_path']}")

            return self.config['data']['processed_path']

        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    transformation = DataTransformation()
    transformation.transform_data()
