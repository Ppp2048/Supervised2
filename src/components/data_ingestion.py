import os
import sys
import pandas as pd
import yfinance as yf
from src.logger import logging
from src.exceptions import CustomException
import yaml

class DataIngestion:
    def __init__(self):
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def download_stock_data(self):
        try:
            logging.info("Starting data ingestion")
            ticker = self.config['stock']['ticker']
            start_date = self.config['stock']['start_date']
            end_date = self.config['stock']['end_date']

            logging.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Ensure data/raw directory exists
            os.makedirs(os.path.dirname(self.config['data']['raw_path']), exist_ok=True)

            # Save to CSV
            data.to_csv(self.config['data']['raw_path'])
            logging.info(f"Data saved to {self.config['data']['raw_path']}")

            return self.config['data']['raw_path']

        except Exception as e:
            logging.error("Error in data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.download_stock_data()
