#!/usr/bin/env python3
"""
Simple test for stock prediction debugging
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import yaml

def test_prediction():
    try:
        # Load config
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Load model
        with open(config['artifacts']['model_path'], 'rb') as f:
            model = pickle.load(f)

        print("✅ Model loaded successfully")
        print(f"📊 Model type: {type(model)}")

        # Test data fetching - fetch more data for proper feature calculation
        ticker = 'AAPL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)  # Increased from 60 to 100 days

        print(f"📡 Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        print(f"📊 Data shape: {data.shape}")
        print(f"📊 Data columns: {list(data.columns)}")

        if data.empty:
            print("❌ No data found")
            return

        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1) if data.columns.nlevels > 1 else data.columns

        print(f"📊 After flattening, columns: {list(data.columns)}")

        # Calculate features
        df = data.copy()
        df['lag_1'] = df['Close'].shift(1)
        df['lag_5'] = df['Close'].shift(5)
        df['lag_10'] = df['Close'].shift(10)

        # Simple moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        features = ['lag_1', 'lag_5', 'lag_10', 'MA20', 'MA50', 'RSI']
        latest_data = df[features].iloc[-1:]  # Get the last row, even if it has NaN

        print(f"📊 Latest data shape: {latest_data.shape}")
        print(f"📊 Latest data:\n{latest_data}")

        if latest_data.empty:
            print("❌ No data available")
            return

        # Fill any remaining NaN values with 0 or appropriate defaults
        latest_data = latest_data.fillna(0)

        print(f"📊 Data after filling NaN:\n{latest_data}")

        # Test prediction
        prediction = model.predict(latest_data.values)[0]
        current_price = df['Close'].iloc[-1]

        print("✅ Prediction successful!")
        print(f"💰 Current price: ${current_price:.2f}")
        print(f"🎯 Predicted price: ${prediction:.2f}")
        print(f"📈 Change: ${prediction - current_price:.2f}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_prediction()