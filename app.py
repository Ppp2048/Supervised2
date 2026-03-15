from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import yaml

app = Flask(__name__)

class StockPredictor:
    def __init__(self):
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Load the trained model
        with open(self.config['artifacts']['model_path'], 'rb') as f:
            self.model = pickle.load(f)

        print("✅ Model loaded successfully")
        print(f"📊 Model type: {type(self.model)}")

        # Test model with dummy data
        try:
            dummy_data = np.array([[100, 105, 110, 102, 108, 65]])  # Sample features
            test_pred = self.model.predict(dummy_data)
            print(f"🧪 Model test prediction: {test_pred}")
        except Exception as e:
            print(f"⚠️ Model test failed: {str(e)}")

    def get_recent_data(self, ticker, days=100):  # Increased from 60 to 100 days
        """Get recent stock data for prediction"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            print(f"📡 Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")
            data = yf.download(ticker, start=start_date, end=end_date)

            print(f"📊 Downloaded data shape: {data.shape}")
            print(f"📊 Data columns: {list(data.columns)}")
            print(f"📊 Data index: {data.index}")

            if data.empty:
                raise Exception(f"No data found for ticker {ticker}")

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                print("🔧 Flattening MultiIndex columns")
                data.columns = data.columns.droplevel(1) if data.columns.nlevels > 1 else data.columns

            print(f"📊 After flattening, columns: {list(data.columns)}")

            # Ensure we have enough data
            if len(data) < 30:
                raise Exception(f"Insufficient data for {ticker}. Only {len(data)} days available, need at least 30.")

            return data
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {str(e)}")
            raise

    def prepare_features(self, df):
        """Prepare features for prediction"""
        try:
            print(f"📊 Preparing features for dataframe with shape: {df.shape}")
            print(f"📊 Data columns: {list(df.columns)}")

            # Calculate lagged features
            df['lag_1'] = df['Close'].shift(1)
            df['lag_5'] = df['Close'].shift(5)
            df['lag_10'] = df['Close'].shift(10)

            # Calculate simple moving averages (no ta library needed)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            # Simple RSI calculation (simplified version)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            print(f"📊 After feature calculation, shape: {df.shape}")

            # Get the most recent row, filling NaN values with 0
            features = ['lag_1', 'lag_5', 'lag_10', 'MA20', 'MA50', 'RSI']
            latest_data = df[features].iloc[-1:]  # Get the last row, even if it has NaN

            print(f"📊 Latest data shape: {latest_data.shape}")
            print(f"📊 Latest data:\n{latest_data}")

            if latest_data.empty:
                raise Exception("No data available after feature calculation")

            # Fill any remaining NaN values with 0
            latest_data = latest_data.fillna(0)

            print(f"📊 Data after filling NaN:\n{latest_data}")

            # Check for infinite values
            if np.isinf(latest_data.values).any():
                raise Exception("Feature data contains infinite values")

            print(f"✅ Feature validation passed")
            return latest_data
        except Exception as e:
            print(f"❌ Error preparing features: {str(e)}")
            print(f"❌ DataFrame info: {df.info()}")
            raise

    def predict_price(self, ticker):
        """Predict next day closing price for given ticker"""
        try:
            print(f"🔮 Starting prediction for {ticker}")

            # Get recent data
            data = self.get_recent_data(ticker)
            print(f"📈 Downloaded data shape: {data.shape}")

            # Prepare features
            features = self.prepare_features(data)
            print(f"🔧 Features prepared, shape: {features.shape}")

            # Make prediction
            print(f"🔢 Feature values: {features.values}")
            print(f"🔢 Feature shape for prediction: {features.values.shape}")

            try:
                prediction = self.model.predict(features.values)[0]
                print(f"🎯 Raw prediction: {prediction}")
            except Exception as pred_error:
                print(f"❌ Model prediction error: {str(pred_error)}")
                print(f"❌ Model type: {type(self.model)}")
                print(f"❌ Feature dtypes: {features.dtypes}")
                raise Exception(f"Model prediction failed: {str(pred_error)}")

            # Get current price
            current_price = data['Close'].iloc[-1]
            print(f"💰 Current price: {current_price}")

            result = {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'predicted_price': round(prediction, 2),
                'change': round(prediction - current_price, 2),
                'change_percent': round(((prediction - current_price) / current_price) * 100, 2)
            }

            print(f"✅ Prediction result: {result}")
            return result

        except Exception as e:
            print(f"❌ Error predicting price for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# Initialize predictor
try:
    predictor = StockPredictor()
except Exception as e:
    print(f"❌ Failed to initialize predictor: {str(e)}")
    predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if predictor is None:
            return jsonify({'error': 'Model not available. Please check the application logs.'}), 500

        ticker = request.form.get('ticker', '').upper()

        if not ticker:
            return jsonify({'error': 'Please provide a stock ticker'}), 400

        result = predictor.predict_price(ticker)

        return jsonify(result)

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)