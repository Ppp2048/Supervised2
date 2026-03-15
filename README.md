# Stock Price Prediction ML Pipeline

This project implements a complete machine learning pipeline for stock price prediction using historical data, with a web interface for easy predictions.

## Project Structure

```
Supervised2/
├── config/
│   └── config.yaml          # Configuration file with model parameters and paths
├── data/
│   ├── raw/                 # Raw downloaded stock data
│   └── processed/           # Processed data with features
├── artifacts/               # Trained models and metrics
├── logs/                    # Application logs
├── templates/               # HTML templates for web interface
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Downloads stock data using yfinance
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Orchestrates the training pipeline
│   │   └── predict_pipeline.py    # For making predictions
│   ├── logger.py            # Logging configuration
│   ├── exceptions.py        # Custom exception handling
│   └── utils.py             # Utility functions
├── app.py                   # Flask web application
├── run.py                   # Script to run the Flask app
├── requirements.txt         # Python dependencies
└── setup.py                 # Package setup
```

## Features

### Data Ingestion
- Downloads historical stock data using yfinance
- Configurable stock ticker and date range
- Saves raw data to `data/raw/stock_data.csv`

### Data Preprocessing
- Handles missing values with forward fill
- Creates lagged features (1, 5, 10 days)
- Adds technical indicators:
  - Moving Average (MA20, MA50)
  - Relative Strength Index (RSI)
- Saves processed data to `data/processed/processed_data.csv`

### Model Training
Trains three different models using time-series split cross-validation:
- **Linear Regression**: Simple baseline model
- **XGBoost**: Tree-based ensemble model
- **LSTM**: Deep learning sequence model

### Model Selection
- Compares models based on Mean Squared Error (MSE)
- Saves the best performing model to `artifacts/model.pkl`
- Saves evaluation metrics to `artifacts/metrics.json`

### Web Interface
- Flask-based web application
- Interactive prediction interface
- Real-time stock data fetching
- Responsive design

## Configuration

All parameters are configurable in `config/config.yaml`:

```yaml
stock:
  ticker: "AAPL"
  start_date: "2020-01-01"
  end_date: "2024-01-01"

models:
  linear_regression:
    fit_intercept: true
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  lstm:
    units: 50
    epochs: 50
    batch_size: 32
```

## Usage

### Training Pipeline
```bash
python src/pipeline/train_pipeline.py
```

This will:
1. Download stock data
2. Process and transform the data
3. Train all models
4. Select and save the best model

### Running the Web App
```bash
python run.py
```

Or directly:
```bash
python app.py
```

The web app will be available at: http://localhost:5000

### Web Interface Features
- **Home Page**: Welcome page with project overview
- **Prediction Page**: Enter stock ticker to get price predictions
- **Real-time Data**: Fetches current market data for predictions
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints
- `GET /`: Home page
- `GET /home`: Prediction interface
- `POST /predict`: Make price prediction (accepts `ticker` parameter)
- `GET /health`: Health check endpoint

## Requirements
- pandas
- numpy
- yfinance
- xgboost
- scikit-learn
- tensorflow
- ta (technical analysis)
- flask

Install dependencies:
```bash
pip install -r requirements.txt
```

## Logging
All operations are logged to timestamped files in the `logs/` directory for debugging and monitoring.

## Exception Handling
Custom exception handling with detailed error messages and logging for robust error tracking.
