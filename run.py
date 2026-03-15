#!/usr/bin/env python3
"""
Stock Price Prediction Flask App Runner
"""

from app import app

if __name__ == '__main__':
    print("🚀 Starting Stock Price Predictor Flask App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("❌ Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)