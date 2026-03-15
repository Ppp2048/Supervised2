#!/usr/bin/env python3
"""
Test script for stock prediction
"""

import sys
import os
sys.path.append('.')

try:
    from app import StockPredictor
    print("✅ StockPredictor imported successfully")

    # Initialize predictor
    predictor = StockPredictor()
    print("✅ StockPredictor initialized successfully")

    # Test prediction with AAPL
    print("🔮 Testing prediction with AAPL...")
    result = predictor.predict_price('AAPL')
    print("✅ Prediction successful:")
    print(f"   Ticker: {result['ticker']}")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()