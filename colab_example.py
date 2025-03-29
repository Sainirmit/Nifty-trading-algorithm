"""
Example script for using Nifty-Trading-bot in Google Colab
This script shows how to handle the specific CSV format and generate features correctly
"""

# Step 1: Install required packages
# !pip install pandas numpy matplotlib scikit-learn pandas-ta yfinance

# Step 2: Clone the repository (if needed)
# !git clone https://github.com/your-username/Nifty-Trading-bot.git
# %cd Nifty-Trading-bot

# Step 3: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Step 4: Download data (if needed) or use your existing data
# !mkdir -p data/raw
# import yfinance as yf
# nifty_data = yf.download('^NSEI', start='2015-01-01', end='2022-12-31')
# nifty_data.to_csv('data/raw/nifty50_data.csv')

# Step 5: Generate features correctly using the helper module
from colab_helper import fix_nifty_csv, generate_features_for_colab

# Fix the CSV format and generate features
input_path = 'data/raw/nifty50_data.csv'
output_path = 'data/features/features_nifty50.csv'

# This function will handle the multi-row header issue and add features
df_features = generate_features_for_colab(input_path, output_path)

# Step 6: Visualize some of the features
if df_features is not None:
    plt.figure(figsize=(14, 7))
    
    # Plot the close price and moving averages
    plt.subplot(2, 1, 1)
    plt.plot(df_features.index, df_features['Close'], label='Close')
    plt.plot(df_features.index, df_features['MA_50'], label='MA 50')
    plt.plot(df_features.index, df_features['MA_200'], label='MA 200')
    plt.title('Nifty 50 with Moving Averages')
    plt.legend()
    
    # Plot RSI
    plt.subplot(2, 1, 2)
    plt.plot(df_features.index, df_features['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('RSI Indicator')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Display statistical summary of the features
    print("\nFeature Statistics:")
    print(df_features.describe())
    
    # Count of features by category
    price_features = [col for col in df_features.columns if 'MA_' in col or 'Price' in col]
    momentum_features = [col for col in df_features.columns if 'RSI' in col or 'MACD' in col]
    volatility_features = [col for col in df_features.columns if 'BBands' in col or 'ATR' in col]
    
    print(f"\nTotal features: {df_features.shape[1]}")
    print(f"Price-related features: {len(price_features)}")
    print(f"Momentum features: {len(momentum_features)}")
    print(f"Volatility features: {len(volatility_features)}")
    
# Step 7: Train a simple model (optional example)
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Define target variable (e.g., price direction)
df_features['Target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)

# Drop NaN values
df_ml = df_features.dropna()

# Select features
feature_cols = ['RSI', 'MACD', 'MA_50', 'MA_200', 'BBands_Width', 'Daily_Return']
X = df_ml[feature_cols]
y = df_ml['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_model.pkl')
""" 