"""
Helper functions for working with Nifty 50 data in Google Colab
This script handles the specific format of the CSV file with multi-row headers
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_nifty_csv(csv_path):
    """
    Fix the Nifty CSV file that has a multi-row header.
    Returns the properly formatted DataFrame with correct column names.
    
    Args:
        csv_path (str): Path to the raw CSV file
        
    Returns:
        pd.DataFrame: DataFrame with proper formatting
    """
    # First, examine the CSV structure
    print(f"Reading CSV file: {csv_path}")
    preview_df = pd.read_csv(csv_path, nrows=10)
    print("CSV Preview:")
    print(preview_df.head(10))
    
    # Skip the first two rows that contain header info
    df = pd.read_csv(csv_path, skiprows=2)
    
    # Rename columns based on what we know about the file structure
    # From the error output, the columns are:
    # 'Price' -> Date, 'Unnamed: 1' -> Close, 'Unnamed: 2' -> High, 
    # 'Unnamed: 3' -> Low, 'Unnamed: 4' -> Open, 'Unnamed: 5' -> Volume
    
    # Get the actual column names from the original file
    columns = preview_df.columns.tolist()
    
    # Create the mapping based on the column positions
    if len(columns) >= 6:
        columns_mapping = {
            columns[0]: 'Date',    # First column is Date
            columns[1]: 'Close',   # Second column is Close
            columns[2]: 'High',    # Third column is High
            columns[3]: 'Low',     # Fourth column is Low
            columns[4]: 'Open',    # Fifth column is Open
            columns[5]: 'Volume'   # Sixth column is Volume
        }
        df = df.rename(columns=columns_mapping)
    
    # Convert Date to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Convert numeric columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Print the fixed DataFrame info
    print("\nFixed DataFrame:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types: {df.dtypes}")
    print(df.head())
    
    return df


def generate_features_for_colab(input_csv_path, output_path=None):
    """
    Special function for Google Colab to fix CSV format issues and generate features.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the output features CSV
    
    Returns:
        pd.DataFrame: DataFrame with generated features
    """
    import sys
    import importlib.util
    
    # Fix the CSV file and load as a DataFrame
    df = fix_nifty_csv(input_csv_path)
    
    try:
        # Try to import the feature engineering module
        print("Importing feature engineering module...")
        sys.path.append(os.getcwd())
        
        # First method: direct import attempt
        try:
            from src.feature_engineering import (add_basic_price_features, 
                                               add_technical_indicators,
                                               add_candlestick_patterns,
                                               add_custom_features)
            use_simple_indicators = False
            print("Successfully imported full feature engineering module")
            
        except ImportError as e:
            print(f"Could not import full module: {e}")
            print("Falling back to simplified technical indicators")
            use_simple_indicators = True
        
        # Add features
        print("Adding features to DataFrame...")
        if not use_simple_indicators:
            # Use the full feature engineering module
            df_with_features = add_basic_price_features(df)
            df_with_features = add_technical_indicators(df_with_features)
            df_with_features = add_candlestick_patterns(df_with_features)
            df_with_features = add_custom_features(df_with_features)
        else:
            # Use simplified indicators (defined in this module)
            df_with_features = simplified_basic_features(df)
            df_with_features = simplified_technical_indicators(df_with_features)
        
        # Save to output path if provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_with_features.to_csv(output_path)
            print(f"Features saved to {output_path}")
        
        print(f"Generated {df_with_features.shape[1]} features for {df_with_features.shape[0]} days")
        return df_with_features
        
    except Exception as e:
        print(f"Error generating features: {e}")
        import traceback
        traceback.print_exc()
        return None


def simplified_basic_features(df):
    """
    Simplified version of basic price features for Colab (if imports fail)
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    print("Adding simplified basic features...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Add calendar-based features if the index is a datetime
    if isinstance(df_features.index, pd.DatetimeIndex):
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_month'] = df_features.index.day
        df_features['month'] = df_features.index.month
        df_features['year'] = df_features.index.year
    
    # Calculate returns
    df_features['Daily_Return'] = df_features['Close'].pct_change() * 100
    
    # Price change features
    df_features['Price_Change'] = df_features['Close'] - df_features['Open']
    df_features['Pct_Change'] = (df_features['Price_Change'] / df_features['Open']) * 100
    
    # Range features
    df_features['Daily_Range'] = df_features['High'] - df_features['Low']
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()
    
    print("Simplified basic features added")
    return df_features


def simplified_technical_indicators(df):
    """
    Simplified version of technical indicators for Colab (if imports fail)
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    print("Adding simplified technical indicators...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # RSI (simplified calculation)
    def calculate_rsi(prices, period=14):
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # Add RSI
    df_features['RSI'] = calculate_rsi(df_features['Close'], period=14)
    
    # MACD (simplified calculation)
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    # Add MACD
    macd, signal, hist = calculate_macd(df_features['Close'])
    df_features['MACD'] = macd
    df_features['MACD_Signal'] = signal
    df_features['MACD_Hist'] = hist
    
    # Bollinger Bands
    def calculate_bollinger_bands(prices, window=20, std_dev=2):
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    # Add Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df_features['Close'])
    df_features['BBands_Upper'] = upper
    df_features['BBands_Middle'] = middle
    df_features['BBands_Lower'] = lower
    
    print("Simplified technical indicators added")
    return df_features 