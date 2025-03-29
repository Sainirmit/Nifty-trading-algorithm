"""
This is a standalone cell for Step 5 that fixes the CSV format problem.
Paste this directly into a Colab cell to process your Nifty 50 data correctly.
"""

# Step 5: Generate features with proper CSV handling
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def fix_nifty_csv(csv_path):
    """Fix the Nifty CSV file that has a multi-row header."""
    # First, examine the CSV structure
    print(f"Reading CSV file: {csv_path}")
    preview_df = pd.read_csv(csv_path, nrows=10)
    print("CSV Preview:")
    print(preview_df.head(10))
    
    # Skip the first two rows that contain header info
    df = pd.read_csv(csv_path, skiprows=2)
    
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

def add_basic_features(df):
    """Add basic price-derived features."""
    print("Adding basic price features...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Add calendar-based features
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
    
    # RSI (simplified calculation)
    delta = df_features['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    fast_ema = df_features['Close'].ewm(span=12, adjust=False).mean()
    slow_ema = df_features['Close'].ewm(span=26, adjust=False).mean()
    df_features['MACD'] = fast_ema - slow_ema
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    middle_band = df_features['Close'].rolling(window=20).mean()
    std = df_features['Close'].rolling(window=20).std()
    df_features['BBands_Upper'] = middle_band + (2 * std)
    df_features['BBands_Middle'] = middle_band
    df_features['BBands_Lower'] = middle_band - (2 * std)
    
    print("Features added successfully!")
    return df_features

# Main execution
try:
    # Define paths
    input_path = 'data/raw/nifty50_data.csv'
    output_path = 'data/features/features_nifty50.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fix the CSV file format and load the data
    df = fix_nifty_csv(input_path)
    
    # Add features
    df_features = add_basic_features(df)
    
    # Save the feature data
    df_features.to_csv(output_path)
    print(f"Generated {df_features.shape[1]} features for {df_features.shape[0]} days")
    print(f"Features saved to {output_path}")
    
    # Plot some features
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_features.index, df_features['Close'], label='Close')
    plt.plot(df_features.index, df_features['MA_50'], label='MA 50')
    plt.plot(df_features.index, df_features['MA_200'], label='MA 200')
    plt.title('Nifty 50 with Moving Averages')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df_features.index, df_features['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('RSI Indicator')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error during feature generation: {e}")
    import traceback
    traceback.print_exc() 