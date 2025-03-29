"""
Functions for preprocessing and cleaning Nifty 50 data
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_price_data(df, fill_method='ffill'):
    """
    Clean price data by handling missing values and outliers
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        fill_method (str): Method to fill missing values ('ffill', 'bfill', 'interpolate')
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logger.info("Cleaning price data...")
    
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Ensure Date column is in datetime format
    if 'Date' in cleaned_df.columns:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in cleaned_df.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
    
    # Check for missing values
    missing_values = cleaned_df[required_columns].isnull().sum()
    
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected: {missing_values}")
        
        # Fill missing values based on method
        if fill_method == 'ffill':
            cleaned_df = cleaned_df.fillna(method='ffill')
            cleaned_df = cleaned_df.fillna(method='bfill')  # In case of missing values at the beginning
        elif fill_method == 'bfill':
            cleaned_df = cleaned_df.fillna(method='bfill')
            cleaned_df = cleaned_df.fillna(method='ffill')  # In case of missing values at the end
        elif fill_method == 'interpolate':
            cleaned_df = cleaned_df.interpolate(method='linear')
            cleaned_df = cleaned_df.fillna(method='ffill')  # For any remaining missing values
            cleaned_df = cleaned_df.fillna(method='bfill')  # For any remaining missing values
    
    # Check for and fix any data inconsistencies
    # High should be >= Open, Close, Low; Low should be <= Open, Close, High
    cleaned_df['High'] = cleaned_df[['High', 'Open', 'Close']].max(axis=1)
    cleaned_df['Low'] = cleaned_df[['Low', 'Open', 'Close']].min(axis=1)
    
    # Check for and handle extreme outliers using IQR method
    for col in ['Open', 'High', 'Low', 'Close']:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Identify outliers
        outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound))
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            logger.warning(f"Found {outlier_count} outliers in {col}")
            
            # Instead of removing, cap them to the bounds
            cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
            cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
    
    # Ensure Volume is non-negative
    if 'Volume' in cleaned_df.columns:
        cleaned_df.loc[cleaned_df['Volume'] < 0, 'Volume'] = 0
    
    # Sort by date
    if 'Date' in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values('Date')
    
    logger.info("Price data cleaning completed")
    
    return cleaned_df

def check_for_gaps(df, date_column='Date'):
    """
    Check for date gaps in the time series data
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        date_column (str): Name of the date column
        
    Returns:
        list: List of missing dates
    """
    # Convert to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Get unique dates
    dates = df[date_column].dt.date.unique()
    
    # Create a full date range
    full_range = pd.date_range(start=min(dates), end=max(dates))
    full_range = [d.date() for d in full_range]
    
    # Find weekdays in the full range
    weekdays = [d for d in full_range if d.weekday() < 5]
    
    # Find missing weekdays
    missing_dates = [d for d in weekdays if d not in dates]
    
    return missing_dates

def fill_date_gaps(df, date_column='Date'):
    """
    Fill gaps in time series data by adding rows for missing dates
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with filled gaps
    """
    logger.info("Checking for and filling date gaps...")
    
    # Check for gaps
    missing_dates = check_for_gaps(df, date_column)
    
    if not missing_dates:
        logger.info("No date gaps found")
        return df
    
    logger.info(f"Found {len(missing_dates)} missing dates")
    
    # Create a copy to avoid modifying the original DataFrame
    filled_df = df.copy()
    
    # Convert to datetime if not already
    filled_df[date_column] = pd.to_datetime(filled_df[date_column])
    
    # For each missing date, add a row
    for missing_date in missing_dates:
        # Find the nearest previous date in the data
        prev_dates = [d for d in filled_df[date_column].dt.date if d < missing_date]
        
        if not prev_dates:
            logger.warning(f"No previous dates for {missing_date}, skipping")
            continue
        
        nearest_prev_date = max(prev_dates)
        
        # Get the row for the nearest previous date
        prev_row = filled_df[filled_df[date_column].dt.date == nearest_prev_date].iloc[0].copy()
        
        # Update the date
        prev_row[date_column] = pd.Timestamp(missing_date)
        
        # Add the row to the DataFrame
        filled_df = pd.concat([filled_df, pd.DataFrame([prev_row])], ignore_index=True)
    
    # Sort by date
    filled_df = filled_df.sort_values(date_column)
    
    logger.info("Date gaps filled")
    
    return filled_df

def create_target_variable(df, target_type='direction', lookahead=1, threshold=0.0):
    """
    Create target variable for ML model
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        target_type (str): Type of target variable ('direction', 'return', 'binary')
        lookahead (int): Number of days to look ahead
        threshold (float): Threshold for binary classification
        
    Returns:
        pd.DataFrame: DataFrame with target variable added
    """
    logger.info(f"Creating target variable of type '{target_type}'...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_with_target = df.copy()
    
    # Create future price column
    df_with_target['Future_Close'] = df_with_target['Close'].shift(-lookahead)
    
    # Calculate price change
    df_with_target['Price_Change'] = df_with_target['Future_Close'] - df_with_target['Close']
    df_with_target['Pct_Change'] = df_with_target['Price_Change'] / df_with_target['Close'] * 100
    
    # Create target variable based on type
    if target_type == 'direction':
        # 1 if price goes up, 0 if price goes down
        df_with_target['target'] = (df_with_target['Price_Change'] > 0).astype(int)
    
    elif target_type == 'return':
        # Percentage change in price
        df_with_target['target'] = df_with_target['Pct_Change']
    
    elif target_type == 'binary':
        # 1 if price change exceeds threshold, 0 otherwise
        df_with_target['target'] = (df_with_target['Pct_Change'] > threshold).astype(int)
    
    # Drop rows with NaN target values (due to shifting)
    df_with_target = df_with_target.dropna(subset=['target'])
    
    logger.info(f"Target variable created with {len(df_with_target)} valid rows")
    
    return df_with_target

def preprocess_data(raw_data_path, processed_data_path=None, target_type='direction', lookahead=1):
    """
    Preprocess raw data and prepare it for feature engineering
    
    Args:
        raw_data_path (str): Path to raw data file
        processed_data_path (str, optional): Path to save processed data
        target_type (str): Type of target variable ('direction', 'return', 'binary')
        lookahead (int): Number of days to look ahead
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    try:
        logger.info(f"Preprocessing data from {raw_data_path}")
        
        # Read raw data
        df = pd.read_csv(raw_data_path)
        
        # Clean price data
        cleaned_df = clean_price_data(df)
        
        # Fill date gaps
        filled_df = fill_date_gaps(cleaned_df)
        
        # Create target variable
        processed_df = create_target_variable(filled_df, target_type, lookahead)
        
        # Save processed data if path is provided
        if processed_data_path:
            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
            processed_df.to_csv(processed_data_path, index=False)
            logger.info(f"Processed data saved to {processed_data_path}")
        
        return processed_df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def merge_price_and_options_data(price_data_path, options_data_path, output_path=None):
    """
    Merge price data with options metrics
    
    Args:
        price_data_path (str): Path to price data file
        options_data_path (str): Path to options metrics file
        output_path (str, optional): Path to save merged data
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        logger.info(f"Merging price data from {price_data_path} with options data from {options_data_path}")
        
        # Read data
        price_df = pd.read_csv(price_data_path)
        options_df = pd.read_csv(options_data_path)
        
        # Ensure Date columns are in datetime format
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        options_df['Date'] = pd.to_datetime(options_df['Date'])
        
        # Merge on Date
        merged_df = pd.merge(price_df, options_df, on='Date', how='left')
        
        # Fill missing options data
        # For missing options data, use forward fill as a simple approach
        options_columns = [col for col in options_df.columns if col != 'Date']
        merged_df[options_columns] = merged_df[options_columns].fillna(method='ffill')
        
        # Save merged data if path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Merged data saved to {output_path}")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
    
    # Preprocess data
    processed_df = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    print(f"Processed {len(processed_df)} rows of data")
    print(processed_df.head())