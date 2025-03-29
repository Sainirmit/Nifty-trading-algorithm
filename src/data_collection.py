"""
Data Collection Module for Nifty 50 Trading Algorithm.

This module provides functions to download historical data for the Nifty 50 index
and its constituent stocks from various sources.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR, 
    START_DATE, 
    END_DATE, 
    INDEX_SYMBOL, 
    NIFTY50_CONSTITUENTS,
    DATA_SOURCE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_nsepy_data(symbol, start_date, end_date, is_index=False):
    """
    Download data from NSE using nsepy library.
    
    Args:
        symbol (str): Stock or index symbol
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
        is_index (bool): Whether the symbol is an index
        
    Returns:
        pandas.DataFrame: Historical OHLCV data
    """
    try:
        import nsepy
        logger.info(f"Downloading {symbol} data from NSEPy")
        
        if is_index:
            data = nsepy.get_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                index=True
            )
        else:
            data = nsepy.get_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        return data
    
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()

def download_yfinance_data(symbol, start_date, end_date, is_index=False):
    """
    Download data from Yahoo Finance using yfinance library.
    
    Args:
        symbol (str): Stock or index symbol
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
        is_index (bool): Whether the symbol is an index (not used for yfinance)
        
    Returns:
        pandas.DataFrame: Historical OHLCV data
    """
    try:
        import yfinance as yf
        logger.info(f"Downloading {symbol} data from Yahoo Finance")
        
        # Append .NS for NSE symbols
        ticker = f"{symbol}.NS"
        
        # For Nifty 50 index use ^NSEI
        if symbol == 'NIFTY 50':
            ticker = "^NSEI"
            
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        return data
    
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()

def download_data(symbol, start_date=START_DATE, end_date=END_DATE, source=DATA_SOURCE, is_index=False):
    """
    Download data from the specified source.
    
    Args:
        symbol (str): Stock or index symbol
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
        source (str): Data source ('nsepy' or 'yfinance')
        is_index (bool): Whether the symbol is an index
        
    Returns:
        pandas.DataFrame: Historical OHLCV data
    """
    if source.lower() == 'nsepy':
        return download_nsepy_data(symbol, start_date, end_date, is_index)
    elif source.lower() == 'yfinance':
        return download_yfinance_data(symbol, start_date, end_date, is_index)
    else:
        logger.error(f"Unknown data source: {source}")
        return pd.DataFrame()

def save_data(data, symbol, directory=RAW_DATA_DIR, is_index=False):
    """
    Save data to CSV file.
    
    Args:
        data (pandas.DataFrame): Data to save
        symbol (str): Stock or index symbol
        directory (str): Directory to save the data
        is_index (bool): Whether the symbol is an index
        
    Returns:
        str: Path to the saved file
    """
    if data.empty:
        logger.warning(f"No data to save for {symbol}")
        return ""
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create subdirectory for indices or stocks
    subdir = "indices" if is_index else "stocks"
    save_dir = os.path.join(directory, subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with date range
    start_date_str = data.index.min().strftime('%Y%m%d')
    end_date_str = data.index.max().strftime('%Y%m%d')
    filename = f"{symbol}_{start_date_str}_{end_date_str}.csv"
    filepath = os.path.join(save_dir, filename)
    
    # Save data
    data.to_csv(filepath)
    logger.info(f"Data saved to {filepath}")
    
    return filepath

def download_nifty_data(start_date=START_DATE, end_date=END_DATE, source=DATA_SOURCE):
    """
    Download Nifty 50 index data.
    
    Args:
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
        source (str): Data source ('nsepy' or 'yfinance')
        
    Returns:
        str: Path to the saved file
    """
    logger.info(f"Downloading Nifty 50 index data from {start_date} to {end_date}")
    
    # Download data
    data = download_data(INDEX_SYMBOL, start_date, end_date, source, is_index=True)
    
    # Save data
    return save_data(data, INDEX_SYMBOL, is_index=True)

def download_nifty_constituents(start_date=START_DATE, end_date=END_DATE, source=DATA_SOURCE):
    """
    Download data for all Nifty 50 constituents.
    
    Args:
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
        source (str): Data source ('nsepy' or 'yfinance')
        
    Returns:
        list: Paths to the saved files
    """
    logger.info(f"Downloading data for {len(NIFTY50_CONSTITUENTS)} Nifty 50 constituents")
    
    filepaths = []
    
    # Download data for each constituent
    for symbol in tqdm(NIFTY50_CONSTITUENTS):
        data = download_data(symbol, start_date, end_date, source, is_index=False)
        filepath = save_data(data, symbol, is_index=False)
        if filepath:
            filepaths.append(filepath)
    
    logger.info(f"Downloaded data for {len(filepaths)} constituents")
    
    return filepaths

def main():
    """Main function to download all required data."""
    logger.info("Starting data collection")
    
    # Download Nifty 50 index data
    nifty_file = download_nifty_data()
    
    # Download Nifty 50 constituents data
    constituent_files = download_nifty_constituents()
    
    logger.info("Data collection completed")
    
    return nifty_file, constituent_files

if __name__ == "__main__":
    main()
