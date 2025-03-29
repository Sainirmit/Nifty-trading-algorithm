# Data Directory

This directory is used to store historical data for the Nifty 50 index and its constituent stocks.

## Data Structure

- `raw/` - Raw data downloaded from sources
- `processed/` - Cleaned and preprocessed data
- `features/` - Data with engineered features

## Data Sources

1. **NSEPy**: Used for downloading historical data from the National Stock Exchange of India

   - Daily OHLCV data for Nifty 50 index
   - Daily OHLCV data for constituent stocks

2. **Yahoo Finance (yfinance)**: Alternative source for index and stock data
   - Allows downloading data for longer timeframes
   - Provides additional data like dividends and stock splits

## Data Collection Process

1. Run the `1_data_collection.ipynb` notebook or use the functions in `src/data_collection.py`
2. Configure the date range and symbols in `config.py`
3. Data will be saved in CSV format in the `raw/` subdirectory

## Data Update Schedule

It's recommended to update the data daily after market hours (after 3:30 PM IST) to include the latest trading day.

## Data Backup

Consider backing up the data directory regularly to prevent data loss.

## Note

The data directory is not included in version control (added to .gitignore) due to file size constraints. Each user should generate their own dataset by running the data collection scripts.
