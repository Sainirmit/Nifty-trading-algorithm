"""
Configuration settings for Nifty 50 trading algorithm
"""

import os
from pathlib import Path

# Base directory structure
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DATA_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BACKTEST_RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'backtest')
TRADING_RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'trading')

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR,
    MODELS_DIR, BACKTEST_RESULTS_DIR, TRADING_RESULTS_DIR
]:
    os.makedirs(directory, exist_ok=True)

# Data collection settings
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = None  # None means today
NIFTY50_SYMBOL = 'NIFTY 50'
DATA_SOURCES = {
    'yahoo': {
        'index_prefix': '^',
        'stock_suffix': '.NS'
    },
    'nseindia': {
        'url': 'https://www.nseindia.com/api/historical-data',
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    }
}

# Feature engineering settings
FEATURE_ENGINEERING_PARAMS = {
    'volatility_window': 20,
    'momentum_window': 14,
    'regime_threshold': 200,  # Days for moving average to detect market regime
    'minimal_features': False  # Set to True to generate only essential features
}

# Model settings
TARGET_VARIABLE = 'Daily_Return'
PREDICTION_HORIZON = 1  # Days ahead to predict
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
VALIDATION_SPLIT = 0.2  # 20% of training data for validation
SEQUENCE_LENGTH = 20  # For LSTM models
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lstm': {
        'units': [64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    }
}

# Backtesting settings
INITIAL_CAPITAL = 100000.0
POSITION_SIZE = 0.2  # 20% of capital per position
STOP_LOSS = 0.05  # 5% stop loss
TAKE_PROFIT = 0.10  # 10% take profit
TRADING_COST = 0.001  # 0.1% transaction cost

# Trading settings
TRADING_ENABLED = False  # Set to True to enable live trading
TRADING_SYMBOLS = ['NIFTY 50']  # Symbols to trade
TRADING_TIME = '09:20'  # Time to run the strategy (HH:MM)

# Broker API settings (placeholder - replace with actual API credentials)
BROKER_API_URL = 'https://api.broker.com'
API_KEY = ''  # Enter your broker API key
API_SECRET = ''  # Enter your broker API secret

# Email notification settings
EMAIL_NOTIFICATIONS = False  # Set to True to enable email notifications
EMAIL_SENDER = ''  # Enter your email
EMAIL_PASSWORD = ''  # Enter your email password
EMAIL_RECIPIENTS = ['']  # List of email recipients

# Risk management settings
MAX_TRADES_PER_DAY = 3
MAX_POSITION_SIZE_PCT = 25.0  # Maximum position size as percentage of portfolio
MAX_PORTFOLIO_RISK_PCT = 5.0  # Maximum portfolio risk as percentage

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Debug settings
DEBUG_MODE = False  # Set to True to enable additional debug logs