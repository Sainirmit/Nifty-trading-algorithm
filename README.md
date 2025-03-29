# Nifty 50 Trading Algorithm

A robust algorithmic trading system for the Nifty 50 index and its constituent stocks, featuring comprehensive data collection, feature engineering, machine learning model training, backtesting, and live trading capabilities.

## Project Structure

```
nifty50_trading_algo/
├── data/                      # Data storage
│   ├── raw/                   # Raw market data
│   ├── processed/             # Processed data
│   └── features/              # Feature-engineered data
├── models/                    # Trained ML models
├── notebooks/                 # Jupyter notebooks
│   ├── 1_data_collection.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_model_training.ipynb
│   └── 5_backtest.ipynb
├── results/                   # Results storage
│   ├── backtest/              # Backtesting results
│   └── trading/               # Live trading results
├── src/                       # Source code
│   ├── data_collection.py     # Data collection utilities
│   ├── feature_engineering.py # Feature engineering module
│   ├── model.py               # ML model training and prediction
│   ├── backtest.py            # Backtesting functionality
│   └── trading.py             # Live trading implementation
├── config.py                  # Configuration settings
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Features

- **Comprehensive Data Processing**: Collect and preprocess historical market data for Nifty 50 index and constituent stocks.
- **Advanced Feature Engineering**: Generate 200+ technical, statistical, and custom features for predicting market movement.
- **Flexible Machine Learning**: Support for multiple ML models including Random Forest, XGBoost, LightGBM, and LSTM neural networks.
- **Robust Backtesting**: Evaluate strategies with realistic simulations including transaction costs, stop losses, and take profit orders.
- **Live Trading Capabilities**: Execute strategies in real time with broker integration and risk management.
- **Performance Analytics**: Extensive metrics and visualizations to evaluate trading strategy performance.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nifty50_trading_algo.git
cd nifty50_trading_algo
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure your settings in `config.py`

## Usage

### Data Collection

```bash
python -m src.data_collection --start-date 2015-01-01 --end-date 2023-12-31
```

### Feature Engineering

```bash
python -m src.feature_engineering --input data/raw --output data/features
```

### Model Training

```bash
python -m src.model --model-type xgboost --input data/features/features_nifty50.csv
```

### Backtesting

```bash
python -m src.backtest --input data/features/features_nifty50.csv --model models/xgboost_20231231_123456.joblib
```

### Live Trading

```bash
python -m src.trading --symbols "NIFTY 50"
```

### Running Scheduled Trading

```bash
python -m src.trading --schedule --run-time 09:20
```

## Feature Engineering Details

The `feature_engineering.py` module provides an extensive set of features:

1. **Basic Price Features**: Calendar features, price ratios, body/wick metrics, Z-score normalizations, and 52-week metrics.
2. **Technical Indicators**: MACD, RSI, Bollinger Bands, Ichimoku Cloud, Keltner Channels, Donchian Channels, and more.
3. **Candlestick Patterns**: Recognition of over 25 patterns with manual verification and consensus signals.
4. **Options Metrics Features**: Put-Call Ratio analysis, Max Pain distance, IV Skew features, and options sentiment indicators.
5. **Custom Features**: Market regime detection, volatility bands, gap analysis, and support/resistance levels.

## Machine Learning Models

The `model.py` module supports multiple models:

- **Random Forest**: Robust tree-based ensemble method.
- **XGBoost**: Gradient boosting implementation with regularization.
- **LightGBM**: Efficient gradient boosting framework focusing on leaf-wise tree growth.
- **LSTM**: Deep learning approach for time series with long memory capabilities.

## Backtesting

The `backtest.py` module provides:

- Realistic simulation with transaction costs, slippage, and price impact
- Multiple strategy comparison
- Comprehensive performance metrics (Sharpe ratio, drawdown, win rate, etc.)
- Visualizations of strategy performance

## Live Trading

The `trading.py` module offers:

- Real-time market data integration
- Automated trading signals based on trained models
- Risk management with position sizing, stop losses, and take profit orders
- Portfolio tracking and performance monitoring
- Email notifications for trade alerts

## Configuration

Edit `config.py` to customize:

- Data sources and timeframes
- Feature engineering parameters
- Model hyperparameters
- Backtesting settings
- Trading parameters
- Risk management rules

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- TensorFlow
- TA-Lib
- pandas-ta
- matplotlib
- XGBoost
- LightGBM
- requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
