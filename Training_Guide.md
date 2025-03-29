# Step-by-Step Guide to Training the Nifty 50 Trading Model

This guide will walk you through the process of training a machine learning model for the Nifty 50 trading algorithm.

## Step 1: Collect Historical Data

First, you need historical price data for the Nifty 50 index:

```bash
# Create the data directories if they don't exist
mkdir -p data/raw data/processed data/features

# Run the data collection script
python3 -m src.data_collection --start-date 2015-01-01 --symbols "NIFTY 50" --output data/raw
```

## Step 2: Generate Features

Next, transform the raw data into features for machine learning:

```bash
# Run the feature engineering script
python3 -m src.feature_engineering --input data/raw --output data/features
```

## Step 3: Train a Machine Learning Model

Now, train a model using the generated features:

```bash
# Train a Random Forest model (classification)
python3 -m src.model --model-type random_forest --input data/features/features_nifty50.csv --output models

# Or train an XGBoost model
python3 -m src.model --model-type xgboost --input data/features/features_nifty50.csv --output models

# Or train a LightGBM model
python3 -m src.model --model-type lgbm --input data/features/features_nifty50.csv --output models

# Or train an LSTM model (requires more data and time)
python3 -m src.model --model-type lstm --input data/features/features_nifty50.csv --output models
```

For regression instead of classification (predicting exact returns rather than direction):

```bash
python3 -m src.model --model-type xgboost --regression --input data/features/features_nifty50.csv --output models
```

## Step 4: Understanding Training Results

After training, examine the output files in the `models` directory:

```
models/
├── random_forest_20240101_123456.joblib       # The trained model
├── random_forest_20240101_123456_scaler.joblib # Feature scaler
├── random_forest_20240101_123456_metrics.json  # Model metrics
```

View the metrics file to understand your model's performance:

```bash
cat models/random_forest_20240101_123456_metrics.json
```

Look for:

- High accuracy/precision and low MSE/RMSE
- Feature importance to understand which indicators matter most
- Reasonable training history (for LSTM models)

## Step 5: Backtest Your Model

Once you have a trained model, backtest it to evaluate its performance:

```bash
# Find your best model file in the models directory
ls -la models/

# Backtest using your model
python3 -m src.backtest --input data/features/features_nifty50.csv --model models/your_model_name.joblib --scaler models/your_model_name_scaler.joblib
```

The backtest will:

1. Use your model to generate trading signals on historical data
2. Simulate trading with realistic constraints (position sizing, stop-loss, take-profit)
3. Calculate performance metrics (total return, Sharpe ratio, drawdown, etc.)
4. Generate visualization plots
5. Save detailed results to the `results/backtest` directory

## Step 6: Compare Multiple Models (Optional)

To compare performance across different models:

```bash
# Run backtests with different models, then analyze the results in results/backtest/
```

You can also use the Python API to compare multiple strategies:

```python
from src.backtest import compare_strategies
import pandas as pd
from src.model import ModelPredictor

# Load data
data = pd.read_csv('data/features/features_nifty50.csv', index_col=0, parse_dates=True)

# Load different models
xgb_predictor = ModelPredictor(
    model_path='models/xgboost_20240101_123456.joblib',
    scaler_path='models/xgboost_20240101_123456_scaler.joblib'
)

rf_predictor = ModelPredictor(
    model_path='models/random_forest_20240101_123456.joblib',
    scaler_path='models/random_forest_20240101_123456_scaler.joblib'
)

# Define strategies to compare
strategies = [
    {'name': 'XGBoost', 'method': 'model', 'model_predictor': xgb_predictor},
    {'name': 'Random Forest', 'method': 'model', 'model_predictor': rf_predictor},
    {'name': 'Moving Average', 'method': 'moving_average', 'params': {'short_window': 50, 'long_window': 200}},
    {'name': 'RSI', 'method': 'rsi', 'params': {'window': 14, 'overbought': 70, 'oversold': 30}}
]

# Compare strategies
results = compare_strategies(data, strategies)
```

## Step 7: Fine-Tune Your Model (Optional)

If performance isn't satisfactory:

1. Try different model types or parameters by editing `config.py`
2. Modify feature engineering parameters
3. Adjust trading parameters like stop-loss, take-profit, and position sizing

Some parameters to adjust in `config.py`:

```python
# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,  # Try increasing from 100
        'max_depth': 15,      # Try increasing from 10
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,  # Try lowering from 0.1
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    # ...other models...
}

# Backtesting parameters
STOP_LOSS = 0.03       # Try tighter stop loss (was 0.05)
TAKE_PROFIT = 0.08     # Try lower take profit (was 0.10)
POSITION_SIZE = 0.15   # Try smaller position size (was 0.20)
```

## Step 8: Prepare for Live Trading

Once you're satisfied with the backtest performance:

```bash
# Update trading configuration in config.py (set API keys, etc.)

# Find the best model based on backtest performance
best_model_path="models/your_best_model.joblib"
best_scaler_path="models/your_best_model_scaler.joblib"

# Run a live trading test (paper trading)
python3 -m src.trading --model $best_model_path --scaler $best_scaler_path --symbols "NIFTY 50"

# Or set up scheduled trading
python3 -m src.trading --model $best_model_path --scaler $best_scaler_path --schedule --run-time 09:20
```

## Common Issues and Solutions

- **Missing Features**: If `ta-lib` isn't installed, some technical indicators won't be calculated. Install it with:

  ```
  # For macOS
  brew install ta-lib
  pip install ta-lib

  # For Ubuntu/Debian
  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
  tar -xzf ta-lib-0.4.0-src.tar.gz
  cd ta-lib/
  ./configure --prefix=/usr
  make
  sudo make install
  pip install ta-lib
  ```

- **Memory Issues**: If you encounter memory problems with large datasets, try:

  1. Reducing the date range
  2. Using a subset of features (set `minimal_features: True` in `FEATURE_ENGINEERING_PARAMS`)
  3. Running on a machine with more RAM

- **Model Performance Issues**: If the model doesn't perform well:
  1. Try different model types
  2. Adjust hyperparameters in `config.py`
  3. Engineer additional features
  4. Ensure there's no data leakage
  5. Consider using feature selection to focus on the most predictive features

## Advanced Training Techniques

### Feature Selection

To improve model performance, you can implement feature selection:

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('data/features/features_nifty50.csv', index_col=0, parse_dates=True)

# Create target variable
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna(subset=['target'])

# Exclude non-feature columns
exclude_columns = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Date']
features = [col for col in data.columns if col not in exclude_columns]
X = data[features]
y = data['target']

# Create a feature selector
selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
selector.fit(X, y)

# Get selected features
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features}")

# Use these features in your model training
```

### Cross-Validation for Time Series

For more robust evaluation, implement time-series cross-validation:

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load your data
# ... (same as above) ...

# Create time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Perform cross-validation
cv_scores = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train and evaluate model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    cv_scores.append(score)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f}")
```

### Ensemble Methods

Combine multiple models for better performance:

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# ... (load data as above) ...

# Create base models
rf = RandomForestClassifier(n_estimators=100)
xgb_model = xgb.XGBClassifier(n_estimators=100)
lgbm_model = lgb.LGBMClassifier(n_estimators=100)

# Create voting ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model), ('lgbm', lgbm_model)],
    voting='soft'
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Evaluate
accuracy = ensemble.score(X_test, y_test)
print(f"Ensemble accuracy: {accuracy:.4f}")
```

This comprehensive guide should help you successfully train, evaluate, and implement your Nifty 50 trading algorithm. Remember that successful algorithmic trading requires continuous refinement and adaptation to changing market conditions.
