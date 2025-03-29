"""
Backtesting functionality for Nifty 50 trading algorithm
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURES_DATA_DIR,
    MODELS_DIR,
    BACKTEST_RESULTS_DIR,
    INITIAL_CAPITAL,
    POSITION_SIZE,
    STOP_LOSS,
    TAKE_PROFIT,
    TRADING_COST
)
from src.model import ModelPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data
    """
    
    def __init__(self, data=None, model_predictor=None, initial_capital=INITIAL_CAPITAL, 
                 position_size=POSITION_SIZE, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, 
                 trading_cost=TRADING_COST, output_dir=BACKTEST_RESULTS_DIR):
        """
        Initialize the backtest engine
        
        Args:
            data (pd.DataFrame): Historical price data with features
            model_predictor (ModelPredictor): Trained model predictor
            initial_capital (float): Initial capital for the backtest
            position_size (float): Position size as a percentage of capital
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            trading_cost (float): Trading cost per trade
            output_dir (str): Directory to save backtest results
        """
        self.data = data
        self.model_predictor = model_predictor
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trading_cost = trading_cost
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results
        self.results = None
        self.trades = None
        self.metrics = None
        
        logger.info("Initialized backtest engine")
    
    def set_data(self, data):
        """
        Set the data for backtesting
        
        Args:
            data (pd.DataFrame): Historical price data with features
        """
        self.data = data
        logger.info(f"Set backtest data with shape {data.shape}")
    
    def set_model_predictor(self, model_predictor):
        """
        Set the model predictor for backtesting
        
        Args:
            model_predictor (ModelPredictor): Trained model predictor
        """
        self.model_predictor = model_predictor
        logger.info("Set model predictor")
    
    def generate_signals(self, method='model', threshold=0.5, strategy_params=None):
        """
        Generate trading signals for backtesting
        
        Args:
            method (str): Method to generate signals ('model', 'moving_average', 'rsi', etc.)
            threshold (float): Probability threshold for model signals
            strategy_params (dict): Parameters for the strategy
            
        Returns:
            pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        logger.info(f"Generating signals using method '{method}'")
        
        if self.data is None:
            raise ValueError("No data provided. Please set data before generating signals.")
        
        signals = pd.Series(0, index=self.data.index)
        
        if method == 'model':
            if self.model_predictor is None:
                raise ValueError("No model predictor provided. Please set model predictor before generating signals.")
            
            # Get model predictions
            try:
                if hasattr(self.model_predictor, 'predict_proba'):
                    probas = self.model_predictor.predict_proba(self.data)
                    # Extract probability of positive class
                    if isinstance(probas, np.ndarray) and probas.ndim > 1:
                        buy_probas = probas[:, 1]
                    else:
                        buy_probas = probas
                    
                    # Generate signals based on probability threshold
                    signals = pd.Series(0, index=self.data.index)
                    signals[buy_probas > threshold] = 1
                    signals[buy_probas < (1 - threshold)] = -1
                else:
                    # Use simple predictions
                    preds = self.model_predictor.predict(self.data)
                    signals[preds == 1] = 1
                    signals[preds == 0] = -1
            except Exception as e:
                logger.error(f"Error generating model signals: {str(e)}")
                raise
                
        elif method == 'moving_average':
            # Default parameters
            params = {'short_window': 50, 'long_window': 200}
            if strategy_params:
                params.update(strategy_params)
            
            # Calculate moving averages
            short_ma = self.data['Close'].rolling(window=params['short_window']).mean()
            long_ma = self.data['Close'].rolling(window=params['long_window']).mean()
            
            # Generate signals
            signals[short_ma > long_ma] = 1
            signals[short_ma < long_ma] = -1
            
        elif method == 'rsi':
            # Default parameters
            params = {'window': 14, 'overbought': 70, 'oversold': 30}
            if strategy_params:
                params.update(strategy_params)
            
            # Calculate RSI
            if 'RSI' in self.data.columns:
                rsi = self.data['RSI']
            else:
                delta = self.data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=params['window']).mean()
                avg_loss = loss.rolling(window=params['window']).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            signals[rsi < params['oversold']] = 1
            signals[rsi > params['overbought']] = -1
            
        elif method == 'macd':
            # Default parameters
            params = {'fast': 12, 'slow': 26, 'signal': 9}
            if strategy_params:
                params.update(strategy_params)
            
            # Use existing MACD if available
            if all(col in self.data.columns for col in ['MACD', 'MACD_Signal']):
                macd = self.data['MACD']
                signal = self.data['MACD_Signal']
            else:
                # Calculate MACD
                fast_ema = self.data['Close'].ewm(span=params['fast']).mean()
                slow_ema = self.data['Close'].ewm(span=params['slow']).mean()
                macd = fast_ema - slow_ema
                signal = macd.ewm(span=params['signal']).mean()
            
            # Generate signals
            signals[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1  # Bullish crossover
            signals[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1  # Bearish crossover
            
        else:
            raise ValueError(f"Unsupported signal generation method: {method}")
        
        logger.info(f"Generated {signals[signals != 0].count()} signals")
        
        return signals
    
    def run_backtest(self, signals=None, method='model', threshold=0.5, strategy_params=None):
        """
        Run a backtest using the provided signals or generate signals using the specified method
        
        Args:
            signals (pd.Series): Pre-generated trading signals
            method (str): Method to generate signals if not provided
            threshold (float): Probability threshold for model signals
            strategy_params (dict): Parameters for the strategy
            
        Returns:
            pd.DataFrame: Backtest results
        """
        logger.info("Running backtest")
        
        if self.data is None:
            raise ValueError("No data provided. Please set data before running backtest.")
        
        # Generate signals if not provided
        if signals is None:
            signals = self.generate_signals(method, threshold, strategy_params)
        
        # Ensure 'Close' column exists
        if 'Close' not in self.data.columns:
            raise ValueError("Data must contain a 'Close' column")
        
        # Create a DataFrame to store results
        results = pd.DataFrame(index=self.data.index)
        results['Close'] = self.data['Close']
        results['Signal'] = signals
        
        # Add trading logic with stop loss and take profit
        position = 0
        entry_price = 0
        capital = self.initial_capital
        shares = 0
        trades = []
        
        for i, (idx, row) in enumerate(results.iterrows()):
            price = row['Close']
            signal = row['Signal']
            
            # Initialize positions and returns
            results.loc[idx, 'Position'] = position
            results.loc[idx, 'Capital'] = capital
            
            # Skip if not enough data for stop loss/take profit
            if i == 0:
                continue
            
            # Check for stop loss or take profit if in a position
            if position != 0:
                prev_price = results.iloc[i-1]['Close']
                price_change = (price - entry_price) / entry_price
                
                # Apply stop loss
                if (position == 1 and price_change <= -self.stop_loss) or \
                   (position == -1 and price_change >= self.stop_loss):
                    # Close position due to stop loss
                    trade_return = price - entry_price if position == 1 else entry_price - price
                    trade_value = shares * price
                    trade_cost = trade_value * self.trading_cost
                    capital = capital + trade_value - trade_cost
                    
                    # Record the trade
                    trade_end = idx
                    trade_result = {
                        'entry_date': trade_start,
                        'exit_date': trade_end,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': 'Long' if position == 1 else 'Short',
                        'shares': shares,
                        'return': (trade_return * shares) - trade_cost,
                        'return_pct': (price - entry_price) / entry_price * 100 if position == 1 else \
                                     (entry_price - price) / entry_price * 100,
                        'exit_reason': 'Stop Loss'
                    }
                    trades.append(trade_result)
                    
                    position = 0
                    shares = 0
                    
                # Apply take profit
                elif (position == 1 and price_change >= self.take_profit) or \
                     (position == -1 and price_change <= -self.take_profit):
                    # Close position due to take profit
                    trade_return = price - entry_price if position == 1 else entry_price - price
                    trade_value = shares * price
                    trade_cost = trade_value * self.trading_cost
                    capital = capital + trade_value - trade_cost
                    
                    # Record the trade
                    trade_end = idx
                    trade_result = {
                        'entry_date': trade_start,
                        'exit_date': trade_end,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': 'Long' if position == 1 else 'Short',
                        'shares': shares,
                        'return': (trade_return * shares) - trade_cost,
                        'return_pct': (price - entry_price) / entry_price * 100 if position == 1 else \
                                     (entry_price - price) / entry_price * 100,
                        'exit_reason': 'Take Profit'
                    }
                    trades.append(trade_result)
                    
                    position = 0
                    shares = 0
            
            # Execute new signal if not in a position
            if position == 0 and signal != 0:
                # Calculate position size
                position_value = capital * self.position_size
                shares = position_value / price
                trade_cost = position_value * self.trading_cost
                capital = capital - trade_cost
                
                position = signal
                entry_price = price
                trade_start = idx
                
            # Update position value
            if position != 0:
                position_value = shares * price
                results.loc[idx, 'Position_Value'] = position_value
            else:
                results.loc[idx, 'Position_Value'] = 0
            
            # Update capital
            results.loc[idx, 'Capital'] = capital
            
            # Close out any open position at the end
            if i == len(results) - 1 and position != 0:
                trade_return = price - entry_price if position == 1 else entry_price - price
                trade_value = shares * price
                trade_cost = trade_value * self.trading_cost
                capital = capital + trade_value - trade_cost
                
                # Record the trade
                trade_end = idx
                trade_result = {
                    'entry_date': trade_start,
                    'exit_date': trade_end,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position': 'Long' if position == 1 else 'Short',
                    'shares': shares,
                    'return': (trade_return * shares) - trade_cost,
                    'return_pct': (price - entry_price) / entry_price * 100 if position == 1 else \
                                 (entry_price - price) / entry_price * 100,
                    'exit_reason': 'End of Backtest'
                }
                trades.append(trade_result)
        
        # Calculate portfolio value (capital + position value)
        results['Portfolio_Value'] = results['Capital'] + results['Position_Value'].fillna(0)
        
        # Calculate daily returns
        results['Daily_Return'] = results['Portfolio_Value'].pct_change()
        
        # Calculate cumulative returns
        results['Cumulative_Return'] = (1 + results['Daily_Return']).cumprod() - 1
        
        # Save results and trades
        self.results = results
        self.trades = pd.DataFrame(trades)
        
        # Calculate metrics
        self.calculate_metrics()
        
        logger.info("Backtest completed")
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate backtest performance metrics
        
        Returns:
            dict: Dictionary of performance metrics
        """
        logger.info("Calculating performance metrics")
        
        if self.results is None:
            raise ValueError("No backtest results available. Please run a backtest first.")
        
        # Basic metrics
        total_return = self.results['Portfolio_Value'].iloc[-1] / self.initial_capital - 1
        
        # Calculate annualized return
        days = (self.results.index[-1] - self.results.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        daily_returns = self.results['Daily_Return'].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = self.results['Cumulative_Return']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # Calculate trade metrics
        if self.trades is not None and len(self.trades) > 0:
            num_trades = len(self.trades)
            win_trades = self.trades[self.trades['return'] > 0]
            loss_trades = self.trades[self.trades['return'] <= 0]
            win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
            avg_win = win_trades['return_pct'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['return_pct'].mean() if len(loss_trades) > 0 else 0
            profit_factor = abs(win_trades['return'].sum() / loss_trades['return'].sum()) if loss_trades['return'].sum() != 0 else float('inf')
            
            # Average holding period
            self.trades['holding_days'] = (pd.to_datetime(self.trades['exit_date']) - 
                                          pd.to_datetime(self.trades['entry_date'])).dt.days
            avg_holding_period = self.trades['holding_days'].mean()
        else:
            num_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_period = 0
        
        # Store metrics
        self.metrics = {
            'initial_capital': float(self.initial_capital),
            'final_portfolio_value': float(self.results['Portfolio_Value'].iloc[-1]),
            'total_return_pct': float(total_return * 100),
            'annualized_return_pct': float(annual_return * 100),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': float(max_drawdown * 100),
            'num_trades': int(num_trades),
            'win_rate_pct': float(win_rate * 100),
            'avg_win_pct': float(avg_win),
            'avg_loss_pct': float(avg_loss),
            'profit_factor': float(profit_factor),
            'avg_holding_period_days': float(avg_holding_period)
        }
        
        logger.info(f"Calculated metrics: Total Return: {total_return*100:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}, Win Rate: {win_rate*100:.2f}%")
        
        return self.metrics
    
    def plot_results(self, show_signals=True, save_path=None):
        """
        Plot backtest results
        
        Args:
            show_signals (bool): Whether to show buy/sell signals on the plot
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        logger.info("Plotting backtest results")
        
        if self.results is None:
            raise ValueError("No backtest results available. Please run a backtest first.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price chart
        ax1.plot(self.results.index, self.results['Close'], color='black', linewidth=1, label='Close Price')
        
        # Add buy and sell signals if requested
        if show_signals and 'Signal' in self.results.columns:
            buy_signals = self.results[self.results['Signal'] == 1].index
            sell_signals = self.results[self.results['Signal'] == -1].index
            
            ax1.scatter(buy_signals, self.results.loc[buy_signals, 'Close'], marker='^', color='green', s=100, label='Buy Signal')
            ax1.scatter(sell_signals, self.results.loc[sell_signals, 'Close'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot portfolio value
        ax2.plot(self.results.index, self.results['Portfolio_Value'], color='blue', linewidth=1.5, label='Portfolio Value')
        
        # Calculate drawdown
        portfolio_value = self.results['Portfolio_Value']
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - running_max) / running_max
        
        # Plot drawdown as area chart
        ax2.fill_between(self.results.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        
        # Add formatting
        ax1.set_title('Backtest Results', fontsize=16)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Portfolio Value / Drawdown', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add metrics as text
        if self.metrics:
            metrics_text = (
                f"Total Return: {self.metrics['total_return_pct']:.2f}%\n"
                f"Annual Return: {self.metrics['annualized_return_pct']:.2f}%\n"
                f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {self.metrics['max_drawdown_pct']:.2f}%\n"
                f"Win Rate: {self.metrics['win_rate_pct']:.2f}%"
            )
            
            # Add text box with metrics
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def save_results(self, name=None):
        """
        Save backtest results, trades, and metrics
        
        Args:
            name (str): Name prefix for the saved files
            
        Returns:
            dict: Paths to saved files
        """
        logger.info("Saving backtest results")
        
        if self.results is None:
            raise ValueError("No backtest results available. Please run a backtest first.")
        
        # Create a timestamp and name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            prefix = f"{name}_{timestamp}"
        else:
            prefix = timestamp
        
        # Create directories
        results_dir = Path(self.output_dir) / prefix
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_path = results_dir / "results.csv"
        self.results.to_csv(results_path)
        
        # Save trades
        if self.trades is not None:
            trades_path = results_dir / "trades.csv"
            self.trades.to_csv(trades_path)
        else:
            trades_path = None
        
        # Save metrics
        if self.metrics:
            metrics_path = results_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        else:
            metrics_path = None
        
        # Save plot
        plot_path = results_dir / "plot.png"
        self.plot_results(save_path=plot_path)
        
        logger.info(f"Results saved to {results_dir}")
        
        return {
            'results_dir': str(results_dir),
            'results_path': str(results_path),
            'trades_path': str(trades_path) if trades_path else None,
            'metrics_path': str(metrics_path) if metrics_path else None,
            'plot_path': str(plot_path)
        }


def compare_strategies(data, strategies, initial_capital=INITIAL_CAPITAL, output_dir=BACKTEST_RESULTS_DIR):
    """
    Compare multiple trading strategies on the same data
    
    Args:
        data (pd.DataFrame): Historical price data with features
        strategies (list): List of strategy dictionaries with name, method, and parameters
        initial_capital (float): Initial capital for the backtest
        output_dir (str): Directory to save comparison results
        
    Returns:
        dict: Comparison metrics and paths to saved files
    """
    logger.info(f"Comparing {len(strategies)} strategies")
    
    # Create a backtest engine
    engine = BacktestEngine(data=data, initial_capital=initial_capital, output_dir=output_dir)
    
    # Run backtest for each strategy
    results = {}
    metrics = {}
    
    for strategy in strategies:
        name = strategy.get('name', f"Strategy_{len(results)+1}")
        method = strategy.get('method', 'model')
        threshold = strategy.get('threshold', 0.5)
        params = strategy.get('params', None)
        model_predictor = strategy.get('model_predictor', None)
        
        logger.info(f"Running backtest for strategy: {name}")
        
        # Set model predictor if provided
        if model_predictor:
            engine.set_model_predictor(model_predictor)
        
        # Run backtest
        backtest_results = engine.run_backtest(method=method, threshold=threshold, strategy_params=params)
        
        # Store results and metrics
        results[name] = backtest_results
        metrics[name] = engine.metrics
        
        # Save individual strategy results
        engine.save_results(name=name)
    
    # Create a comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot price in the top subplot
    ax1.plot(data.index, data['Close'], color='black', linewidth=1, label='Close Price')
    ax1.set_title('Strategy Comparison', fontsize=16)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative returns for each strategy in the bottom subplot
    for name, result in results.items():
        ax2.plot(result.index, result['Portfolio_Value'], linewidth=1.5, label=f"{name}")
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Portfolio Value', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add metrics as a table
    metrics_table = []
    headers = ["Strategy", "Total Return", "Annual Return", "Sharpe", "Max DD", "Win Rate"]
    
    for name, metric in metrics.items():
        metrics_table.append([
            name,
            f"{metric['total_return_pct']:.2f}%",
            f"{metric['annualized_return_pct']:.2f}%",
            f"{metric['sharpe_ratio']:.2f}",
            f"{metric['max_drawdown_pct']:.2f}%",
            f"{metric['win_rate_pct']:.2f}%"
        ])
    
    plt.tight_layout()
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison directory
    comparison_dir = Path(output_dir) / f"comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison plot
    plot_path = comparison_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Save metrics
    metrics_path = comparison_dir / "comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Strategy comparison saved to {comparison_dir}")
    
    return {
        'metrics': metrics,
        'comparison_dir': str(comparison_dir),
        'plot_path': str(plot_path),
        'metrics_path': str(metrics_path)
    }


def main():
    """
    Main function for backtesting
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Backtest trading strategies for Nifty 50 trading algorithm')
    parser.add_argument('--input', '-i', type=str, default=None, 
                        help='Input file with features data')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--scaler', '-s', type=str, default=None,
                        help='Path to scaler file')
    parser.add_argument('--output', '-o', type=str, default=BACKTEST_RESULTS_DIR,
                        help='Output directory for backtest results')
    parser.add_argument('--method', type=str, default='model', 
                        choices=['model', 'moving_average', 'rsi', 'macd'],
                        help='Signal generation method')
    parser.add_argument('--capital', '-c', type=float, default=INITIAL_CAPITAL,
                        help='Initial capital for backtest')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Probability threshold for model signals')
    
    args = parser.parse_args()
    
    logger.info("Starting backtest")
    
    # Find the input file if not specified
    input_file = args.input
    if not input_file:
        # Use the latest features file in the features data directory
        feature_files = [f for f in os.listdir(FEATURES_DATA_DIR) if f.startswith('features_') and f.endswith('.csv')]
        if feature_files:
            # Sort by modification time (latest first)
            feature_files.sort(key=lambda x: os.path.getmtime(os.path.join(FEATURES_DATA_DIR, x)), reverse=True)
            input_file = os.path.join(FEATURES_DATA_DIR, feature_files[0])
            logger.info(f"Using latest features file: {input_file}")
        else:
            logger.error("No features files found in the input directory")
            return
    
    # Load the data
    data = pd.read_csv(input_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded data with shape {data.shape}")
    
    # Create model predictor if model path is provided
    model_predictor = None
    if args.method == 'model':
        if args.model and args.scaler:
            model_path = args.model
            scaler_path = args.scaler
        else:
            # Find the best model
            from src.model import find_best_model
            model_path, scaler_path = find_best_model(MODELS_DIR)
            
        if model_path and scaler_path:
            from src.model import ModelPredictor
            model_predictor = ModelPredictor(model_path=model_path, scaler_path=scaler_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error("No model found for backtesting")
            return
    
    # Initialize backtest engine
    engine = BacktestEngine(
        data=data,
        model_predictor=model_predictor,
        initial_capital=args.capital,
        output_dir=args.output
    )
    
    # Run backtest
    results = engine.run_backtest(method=args.method, threshold=args.threshold)
    
    # Save results
    saved_paths = engine.save_results()
    
    logger.info("Backtest completed")
    
    return saved_paths


if __name__ == "__main__":
    main()
