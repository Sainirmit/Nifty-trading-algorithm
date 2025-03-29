"""
Live trading implementation for Nifty 50 trading algorithm
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
import time
import schedule
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURES_DATA_DIR,
    MODELS_DIR,
    TRADING_RESULTS_DIR,
    POSITION_SIZE,
    STOP_LOSS,
    TAKE_PROFIT,
    TRADING_COST,
    EMAIL_NOTIFICATIONS,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECIPIENTS,
    API_KEY,
    API_SECRET,
    BROKER_API_URL
)
from src.model import ModelPredictor, find_best_model
from src.feature_engineering import add_basic_price_features, add_technical_indicators, add_candlestick_patterns, add_custom_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(TRADING_RESULTS_DIR, 'trading.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create trading results directory if it doesn't exist
os.makedirs(TRADING_RESULTS_DIR, exist_ok=True)

class BrokerAPI:
    """
    API wrapper for broker interactions
    
    This is a placeholder implementation. In a real trading system, this would be
    replaced with actual broker-specific API calls using libraries like pykiteconnect
    for Zerodha/Kite, or interactive brokers API, etc.
    """
    
    def __init__(self, api_key=API_KEY, api_secret=API_SECRET, api_url=BROKER_API_URL):
        """
        Initialize the broker API
        
        Args:
            api_key (str): API key for the broker
            api_secret (str): API secret for the broker
            api_url (str): Base URL for the broker's API
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url
        self.session = requests.Session()
        self.access_token = None
        
        # Simulate authentication
        if self.api_key and self.api_secret:
            self._authenticate()
        
        logger.info("Initialized broker API")
    
    def _authenticate(self):
        """
        Authenticate with the broker API
        
        In a real implementation, this would make an API call to the broker's authentication endpoint.
        For this placeholder, we'll just simulate authentication.
        
        Returns:
            bool: True if authentication was successful
        """
        try:
            logger.info("Authenticating with broker API")
            
            # Simulated authentication
            if self.api_key and self.api_secret:
                self.access_token = "SIMULATED_ACCESS_TOKEN"
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'X-API-Key': self.api_key
                })
                logger.info("Authentication successful")
                return True
            else:
                logger.warning("API key or secret not provided")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def get_current_data(self, symbol):
        """
        Get current market data for a symbol
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            dict: Market data including latest price, OHLC, etc.
        """
        try:
            logger.info(f"Getting current data for {symbol}")
            
            # In a real implementation, this would make an API call to the broker's market data endpoint
            # For this placeholder, we'll simulate by using a dummy response
            
            # Current time
            now = datetime.now()
            
            # Simulated data (in a real implementation, this would be from the broker API)
            # For testing purposes, we're creating a stub that returns simulated data
            last_price = float(np.random.normal(18500, 100))  # Simulated Nifty 50 price
            
            data = {
                'symbol': symbol,
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'last_price': last_price,
                'open': last_price * 0.995,
                'high': last_price * 1.005,
                'low': last_price * 0.993,
                'close': last_price,
                'volume': int(np.random.normal(10000000, 2000000))
            }
            
            logger.info(f"Retrieved current data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting current data: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, interval='day', days=60):
        """
        Get historical market data for a symbol
        
        Args:
            symbol (str): Symbol to get data for
            interval (str): Data interval ('minute', 'hour', 'day')
            days (int): Number of days of historical data to retrieve
            
        Returns:
            pd.DataFrame: Historical market data
        """
        try:
            logger.info(f"Getting historical data for {symbol} ({interval}, {days} days)")
            
            # In a real implementation, this would make an API call to the broker's historical data endpoint
            # For this placeholder, we'll simulate by generating random data
            
            # Generate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if interval == 'minute':
                # Minute data (only for market hours, 9:15 AM to 3:30 PM, weekdays)
                dates = []
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Monday to Friday
                        for hour in range(9, 16):
                            for minute in range(0, 60, 1):
                                if (hour == 9 and minute < 15) or (hour == 15 and minute > 30):
                                    continue
                                dates.append(current_date.replace(hour=hour, minute=minute))
                    current_date += timedelta(days=1)
                
                dates = dates[-60*6.25*days:]  # Approximately 6.25 hours per day
                
            elif interval == 'hour':
                # Hourly data (only for market hours, 9 AM to 3 PM, weekdays)
                dates = []
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Monday to Friday
                        for hour in range(9, 16):
                            dates.append(current_date.replace(hour=hour, minute=0))
                    current_date += timedelta(days=1)
                
                dates = dates[-7*days:]  # Approximately 7 hours per day
                
            else:  # Daily data
                # Daily data (only weekdays)
                dates = []
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Monday to Friday
                        dates.append(current_date)
                    current_date += timedelta(days=1)
            
            # Create a DataFrame with simulated data
            seed = hash(symbol) % 10000  # Use symbol to seed the random generator for consistent simulation
            np.random.seed(seed)
            
            # Start with a base price
            if symbol == 'NIFTY 50':
                base_price = 18500  # Simulated Nifty 50 index
            else:
                base_price = 1000  # Default base price for stocks
            
            # Generate random walk prices
            price_changes = np.random.normal(0, 0.01, len(dates))
            price_multipliers = np.cumprod(1 + price_changes)
            prices = base_price * price_multipliers
            
            # Generate OHLCV data
            df = pd.DataFrame(index=dates)
            df['Open'] = prices * np.random.uniform(0.995, 1.000, len(dates))
            df['High'] = prices * np.random.uniform(1.000, 1.010, len(dates))
            df['Low'] = prices * np.random.uniform(0.990, 0.995, len(dates))
            df['Close'] = prices
            df['Volume'] = np.random.normal(10000000, 2000000, len(dates)).astype(int)
            df.index.name = 'Date'
            
            logger.info(f"Retrieved historical data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def place_order(self, symbol, order_type, quantity, price=None, trigger_price=None):
        """
        Place an order with the broker
        
        Args:
            symbol (str): Symbol to trade
            order_type (str): Type of order ('BUY', 'SELL')
            quantity (int): Number of shares to trade
            price (float): Limit price (optional)
            trigger_price (float): Trigger price for stop orders (optional)
            
        Returns:
            dict: Order details including order ID
        """
        try:
            logger.info(f"Placing {order_type} order for {quantity} shares of {symbol}")
            
            # In a real implementation, this would make an API call to the broker's order placement endpoint
            # For this placeholder, we'll simulate by returning a dummy order ID
            
            # Current time
            now = datetime.now()
            
            # Simulated order details
            order_id = f"ORD{int(time.time())}"
            
            # Simulated order response
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'trigger_price': trigger_price,
                'status': 'OPEN',
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Order placed successfully: {order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def get_order_status(self, order_id):
        """
        Get the status of an order
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order details including status
        """
        try:
            logger.info(f"Getting status for order {order_id}")
            
            # In a real implementation, this would make an API call to the broker's order status endpoint
            # For this placeholder, we'll simulate by returning a dummy status
            
            # Simulated order status (randomly completed or open)
            status = np.random.choice(['COMPLETE', 'OPEN'], p=[0.8, 0.2])
            
            # Simulated order details
            order = {
                'order_id': order_id,
                'status': status,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Order {order_id} status: {status}")
            return order
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return None
    
    def get_portfolio(self):
        """
        Get current portfolio holdings
        
        Returns:
            dict: Portfolio details including positions, funds, etc.
        """
        try:
            logger.info("Getting portfolio details")
            
            # In a real implementation, this would make an API call to the broker's portfolio endpoint
            # For this placeholder, we'll simulate by returning dummy positions
            
            # Simulated portfolio details
            portfolio = {
                'funds': 100000.0,
                'positions': [
                    {
                        'symbol': 'NIFTY 50',
                        'quantity': 10,
                        'average_price': 18400.0,
                        'last_price': 18500.0,
                        'pnl': 1000.0
                    }
                ],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("Retrieved portfolio details")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {str(e)}")
            return None


class TradingEngine:
    """
    Trading engine for executing the trading strategy
    """
    
    def __init__(self, model_path=None, scaler_path=None, symbols=None, 
                 position_size=POSITION_SIZE, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT,
                 broker=None, output_dir=TRADING_RESULTS_DIR):
        """
        Initialize the trading engine
        
        Args:
            model_path (str): Path to the trained model
            scaler_path (str): Path to the scaler
            symbols (list): List of symbols to trade
            position_size (float): Position size as a percentage of capital
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            broker (BrokerAPI): Broker API instance
            output_dir (str): Directory to save trading results
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.symbols = symbols if symbols else ['NIFTY 50']
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.output_dir = output_dir
        
        # Create a ModelPredictor if model path is provided
        self.model_predictor = None
        if model_path and scaler_path:
            self.model_predictor = ModelPredictor(model_path=model_path, scaler_path=scaler_path)
        
        # Create a broker API instance if not provided
        self.broker = broker if broker else BrokerAPI()
        
        # Initialize positions and orders tracking
        self.positions = {}
        self.orders = []
        self.cash = 0.0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing state if available
        self._load_state()
        
        logger.info("Initialized trading engine")
    
    def _load_state(self):
        """
        Load trading state from file
        """
        state_file = os.path.join(self.output_dir, 'trading_state.json')
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.positions = state.get('positions', {})
                self.orders = state.get('orders', [])
                self.cash = state.get('cash', 0.0)
                
                logger.info("Loaded trading state from file")
            except Exception as e:
                logger.error(f"Error loading trading state: {str(e)}")
    
    def _save_state(self):
        """
        Save trading state to file
        """
        state_file = os.path.join(self.output_dir, 'trading_state.json')
        
        try:
            state = {
                'positions': self.positions,
                'orders': self.orders,
                'cash': self.cash,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
            
            logger.info("Saved trading state to file")
        except Exception as e:
            logger.error(f"Error saving trading state: {str(e)}")
    
    def setup(self):
        """
        Setup the trading engine
        
        Returns:
            bool: True if setup was successful
        """
        logger.info("Setting up trading engine")
        
        # Update portfolio information
        portfolio = self.broker.get_portfolio()
        if portfolio:
            self.cash = portfolio.get('funds', 0.0)
            
            # Update positions from portfolio
            for position in portfolio.get('positions', []):
                symbol = position.get('symbol')
                if symbol in self.symbols:
                    self.positions[symbol] = {
                        'quantity': position.get('quantity', 0),
                        'entry_price': position.get('average_price', 0.0),
                        'last_price': position.get('last_price', 0.0),
                        'entry_date': None  # We don't know the entry date from the portfolio
                    }
        
        # If model predictor is not provided, find the best model
        if not self.model_predictor:
            model_path, scaler_path = find_best_model(MODELS_DIR)
            
            if model_path and scaler_path:
                self.model_predictor = ModelPredictor(model_path=model_path, scaler_path=scaler_path)
                self.model_path = model_path
                self.scaler_path = scaler_path
                logger.info(f"Using model: {model_path}")
            else:
                logger.error("No model found for trading")
                return False
        
        # Save initial state
        self._save_state()
        
        logger.info("Trading engine setup complete")
        return True
    
    def get_data_with_features(self, symbol, days=60):
        """
        Get historical data for a symbol and add features
        
        Args:
            symbol (str): Symbol to get data for
            days (int): Number of days of historical data
            
        Returns:
            pd.DataFrame: Historical data with features
        """
        logger.info(f"Getting data with features for {symbol}")
        
        # Get historical data
        data = self.broker.get_historical_data(symbol, interval='day', days=days)
        
        if data is None:
            logger.error(f"Failed to get historical data for {symbol}")
            return None
        
        # Add features
        try:
            # Apply feature engineering functions
            data = add_basic_price_features(data)
            data = add_technical_indicators(data)
            data = add_candlestick_patterns(data)
            data = add_custom_features(data)
            
            logger.info(f"Added features to {symbol} data")
            return data
        except Exception as e:
            logger.error(f"Error adding features: {str(e)}")
            return None
    
    def generate_signal(self, data, threshold=0.5):
        """
        Generate a trading signal for the given data
        
        Args:
            data (pd.DataFrame): Data with features
            threshold (float): Probability threshold for signal generation
            
        Returns:
            int: Signal (1 for buy, -1 for sell, 0 for hold)
        """
        logger.info("Generating trading signal")
        
        if self.model_predictor is None:
            logger.error("No model predictor available")
            return 0
        
        try:
            # Get the latest row of data
            latest_data = data.iloc[-1:].copy()
            
            # Generate signal
            if hasattr(self.model_predictor, 'predict_proba'):
                # Get probabilities
                probas = self.model_predictor.predict_proba(latest_data)
                
                # Extract probability of positive class
                if isinstance(probas, np.ndarray) and probas.ndim > 1:
                    buy_proba = probas[0, 1]
                else:
                    buy_proba = probas[0]
                
                # Generate signal based on probability threshold
                if buy_proba > threshold:
                    signal = 1  # Buy
                elif buy_proba < (1 - threshold):
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold
                
                logger.info(f"Generated signal: {signal} (buy probability: {buy_proba:.4f})")
            else:
                # Use simple prediction
                pred = self.model_predictor.predict(latest_data)[0]
                signal = 1 if pred == 1 else -1
                
                logger.info(f"Generated signal: {signal}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return 0
    
    def execute_trade(self, symbol, signal):
        """
        Execute a trade based on the signal
        
        Args:
            symbol (str): Symbol to trade
            signal (int): Trading signal (1 for buy, -1 for sell, 0 for hold)
            
        Returns:
            dict: Trade details
        """
        logger.info(f"Executing trade for {symbol} with signal {signal}")
        
        # Get current data for the symbol
        current_data = self.broker.get_current_data(symbol)
        
        if current_data is None:
            logger.error(f"Failed to get current data for {symbol}")
            return None
        
        current_price = current_data.get('last_price', 0.0)
        
        # Check if we already have a position in this symbol
        has_position = symbol in self.positions and self.positions[symbol].get('quantity', 0) > 0
        
        # Determine action based on signal and current position
        action = None
        quantity = 0
        
        if signal == 1 and not has_position:
            # Buy signal and no position -> buy
            action = 'BUY'
            
            # Calculate quantity based on position size
            position_value = self.cash * self.position_size
            quantity = int(position_value / current_price)
            
        elif signal == -1 and has_position:
            # Sell signal and have position -> sell
            action = 'SELL'
            quantity = self.positions[symbol].get('quantity', 0)
            
        elif signal == 0:
            # Hold signal -> check for stop loss or take profit
            if has_position:
                entry_price = self.positions[symbol].get('entry_price', 0.0)
                price_change = (current_price - entry_price) / entry_price
                
                if price_change <= -self.stop_loss:
                    # Stop loss triggered
                    action = 'SELL'
                    quantity = self.positions[symbol].get('quantity', 0)
                    logger.info(f"Stop loss triggered for {symbol}")
                    
                elif price_change >= self.take_profit:
                    # Take profit triggered
                    action = 'SELL'
                    quantity = self.positions[symbol].get('quantity', 0)
                    logger.info(f"Take profit triggered for {symbol}")
        
        # Execute the trade if an action was determined
        if action and quantity > 0:
            order = self.broker.place_order(symbol, action, quantity)
            
            if order:
                # Update orders list
                self.orders.append(order)
                
                # Update positions and cash
                if action == 'BUY':
                    # Update cash
                    trade_value = quantity * current_price
                    self.cash -= trade_value
                    
                    # Update position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'last_price': current_price,
                        'entry_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    logger.info(f"Bought {quantity} shares of {symbol} at {current_price}")
                    
                elif action == 'SELL':
                    # Update cash
                    trade_value = quantity * current_price
                    self.cash += trade_value
                    
                    # Update position
                    if symbol in self.positions:
                        self.positions.pop(symbol)
                    
                    logger.info(f"Sold {quantity} shares of {symbol} at {current_price}")
                
                # Save state after trade
                self._save_state()
                
                # Send notification
                self._send_trade_notification(symbol, action, quantity, current_price)
                
                return {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': current_price,
                    'order_id': order.get('order_id'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return None
    
    def run_strategy(self, symbol):
        """
        Run the trading strategy for a symbol
        
        Args:
            symbol (str): Symbol to trade
            
        Returns:
            dict: Strategy results
        """
        logger.info(f"Running strategy for {symbol}")
        
        # Get data with features
        data = self.get_data_with_features(symbol)
        
        if data is None:
            logger.error(f"Failed to get data for {symbol}")
            return None
        
        # Generate signal
        signal = self.generate_signal(data)
        
        # Execute trade based on signal
        trade = self.execute_trade(symbol, signal)
        
        # Save data for reference
        data_file = os.path.join(self.output_dir, f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv")
        data.to_csv(data_file)
        
        # Return results
        results = {
            'symbol': symbol,
            'signal': signal,
            'trade': trade,
            'data_file': data_file,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results
    
    def update_portfolio(self):
        """
        Update portfolio information from broker
        
        Returns:
            dict: Updated portfolio
        """
        logger.info("Updating portfolio information")
        
        portfolio = self.broker.get_portfolio()
        
        if portfolio:
            self.cash = portfolio.get('funds', 0.0)
            
            # Update positions
            for position in portfolio.get('positions', []):
                symbol = position.get('symbol')
                if symbol in self.symbols:
                    if symbol in self.positions:
                        # Update existing position
                        self.positions[symbol].update({
                            'quantity': position.get('quantity', 0),
                            'last_price': position.get('last_price', 0.0)
                        })
                    else:
                        # Add new position
                        self.positions[symbol] = {
                            'quantity': position.get('quantity', 0),
                            'entry_price': position.get('average_price', 0.0),
                            'last_price': position.get('last_price', 0.0),
                            'entry_date': datetime.now().strftime('%Y-%m-%d')  # Approximate entry date
                        }
            
            # Save state
            self._save_state()
            
            logger.info("Portfolio updated")
            return portfolio
        
        return None
    
    def _send_trade_notification(self, symbol, action, quantity, price):
        """
        Send a notification about a trade
        
        Args:
            symbol (str): Symbol traded
            action (str): Action taken ('BUY' or 'SELL')
            quantity (int): Quantity traded
            price (float): Trade price
        """
        if not EMAIL_NOTIFICATIONS or not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENTS:
            return
        
        try:
            logger.info("Sending trade notification")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = ', '.join(EMAIL_RECIPIENTS)
            msg['Subject'] = f"Trade Alert: {action} {quantity} {symbol} @ {price}"
            
            # Message body
            body = f"""
            Trade Alert:
            
            Symbol: {symbol}
            Action: {action}
            Quantity: {quantity}
            Price: {price}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Current Portfolio:
            Cash: {self.cash}
            Positions: {self.positions}
            
            This is an automated message from the Nifty 50 Trading Algorithm.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            
            logger.info("Trade notification sent")
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {str(e)}")
    
    def run(self):
        """
        Run the trading engine for all symbols
        
        Returns:
            dict: Results for all symbols
        """
        logger.info("Running trading engine")
        
        # Ensure the engine is setup
        if not hasattr(self, 'cash') or self.cash == 0:
            self.setup()
        
        # Update portfolio
        self.update_portfolio()
        
        # Run strategy for each symbol
        results = {}
        
        for symbol in self.symbols:
            try:
                symbol_results = self.run_strategy(symbol)
                results[symbol] = symbol_results
            except Exception as e:
                logger.error(f"Error running strategy for {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        # Save results
        results_file = os.path.join(self.output_dir, f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
        
        return results


def run_daily_strategy():
    """
    Function to run the trading strategy on a daily basis
    """
    logger.info("Running daily trading strategy")
    
    try:
        # Find the best model
        model_path, scaler_path = find_best_model(MODELS_DIR)
        
        if model_path and scaler_path:
            # Initialize trading engine
            engine = TradingEngine(
                model_path=model_path,
                scaler_path=scaler_path,
                symbols=['NIFTY 50']  # Default to Nifty 50 index
            )
            
            # Run the engine
            results = engine.run()
            
            logger.info("Daily trading strategy completed")
            return results
        else:
            logger.error("No model found for trading")
            return None
            
    except Exception as e:
        logger.error(f"Error running daily strategy: {str(e)}")
        return None


def start_scheduled_trading(run_time='09:20', symbols=None):
    """
    Start scheduled trading at the specified time on market days
    
    Args:
        run_time (str): Time to run the strategy (HH:MM)
        symbols (list): List of symbols to trade
    """
    logger.info(f"Starting scheduled trading at {run_time} daily")
    
    # Schedule the job
    schedule.every().day.at(run_time).do(run_daily_strategy)
    
    # Run the scheduler
    while True:
        try:
            # Check if it's a weekday
            if datetime.now().weekday() < 5:  # Monday to Friday
                schedule.run_pending()
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Scheduled trading stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduled trading: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying


def main():
    """
    Main function for trading
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Live trading for Nifty 50 trading algorithm')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (if not specified, will use the best model)')
    parser.add_argument('--scaler', '-s', type=str, default=None,
                        help='Path to scaler file')
    parser.add_argument('--symbols', type=str, nargs='+', default=['NIFTY 50'],
                        help='Symbols to trade')
    parser.add_argument('--schedule', action='store_true',
                        help='Start scheduled trading')
    parser.add_argument('--run-time', type=str, default='09:20',
                        help='Time to run the strategy (HH:MM)')
    parser.add_argument('--output', '-o', type=str, default=TRADING_RESULTS_DIR,
                        help='Output directory for trading results')
    
    args = parser.parse_args()
    
    logger.info("Starting trading")
    
    # If scheduled trading is requested
    if args.schedule:
        start_scheduled_trading(args.run_time, args.symbols)
        return
    
    # Otherwise, run the strategy once
    
    # Find model if not specified
    model_path = args.model
    scaler_path = args.scaler
    
    if not model_path or not scaler_path:
        model_path, scaler_path = find_best_model(MODELS_DIR)
    
    if model_path and scaler_path:
        # Initialize trading engine
        engine = TradingEngine(
            model_path=model_path,
            scaler_path=scaler_path,
            symbols=args.symbols,
            output_dir=args.output
        )
        
        # Run the engine
        results = engine.run()
        
        logger.info("Trading completed")
        return results
    else:
        logger.error("No model found for trading")
        return None


if __name__ == "__main__":
    main()
