"""
Functions for feature engineering for the Nifty 50 trading algorithm
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR,
    FEATURES_DATA_DIR,
    RAW_DATA_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_basic_price_features(df):
    """
    Add basic price-derived features
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    logger.info("Adding basic price features...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df_features.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add calendar-based features if the index is a datetime
    if isinstance(df_features.index, pd.DatetimeIndex):
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_month'] = df_features.index.day
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        df_features['year'] = df_features.index.year
        df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
        df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
    
    # Calculate returns
    df_features['Daily_Return'] = df_features['Close'].pct_change() * 100
    
    # Calculate log returns (better for statistical analysis)
    df_features['Log_Return'] = np.log(df_features['Close'] / df_features['Close'].shift(1)) * 100
    
    # Price change features
    df_features['Price_Change'] = df_features['Close'] - df_features['Open']
    df_features['Pct_Change'] = (df_features['Price_Change'] / df_features['Open']) * 100
    
    # Range features
    df_features['Daily_Range'] = df_features['High'] - df_features['Low']
    df_features['Daily_Range_Pct'] = (df_features['Daily_Range'] / df_features['Open']) * 100
    
    # Gap features
    df_features['Gap'] = df_features['Open'] - df_features['Close'].shift(1)
    df_features['Gap_Pct'] = (df_features['Gap'] / df_features['Close'].shift(1)) * 100
    
    # Position of close in daily range (0 to 1)
    df_features['Close_Position'] = (df_features['Close'] - df_features['Low']) / df_features['Daily_Range']
    
    # Price ratio features
    df_features['hl_ratio'] = df_features['High'] / df_features['Low']
    df_features['co_ratio'] = df_features['Close'] / df_features['Open']
    df_features['ho_ratio'] = df_features['High'] / df_features['Open']
    df_features['lo_ratio'] = df_features['Low'] / df_features['Open']
    df_features['hc_ratio'] = df_features['High'] / df_features['Close']
    df_features['lc_ratio'] = df_features['Low'] / df_features['Close']
    
    # Body size and wick analysis
    df_features['body_size'] = abs(df_features['Close'] - df_features['Open'])
    df_features['body_pct'] = df_features['body_size'] / df_features['Close'] * 100
    df_features['upper_wick'] = df_features.apply(lambda x: x['High'] - max(x['Open'], x['Close']), axis=1)
    df_features['upper_wick_pct'] = df_features['upper_wick'] / df_features['Close'] * 100
    df_features['lower_wick'] = df_features.apply(lambda x: min(x['Open'], x['Close']) - x['Low'], axis=1)
    df_features['lower_wick_pct'] = df_features['lower_wick'] / df_features['Close'] * 100
    
    # Volume features
    df_features['Volume_Change'] = df_features['Volume'].pct_change() * 100
    df_features['Volume_MA5'] = df_features['Volume'].rolling(window=5).mean()
    df_features['Relative_Volume'] = df_features['Volume'] / df_features['Volume_MA5']
    
    # Rolling statistics for Close prices
    windows = [5, 10, 20, 50, 200]
    
    for window in windows:
        # Moving averages
        df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()
        
        # Standard deviation (volatility)
        df_features[f'Std_{window}'] = df_features['Close'].rolling(window=window).std()
        
        # Price relative to MA
        df_features[f'Close_Rel_MA_{window}'] = (df_features['Close'] / df_features[f'MA_{window}'] - 1) * 100
        
        # Z-score of price relative to MA (how many standard deviations from mean)
        df_features[f'zscore_{window}d'] = (df_features['Close'] - df_features[f'MA_{window}']) / df_features[f'Std_{window}']
        
        # Return over rolling window
        df_features[f'Return_{window}d'] = (df_features['Close'] / df_features['Close'].shift(window) - 1) * 100
        
        # Skewness and Kurtosis of returns
        df_features[f'returns_skew_{window}d'] = df_features['Daily_Return'].rolling(window=window).skew()
        df_features[f'returns_kurt_{window}d'] = df_features['Daily_Return'].rolling(window=window).kurt()
        
        # Min/Max values over window
        df_features[f'Min_{window}d'] = df_features['Close'].rolling(window=window).min()
        df_features[f'Max_{window}d'] = df_features['Close'].rolling(window=window).max()
        
        # Distance from window high/low
        df_features[f'Pct_From_High_{window}d'] = (df_features['Close'] / df_features[f'Max_{window}d'] - 1) * 100
        df_features[f'Pct_From_Low_{window}d'] = (df_features['Close'] / df_features[f'Min_{window}d'] - 1) * 100
    
    # 52-week (252 trading days) high/low metrics
    df_features['52w_High'] = df_features['Close'].rolling(window=252).max()
    df_features['52w_Low'] = df_features['Close'].rolling(window=252).min()
    df_features['Pct_From_52w_High'] = (df_features['Close'] / df_features['52w_High'] - 1) * 100
    df_features['Pct_From_52w_Low'] = (df_features['Close'] / df_features['52w_Low'] - 1) * 100
    
    # Drop rows with NaN values
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    dropped_rows = initial_rows - len(df_features)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with NaN values after adding basic features")
    
    logger.info("Basic price features added")
    
    return df_features

def add_technical_indicators(df):
    """
    Add technical indicators using TA-Lib and pandas_ta
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    logger.info("Adding technical indicators...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df_features.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize arrays for TA-Lib functions
    open_price = df_features['Open'].values
    high_price = df_features['High'].values
    low_price = df_features['Low'].values
    close_price = df_features['Close'].values
    volume = df_features['Volume'].values
    
    try:
        # --- Trend Indicators ---
        
        # Moving Average Convergence Divergence (MACD)
        macd, macd_signal, macd_hist = ta.macd(close_price)
        df_features['MACD'] = macd
        df_features['MACD_Signal'] = macd_signal
        df_features['MACD_Hist'] = macd_hist
        
        # Average Directional Movement Index (ADX)
        df_features['ADX'] = ta.adx(high_price, low_price, close_price)
        
        # Parabolic SAR
        df_features['SAR'] = ta.sar(high_price, low_price)
        df_features['SAR_Ratio'] = df_features['Close'] / df_features['SAR']
        
        # Aroon Oscillator
        df_features['AroonOsc'] = ta.aroonosc(high_price, low_price)
        
        # --- Momentum Indicators ---
        
        # Relative Strength Index (RSI)
        df_features['RSI'] = ta.rsi(close_price)
        
        # Stochastic Oscillator
        df_features['SlowK'], df_features['SlowD'] = ta.stoch(high_price, low_price, close_price)
        
        # Commodity Channel Index (CCI)
        df_features['CCI'] = ta.cci(high_price, low_price, close_price)
        
        # Money Flow Index (MFI)
        df_features['MFI'] = ta.mfi(high_price, low_price, close_price, volume)
        
        # Williams %R
        df_features['WillR'] = ta.willr(high_price, low_price, close_price)
        
        # Rate of Change (ROC)
        df_features['ROC'] = ta.roc(close_price)
        df_features['ROC_5'] = ta.roc(close_price, length=5)
        df_features['ROC_21'] = ta.roc(close_price, length=21)
        
        # --- Volatility Indicators ---
        
        # Bollinger Bands
        bbands = ta.bbands(close_price)
        df_features['BBands_Upper'] = bbands['BBU_20_2.0']
        df_features['BBands_Middle'] = bbands['BBM_20_2.0']
        df_features['BBands_Lower'] = bbands['BBL_20_2.0']
        
        # BB Width and %B
        df_features['BBands_Width'] = (df_features['BBands_Upper'] - df_features['BBands_Lower']) / df_features['BBands_Middle']
        df_features['BBands_PctB'] = (df_features['Close'] - df_features['BBands_Lower']) / (df_features['BBands_Upper'] - df_features['BBands_Lower'])
        
        # Average True Range (ATR)
        df_features['ATR'] = ta.atr(high_price, low_price, close_price)
        df_features['ATR_Pct'] = df_features['ATR'] / df_features['Close'] * 100
        
        # --- Volume Indicators ---
        
        # On-Balance Volume (OBV)
        df_features['OBV'] = ta.obv(close_price, volume)
        
        # Chaikin A/D Line
        df_features['ADLINE'] = ta.ad(high_price, low_price, close_price, volume)
        
        # Chaikin Money Flow (CMF)
        df_features['CMF'] = ta.cmf(high_price, low_price, close_price, volume)
        
        # --- Additional Indicators using pandas_ta ---
        
        # Convert to pandas_ta format
        df_ta = df_features.copy()
        
        # Awesome Oscillator
        df_features['AO'] = ta.ao(df_ta['High'], df_ta['Low'])
        
        # Ease of Movement
        ema = ta.eom(df_ta['High'], df_ta['Low'], df_ta['Close'], df_ta['Volume'])
        df_features['EOM'] = ema['EOM_14']
        
        # Mass Index
        mass = ta.mass(df_ta['High'], df_ta['Low'])
        df_features['MASS'] = mass['MASS_25_9']
        
        # VWAP
        try:
            vwap = ta.vwap(df_ta['High'], df_ta['Low'], df_ta['Close'], df_ta['Volume'])
            df_features['VWAP'] = vwap
        except:
            logger.warning("Could not calculate VWAP - may require datetime index")
        
        # Elder's Force Index
        df_features['EFI'] = ta.efi(df_ta['Close'], df_ta['Volume'], length=13)
        
        # Trend Direction Force Index
        try:
            tdfi = ta.tdfi(df_ta['Close'], df_ta['High'], df_ta['Low'], df_ta['Open'])
            df_features['TDFI'] = tdfi['TDFI_8']
        except:
            logger.warning("Could not calculate TDFI - skipping")
        
        # Ichimoku Cloud
        try:
            ichimoku = ta.ichimoku(df_ta['High'], df_ta['Low'], df_ta['Close'],
                                 tenkan=9, kijun=26, senkou=52)
            
            # Extract components
            df_features['ICH_tenkan_sen'] = ichimoku['ITS_9']
            df_features['ICH_kijun_sen'] = ichimoku['IKS_26']
            df_features['ICH_senkou_span_a'] = ichimoku['ISA_9']
            df_features['ICH_senkou_span_b'] = ichimoku['ISB_26']
            
            # Ichimoku signals
            df_features['ICH_cloud_breakout_up'] = ((df_features['Close'] > df_features['ICH_senkou_span_a']) & 
                                                  (df_features['Close'] > df_features['ICH_senkou_span_b']) &
                                                  (df_features['Close'].shift(1) <= df_features['ICH_senkou_span_a'].shift(1) | 
                                                   df_features['Close'].shift(1) <= df_features['ICH_senkou_span_b'].shift(1))).astype(int)
                                                  
            df_features['ICH_cloud_breakout_down'] = ((df_features['Close'] < df_features['ICH_senkou_span_a']) & 
                                                    (df_features['Close'] < df_features['ICH_senkou_span_b']) &
                                                    (df_features['Close'].shift(1) >= df_features['ICH_senkou_span_a'].shift(1) | 
                                                     df_features['Close'].shift(1) >= df_features['ICH_senkou_span_b'].shift(1))).astype(int)
        except:
            logger.warning("Could not calculate Ichimoku Cloud - skipping")
        
        # Keltner Channels
        try:
            keltner = ta.kc(df_ta['High'], df_ta['Low'], df_ta['Close'], length=20, scalar=2)
            df_features['KC_Upper'] = keltner['KCUe_20_2']
            df_features['KC_Lower'] = keltner['KCLe_20_2']
            df_features['KC_Middle'] = keltner['KCMe_20_2']
            
            # Keltner Channel signals
            df_features['KC_Squeeze'] = ((df_features['BBands_Upper'] < df_features['KC_Upper']) & 
                                        (df_features['BBands_Lower'] > df_features['KC_Lower'])).astype(int)
        except:
            logger.warning("Could not calculate Keltner Channels - skipping")
        
        # Donchian Channels
        try:
            donchian = ta.donchian(df_ta['High'], df_ta['Low'], lower_length=20, upper_length=20)
            df_features['DC_Upper'] = donchian['DCU_20_20']
            df_features['DC_Lower'] = donchian['DCL_20_20']
            df_features['DC_Middle'] = donchian['DCM_20_20']
            
            # Donchian Channel breakout signals
            df_features['DC_Breakout_Up'] = ((df_features['Close'] > df_features['DC_Upper'].shift(1))).astype(int)
            df_features['DC_Breakout_Down'] = ((df_features['Close'] < df_features['DC_Lower'].shift(1))).astype(int)
        except:
            logger.warning("Could not calculate Donchian Channels - skipping")
        
        logger.info("Technical indicators added")
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        raise
    
    return df_features

def add_candlestick_patterns(df):
    """
    Add candlestick pattern recognition features
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added candlestick pattern features
    """
    logger.info("Adding candlestick pattern features...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in df_features.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize arrays for TA-Lib functions
    open_price = df_features['Open'].values
    high_price = df_features['High'].values
    low_price = df_features['Low'].values
    close_price = df_features['Close'].values
    
    try:
        # Single candlestick patterns
        df_features['CDL_DOJI'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="doji")
        df_features['CDL_HAMMER'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="hammer")
        df_features['CDL_INVHAMMER'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="inverted_hammer")
        df_features['CDL_MARUBOZU'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="marubozu")
        df_features['CDL_SHOOTING_STAR'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="shooting_star")
        df_features['CDL_SPINNING_TOP'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="spinning_top")
        df_features['CDL_HANGING_MAN'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="hanging_man")
        
        # Double candlestick patterns
        df_features['CDL_ENGULFING'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="engulfing")
        df_features['CDL_HARAMI'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="harami")
        df_features['CDL_PIERCING'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="piercing")
        df_features['CDL_TWEEZER_TOP'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="twisting_top")
        df_features['CDL_TWEEZER_BOTTOM'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="twisting_bottom")
        df_features['CDL_DARK_CLOUD_COVER'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="dark_cloud_cover")
        
        # Triple candlestick patterns
        df_features['CDL_MORNING_STAR'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="morning_star")
        df_features['CDL_EVENING_STAR'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="evening_star")
        df_features['CDL_3INSIDE_UP'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="three_inside_up")
        df_features['CDL_3OUTSIDE_UP'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="three_outside_up")
        df_features['CDL_3WHITESOLDIERS'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="three_white_soldiers")
        df_features['CDL_3BLACK_CROWS'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="three_black_crows")
        df_features['CDL_3LINE_STRIKE'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="three_line_strike")
        df_features['CDL_TASUKI_GAP'] = ta.cdl_pattern(open_price, high_price, low_price, close_price, name="tasuki_gap")
        
        # Normalize pattern signals to -1, 0, 1
        pattern_columns = [col for col in df_features.columns if col.startswith('CDL_')]
        for col in pattern_columns:
            df_features[col] = df_features[col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Separate bullish and bearish patterns for easier analysis
        bullish_patterns = [
            'CDL_HAMMER', 'CDL_INVHAMMER', 'CDL_PIERCING', 'CDL_MORNING_STAR',
            'CDL_3WHITESOLDIERS', 'CDL_3INSIDE_UP', 'CDL_3OUTSIDE_UP',
            'CDL_TWEEZER_BOTTOM'
        ]
        
        bearish_patterns = [
            'CDL_SHOOTING_STAR', 'CDL_HANGING_MAN', 'CDL_EVENING_STAR',
            'CDL_3BLACK_CROWS', 'CDL_DARK_CLOUD_COVER', 'CDL_TWEEZER_TOP'
        ]
        
        # The ENGULFING pattern can be bullish or bearish
        df_features['CDL_ENGULFING_BULLISH'] = (df_features['CDL_ENGULFING'] > 0).astype(int)
        df_features['CDL_ENGULFING_BEARISH'] = (df_features['CDL_ENGULFING'] < 0).astype(int)
        
        # Calculate pattern count features
        df_features['Bullish_Patterns'] = df_features[pattern_columns].apply(lambda x: (x > 0).sum(), axis=1)
        df_features['Bearish_Patterns'] = df_features[pattern_columns].apply(lambda x: (x < 0).sum(), axis=1)
        df_features['Pattern_Bias'] = df_features['Bullish_Patterns'] - df_features['Bearish_Patterns']
        
        # Specifically count known bullish/bearish patterns
        df_features['BULLISH_PATTERNS_COUNT'] = df_features[bullish_patterns].apply(
            lambda x: sum(1 for val in x if val > 0), axis=1
        )
        
        df_features['BEARISH_PATTERNS_COUNT'] = df_features[bearish_patterns].apply(
            lambda x: sum(1 for val in x if val < 0), axis=1
        )
        
        # Body and wick analysis - use rolling averages for detection
        # Long bodies
        body_size = abs(df_features['Close'] - df_features['Open'])
        body_size_avg = body_size.rolling(window=20).mean()
        df_features['LONG_BODY'] = (body_size > (1.5 * body_size_avg)).astype(int)
        df_features['SHORT_BODY'] = (body_size < (0.5 * body_size_avg)).astype(int)
        
        # Upper wick analysis
        upper_wick = df_features.apply(lambda x: x['High'] - max(x['Open'], x['Close']), axis=1)
        upper_wick_avg = upper_wick.rolling(window=20).mean()
        df_features['LONG_UPPER_WICK'] = (upper_wick > (1.5 * upper_wick_avg)).astype(int)
        
        # Lower wick analysis
        lower_wick = df_features.apply(lambda x: min(x['Open'], x['Close']) - x['Low'], axis=1)
        lower_wick_avg = lower_wick.rolling(window=20).mean()
        df_features['LONG_LOWER_WICK'] = (lower_wick > (1.5 * lower_wick_avg)).astype(int)
        
        # Bullish and bearish candles
        df_features['BULLISH_CANDLE'] = (df_features['Close'] > df_features['Open']).astype(int)
        df_features['BEARISH_CANDLE'] = (df_features['Close'] < df_features['Open']).astype(int)
        
        # Consecutive candle patterns
        df_features['THREE_WHITE_SOLDIERS_MANUAL'] = (
            (df_features['BULLISH_CANDLE'] == 1) & 
            (df_features['BULLISH_CANDLE'].shift(1) == 1) & 
            (df_features['BULLISH_CANDLE'].shift(2) == 1) &
            (df_features['Close'] > df_features['Close'].shift(1)) &
            (df_features['Close'].shift(1) > df_features['Close'].shift(2))
        ).astype(int)
        
        df_features['THREE_BLACK_CROWS_MANUAL'] = (
            (df_features['BEARISH_CANDLE'] == 1) & 
            (df_features['BEARISH_CANDLE'].shift(1) == 1) & 
            (df_features['BEARISH_CANDLE'].shift(2) == 1) &
            (df_features['Close'] < df_features['Close'].shift(1)) &
            (df_features['Close'].shift(1) < df_features['Close'].shift(2))
        ).astype(int)
        
        # Inside and outside bars
        df_features['INSIDE_BAR'] = (
            (df_features['High'] < df_features['High'].shift(1)) &
            (df_features['Low'] > df_features['Low'].shift(1))
        ).astype(int)
        
        df_features['OUTSIDE_BAR'] = (
            (df_features['High'] > df_features['High'].shift(1)) &
            (df_features['Low'] < df_features['Low'].shift(1))
        ).astype(int)
        
        # Current candlestick vs average of previous N candles
        n = 5
        df_features['ABOVE_AVG_RANGE'] = (df_features['Daily_Range'] > df_features['Daily_Range'].rolling(window=n).mean()).astype(int)
        
        # Pattern consensus
        df_features['STRONG_BULLISH_SIGNAL'] = (
            ((df_features['Bullish_Patterns'] >= 2) | (df_features['BULLISH_PATTERNS_COUNT'] >= 2)) &
            (df_features['Bearish_Patterns'] == 0)
        ).astype(int)
        
        df_features['STRONG_BEARISH_SIGNAL'] = (
            ((df_features['Bearish_Patterns'] >= 2) | (df_features['BEARISH_PATTERNS_COUNT'] >= 2)) &
            (df_features['Bullish_Patterns'] == 0)
        ).astype(int)
        
        logger.info("Candlestick pattern features added")
        
    except Exception as e:
        logger.error(f"Error adding candlestick pattern features: {str(e)}")
        raise
    
    return df_features

def add_custom_features(df):
    """
    Add custom trading features
    
    Args:
        df (pd.DataFrame): DataFrame with basic features and indicators
        
    Returns:
        pd.DataFrame: DataFrame with added custom features
    """
    logger.info("Adding custom features...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    try:
        # RSI divergence detection
        if 'RSI' in df_features.columns and 'Close' in df_features.columns:
            # Get previous n values for price and RSI
            n = 5
            
            # Price higher highs/lower lows
            df_features['Price_Higher_High'] = df_features['Close'] > df_features['Close'].rolling(window=n).max().shift(1)
            df_features['Price_Lower_Low'] = df_features['Close'] < df_features['Close'].rolling(window=n).min().shift(1)
            
            # RSI higher highs/lower lows
            df_features['RSI_Higher_High'] = df_features['RSI'] > df_features['RSI'].rolling(window=n).max().shift(1)
            df_features['RSI_Lower_Low'] = df_features['RSI'] < df_features['RSI'].rolling(window=n).min().shift(1)
            
            # Divergence signals
            df_features['Bullish_Divergence'] = (df_features['Price_Lower_Low'] & ~df_features['RSI_Lower_Low']).astype(int)
            df_features['Bearish_Divergence'] = (df_features['Price_Higher_High'] & ~df_features['RSI_Higher_High']).astype(int)
        
        # Support and resistance levels
        if 'High' in df_features.columns and 'Low' in df_features.columns:
            # Simple approach using recent highs and lows
            n_resistance = 20
            n_support = 20
            
            df_features['Resistance'] = df_features['High'].rolling(window=n_resistance).max()
            df_features['Support'] = df_features['Low'].rolling(window=n_support).min()
            
            # Distance to support and resistance
            df_features['Dist_To_Resistance'] = (df_features['Resistance'] - df_features['Close']) / df_features['Close'] * 100
            df_features['Dist_To_Support'] = (df_features['Close'] - df_features['Support']) / df_features['Close'] * 100
            
            # Support/Resistance breakthrough
            df_features['Resistance_Break'] = (df_features['Close'] > df_features['Resistance'].shift(1)).astype(int)
            df_features['Support_Break'] = (df_features['Close'] < df_features['Support'].shift(1)).astype(int)
        
        # Volatility Bands
        if 'Close' in df_features.columns and 'ATR' in df_features.columns:
            atr_multiplier = 2.0
            df_features['Upper_Vol_Band'] = df_features['Close'] + atr_multiplier * df_features['ATR']
            df_features['Lower_Vol_Band'] = df_features['Close'] - atr_multiplier * df_features['ATR']
            df_features['In_Vol_Range'] = ((df_features['Close'] > df_features['Lower_Vol_Band']) & 
                                          (df_features['Close'] < df_features['Upper_Vol_Band'])).astype(int)
        
        # MACD-based features
        if all(col in df_features.columns for col in ['MACD', 'MACD_Signal']):
            df_features['MACD_Crossover'] = ((df_features['MACD'] > df_features['MACD_Signal']) & 
                                            (df_features['MACD'].shift(1) <= df_features['MACD_Signal'].shift(1))).astype(int)
            df_features['MACD_Crossunder'] = ((df_features['MACD'] < df_features['MACD_Signal']) & 
                                             (df_features['MACD'].shift(1) >= df_features['MACD_Signal'].shift(1))).astype(int)
        
        # Volume Trend Detection
        if 'Volume' in df_features.columns:
            volume_ma_period = 20
            df_features['Vol_SMA'] = df_features['Volume'].rolling(window=volume_ma_period).mean()
            df_features['Vol_Trend'] = (df_features['Volume'] / df_features['Vol_SMA'] - 1) * 100
            df_features['Rising_Vol'] = (df_features['Vol_Trend'] > 20).astype(int)
            df_features['Falling_Vol'] = (df_features['Vol_Trend'] < -20).astype(int)
        
        # Price Momentum Features
        if 'Close' in df_features.columns:
            # Rate of Change (RoC) over different periods
            for period in [3, 5, 10]:
                df_features[f'RoC_{period}'] = (df_features['Close'] / df_features['Close'].shift(period) - 1) * 100
            
            # Acceleration of price movement
            df_features['Price_Accel'] = df_features['RoC_3'] - df_features['RoC_3'].shift(1)
        
        # Gap Analysis
        if all(col in df_features.columns for col in ['Open', 'Close', 'High', 'Low']):
            # Gap up and gap down
            df_features['Gap_Up'] = (df_features['Open'] > df_features['Close'].shift(1)).astype(int)
            df_features['Gap_Down'] = (df_features['Open'] < df_features['Close'].shift(1)).astype(int)
            
            # Gap size
            df_features['Gap_Size'] = (df_features['Open'] - df_features['Close'].shift(1)) / df_features['Close'].shift(1) * 100
            
            # Gap fill
            df_features['Gap_Fill'] = ((df_features['Gap_Up'] & (df_features['Low'] <= df_features['Close'].shift(1))) | 
                                      (df_features['Gap_Down'] & (df_features['High'] >= df_features['Close'].shift(1)))).astype(int)
        
        # Market Regime Detection (using simple 200 day MA)
        if 'Close' in df_features.columns and 'MA_200' in df_features.columns:
            df_features['Bull_Market'] = (df_features['Close'] > df_features['MA_200']).astype(int)
            df_features['Bear_Market'] = (df_features['Close'] < df_features['MA_200']).astype(int)
            
            # Golden Cross and Death Cross
            df_features['Golden_Cross'] = ((df_features['MA_50'] > df_features['MA_200']) & 
                                          (df_features['MA_50'].shift(1) <= df_features['MA_200'].shift(1))).astype(int)
            df_features['Death_Cross'] = ((df_features['MA_50'] < df_features['MA_200']) & 
                                         (df_features['MA_50'].shift(1) >= df_features['MA_200'].shift(1))).astype(int)
        
        # Mean Reversion Potential
        if 'Close' in df_features.columns and 'BBands_PctB' in df_features.columns:
            df_features['Overbought'] = (df_features['BBands_PctB'] > 0.8).astype(int)
            df_features['Oversold'] = (df_features['BBands_PctB'] < 0.2).astype(int)
            
            # RSI confirmation of overbought/oversold
            if 'RSI' in df_features.columns:
                df_features['Confirmed_Overbought'] = ((df_features['Overbought'] == 1) & (df_features['RSI'] > 70)).astype(int)
                df_features['Confirmed_Oversold'] = ((df_features['Oversold'] == 1) & (df_features['RSI'] < 30)).astype(int)
        
        # Market Breadth Features (if multiple stocks data available)
        # This is a placeholder - in a real implementation, these would be calculated 
        # using data from all 50 Nifty constituents
        if 'Symbol' in df_features.columns:
            # These would be calculated from a dataset containing all symbols
            # For now, we'll leave them as NaN placeholders
            df_features['Advance_Decline_Ratio'] = np.nan
            df_features['Percentage_Above_MA50'] = np.nan
            df_features['New_Highs_Count'] = np.nan
            df_features['New_Lows_Count'] = np.nan
        
        # PCR_OI_zscore and PCR_OI_extreme_low
        if 'PCR_OI' in df_features.columns:
            pcr_mean = df_features['PCR_OI'].mean()
            pcr_std = df_features['PCR_OI'].std()
            df_features['PCR_OI_zscore'] = (df_features['PCR_OI'] - pcr_mean) / pcr_std
            df_features['PCR_OI_extreme_low'] = (df_features['PCR_OI_zscore'] < -2).astype(int)
        
        # Drop rows with NaN values (optional, depending on requirements)
        # df_features = df_features.dropna()
        
        logger.info("Custom features added successfully")
        
    except Exception as e:
        logger.error(f"Error adding custom features: {str(e)}")
        raise
    
    return df_features

def add_options_metrics_features(df):
    """
    Add derived features from options metrics
    
    Args:
        df (pd.DataFrame): DataFrame with price and options metrics data
        
    Returns:
        pd.DataFrame: DataFrame with options metrics features added
    """
    logger.info("Adding options metrics features...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Check if options metrics columns exist
    options_columns = ['PCR_OI', 'PCR_Volume', 'MaxPain', 'IV_Skew', 
                      'TotalCallOI', 'TotalPutOI', 'TotalCallVolume', 'TotalPutVolume']
    
    missing_columns = [col for col in options_columns if col not in df_features.columns]
    
    if len(missing_columns) == len(options_columns):
        logger.warning("No options metrics columns found, skipping options features")
        return df_features
    
    try:
        # Calculate moving averages of PCR
        if 'PCR_OI' in df_features.columns:
            df_features['PCR_OI_MA5'] = df_features['PCR_OI'].rolling(window=5).mean()
            df_features['PCR_OI_MA10'] = df_features['PCR_OI'].rolling(window=10).mean()
            df_features['PCR_OI_change'] = df_features['PCR_OI'].pct_change()
            
            # Extreme PCR values
            pcr_mean = df_features['PCR_OI'].rolling(window=20).mean()
            pcr_std = df_features['PCR_OI'].rolling(window=20).std()
            df_features['PCR_OI_zscore'] = (df_features['PCR_OI'] - pcr_mean) / pcr_std
            df_features['PCR_OI_extreme_low'] = (df_features['PCR_OI_zscore'] < -2).astype(int)
            df_features['PCR_OI_extreme_high'] = (df_features['PCR_OI_zscore'] > 2).astype(int)
        
        if 'PCR_Volume' in df_features.columns:
            df_features['PCR_Volume_MA5'] = df_features['PCR_Volume'].rolling(window=5).mean()
            df_features['PCR_Volume_MA10'] = df_features['PCR_Volume'].rolling(window=10).mean()
            df_features['PCR_Volume_change'] = df_features['PCR_Volume'].pct_change()
        
        # Distance from Max Pain to current price
        if 'MaxPain' in df_features.columns and 'Close' in df_features.columns:
            df_features['MaxPain_distance'] = (df_features['Close'] - df_features['MaxPain']) / df_features['MaxPain'] * 100
            df_features['MaxPain_distance_zscore'] = (df_features['MaxPain_distance'] - 
                                                   df_features['MaxPain_distance'].rolling(window=20).mean()) / \
                                                   df_features['MaxPain_distance'].rolling(window=20).std()
        
        # IV Skew features
        if 'IV_Skew' in df_features.columns:
            df_features['IV_Skew_MA5'] = df_features['IV_Skew'].rolling(window=5).mean()
            df_features['IV_Skew_change'] = df_features['IV_Skew'].diff()
            df_features['IV_Skew_zscore'] = (df_features['IV_Skew'] - 
                                          df_features['IV_Skew'].rolling(window=20).mean()) / \
                                          df_features['IV_Skew'].rolling(window=20).std()
        
        # Open Interest and Volume trends
        oi_volume_columns = ['TotalCallOI', 'TotalPutOI', 'TotalCallVolume', 'TotalPutVolume']
        existing_columns = [col for col in oi_volume_columns if col in df_features.columns]
        
        for col in existing_columns:
            # Calculate change
            df_features[f'{col}_change'] = df_features[col].pct_change()
            
            # Calculate 5-day rate of change
            df_features[f'{col}_ROC5'] = df_features[col].pct_change(periods=5)
            
            # Normalize by 20-day average
            df_features[f'{col}_normalized'] = df_features[col] / df_features[col].rolling(window=20).mean()
        
        # Call-Put OI Difference
        if 'TotalCallOI' in df_features.columns and 'TotalPutOI' in df_features.columns:
            df_features['CallPut_OI_diff'] = df_features['TotalCallOI'] - df_features['TotalPutOI']
            df_features['CallPut_OI_ratio'] = df_features['TotalCallOI'] / df_features['TotalPutOI']
        
        # Call-Put Volume Difference
        if 'TotalCallVolume' in df_features.columns and 'TotalPutVolume' in df_features.columns:
            df_features['CallPut_Vol_diff'] = df_features['TotalCallVolume'] - df_features['TotalPutVolume']
            df_features['CallPut_Vol_ratio'] = df_features['TotalCallVolume'] / df_features['TotalPutVolume']
        
        # Option sentiment indicators
        if 'PCR_OI' in df_features.columns and 'PCR_Volume' in df_features.columns:
            # Combined PCR score (average of OI and Volume PCR)
            df_features['PCR_Combined'] = (df_features['PCR_OI'] + df_features['PCR_Volume']) / 2
            
            # Option sentiment classification
            df_features['Bullish_Option_Sentiment'] = (df_features['PCR_OI'] > 1.5).astype(int)
            df_features['Bearish_Option_Sentiment'] = (df_features['PCR_OI'] < 0.7).astype(int)
            df_features['Neutral_Option_Sentiment'] = ((df_features['PCR_OI'] >= 0.7) & 
                                                    (df_features['PCR_OI'] <= 1.5)).astype(int)
        
        logger.info("Options metrics features added successfully")
        
    except Exception as e:
        logger.error(f"Error adding options metrics features: {str(e)}")
        # Continue with the features we have
    
    return df_features

def process_file(file_path, output_dir=FEATURES_DATA_DIR):
    """
    Process a single file with OHLCV data to add features
    
    Args:
        file_path (str): Path to the CSV file with OHLCV data
        output_dir (str): Directory to save the processed file
        
    Returns:
        str: Path to the processed file
    """
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Check if data is suitable (has required columns)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # If columns are capitalized, convert to lowercase
        if any(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            df.columns = [col.lower() for col in df.columns]
        
        # Verify all required columns now exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Rename columns to match TA-Lib expectations (capitalized)
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Add features
        df_with_features = add_basic_price_features(df)
        df_with_features = add_technical_indicators(df_with_features)
        df_with_features = add_candlestick_patterns(df_with_features)
        df_with_features = add_custom_features(df_with_features)
        
        # Add options metrics features if they exist
        options_columns = ['PCR_OI', 'PCR_Volume', 'MaxPain', 'IV_Skew', 
                          'TotalCallOI', 'TotalPutOI', 'TotalCallVolume', 'TotalPutVolume']
        if any(col in df_with_features.columns for col in options_columns):
            df_with_features = add_options_metrics_features(df_with_features)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"features_{filename}")
        
        # Save processed data
        df_with_features.to_csv(output_path)
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def process_directory(input_dir=RAW_DATA_DIR, output_dir=FEATURES_DATA_DIR, pattern='*.csv'):
    """
    Process all CSV files in a directory to add features
    
    Args:
        input_dir (str): Directory containing CSV files with OHLCV data
        output_dir (str): Directory to save the processed files
        pattern (str): Glob pattern to match files
        
    Returns:
        list: Paths to the processed files
    """
    import glob
    
    try:
        logger.info(f"Processing files in directory: {input_dir}")
        
        # Find all CSV files in the directory and its subdirectories
        all_files = []
        
        # Check if this is a directory that has indices and stocks subdirectories
        indices_dir = os.path.join(input_dir, 'indices')
        stocks_dir = os.path.join(input_dir, 'stocks')
        
        if os.path.isdir(indices_dir):
            index_files = glob.glob(os.path.join(indices_dir, pattern))
            all_files.extend(index_files)
            
            # Create corresponding output directory
            os.makedirs(os.path.join(output_dir, 'indices'), exist_ok=True)
        
        if os.path.isdir(stocks_dir):
            stock_files = glob.glob(os.path.join(stocks_dir, pattern))
            all_files.extend(stock_files)
            
            # Create corresponding output directory
            os.makedirs(os.path.join(output_dir, 'stocks'), exist_ok=True)
        
        # If no indices/stocks subdirectories, just search in the input directory
        if not all_files:
            all_files = glob.glob(os.path.join(input_dir, pattern))
        
        # Process each file
        processed_files = []
        
        for file_path in all_files:
            # Determine output directory (preserve subdirectory structure)
            rel_path = os.path.relpath(file_path, input_dir)
            rel_dir = os.path.dirname(rel_path)
            file_output_dir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
            
            # Process file
            processed_file = process_file(file_path, file_output_dir)
            
            if processed_file:
                processed_files.append(processed_file)
        
        logger.info(f"Processed {len(processed_files)} files")
        
        return processed_files
    
    except Exception as e:
        logger.error(f"Error processing directory {input_dir}: {str(e)}")
        return []

def main():
    """
    Main function to process data and create features.
    Can be run as a script to process all data files.
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate features for Nifty 50 trading algorithm')
    parser.add_argument('--input', '-i', type=str, default=RAW_DATA_DIR, 
                        help='Input directory containing raw OHLCV data')
    parser.add_argument('--output', '-o', type=str, default=FEATURES_DATA_DIR,
                        help='Output directory for feature data')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='Process a single file instead of a directory')
    
    args = parser.parse_args()
    
    logger.info("Starting feature engineering process")
    
    # Process a single file or an entire directory
    if args.file:
        if os.path.isfile(args.file):
            processed_file = process_file(args.file, args.output)
            
            if processed_file:
                logger.info(f"Successfully processed file: {processed_file}")
            else:
                logger.error("Failed to process file")
        else:
            logger.error(f"File not found: {args.file}")
    else:
        processed_files = process_directory(args.input, args.output)
        
        if processed_files:
            logger.info(f"Successfully processed {len(processed_files)} files")
        else:
            logger.warning("No files were processed")
    
    logger.info("Feature engineering process complete")

if __name__ == "__main__":
    main()