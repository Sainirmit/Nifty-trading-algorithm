"""
Machine learning model functionality for Nifty 50 trading algorithm
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURES_DATA_DIR,
    MODELS_DIR,
    MODEL_PARAMS,
    TRAIN_TEST_SPLIT,
    VALIDATION_SPLIT,
    TARGET_VARIABLE,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating machine learning models
    """
    
    def __init__(self, data=None, target_variable=TARGET_VARIABLE, prediction_horizon=PREDICTION_HORIZON,
                 model_type='random_forest', model_params=None, output_dir=MODELS_DIR):
        """
        Initialize the model trainer
        
        Args:
            data (pd.DataFrame): Features DataFrame
            target_variable (str): Name of the target variable column
            prediction_horizon (int): Number of days ahead to predict
            model_type (str): Type of model to train ('random_forest', 'xgboost', 'lgbm', 'lstm')
            model_params (dict): Parameters for the model
            output_dir (str): Directory to save trained models
        """
        self.data = data
        self.target_variable = target_variable
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.model_params = model_params if model_params else MODEL_PARAMS.get(model_type, {})
        self.output_dir = output_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = None
        self.training_history = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized {model_type} model trainer")
        
    def prepare_data(self, classification=True, sequence=False):
        """
        Prepare data for training
        
        Args:
            classification (bool): Whether to prepare for classification or regression
            sequence (bool): Whether to prepare sequential data for LSTM
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training")
        
        if self.data is None:
            raise ValueError("No data provided. Please set data before preparing.")
        
        # Create target variable (future returns)
        self.data[f'future_return_{self.prediction_horizon}d'] = self.data[self.target_variable].shift(-self.prediction_horizon)
        
        # For classification, convert to binary target (1 if positive return, 0 otherwise)
        if classification:
            self.data['target'] = (self.data[f'future_return_{self.prediction_horizon}d'] > 0).astype(int)
        else:
            self.data['target'] = self.data[f'future_return_{self.prediction_horizon}d']
        
        # Drop rows with NaN target
        self.data = self.data.dropna(subset=['target'])
        
        # Select features
        exclude_columns = [
            'target', 'future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d',
            'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Symbol', 'day_of_week', 'day_of_month',
            'month', 'quarter', 'year', 'is_month_start', 'is_month_end', 'index', 'datetime', 'date'
        ]
        
        # Remove target variable and any direct future information
        features = [col for col in self.data.columns if col not in exclude_columns and 
                   'future' not in col.lower()]
        
        # Remove columns where all values are NaN
        features = [f for f in features if not self.data[f].isna().all()]
        
        # Fill remaining NaN values with forward or backward fill
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Split into training and testing sets (time-based split)
        train_size = int(len(self.data) * TRAIN_TEST_SPLIT)
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]
        
        X_train = train_data[features]
        y_train = train_data['target']
        X_test = test_data[features]
        y_test = test_data['target']
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save feature names for later
        self.feature_names = features
        
        # Prepare sequence data for LSTM
        if sequence:
            X_train_seq = []
            y_train_seq = []
            X_test_seq = []
            y_test_seq = []
            
            # Create sequences for training data
            for i in range(SEQUENCE_LENGTH, len(X_train_scaled)):
                X_train_seq.append(X_train_scaled[i-SEQUENCE_LENGTH:i])
                y_train_seq.append(y_train.iloc[i])
            
            # Create sequences for testing data
            for i in range(SEQUENCE_LENGTH, len(X_test_scaled)):
                X_test_seq.append(X_test_scaled[i-SEQUENCE_LENGTH:i])
                y_test_seq.append(y_test.iloc[i])
            
            X_train_scaled = np.array(X_train_seq)
            y_train = np.array(y_train_seq)
            X_test_scaled = np.array(X_test_seq)
            y_test = np.array(y_test_seq)
        
        logger.info(f"Data prepared: X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features
    
    def train_random_forest(self, X_train, y_train, classification=True):
        """
        Train a Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            classification (bool): Whether to train a classifier or regressor
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest model")
        
        # Get model parameters from config or use defaults
        params = self.model_params.copy()
        
        # Choose classifier or regressor
        if classification:
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, classification=True):
        """
        Train an XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            classification (bool): Whether to train a classifier or regressor
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost model")
        
        # Get model parameters from config or use defaults
        params = self.model_params.copy()
        
        # Set validation data if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Choose classifier or regressor
        if classification:
            model = xgb.XGBClassifier(**params)
            objective = 'binary:logistic'
        else:
            model = xgb.XGBRegressor(**params)
            objective = 'reg:squarederror'
        
        # Update objective in parameters
        if 'objective' not in params:
            model.set_params(objective=objective)
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='logloss' if classification else 'rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, classification=True):
        """
        Train a LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            classification (bool): Whether to train a classifier or regressor
            
        Returns:
            Trained model
        """
        logger.info("Training LightGBM model")
        
        # Get model parameters from config or use defaults
        params = self.model_params.copy()
        
        # Set validation data if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Choose classifier or regressor
        if classification:
            model = lgb.LGBMClassifier(**params)
            objective = 'binary'
        else:
            model = lgb.LGBMRegressor(**params)
            objective = 'regression'
        
        # Update objective in parameters
        if 'objective' not in params:
            model.set_params(objective=objective)
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='binary_logloss' if classification else 'rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return model
    
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None, classification=True):
        """
        Train an LSTM model
        
        Args:
            X_train: Training features (3D array for LSTM)
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            classification (bool): Whether to train a classifier or regressor
            
        Returns:
            Trained model
        """
        logger.info("Training LSTM model")
        
        # Get model parameters from config or use defaults
        params = self.model_params.copy()
        units = params.get('units', [64, 32])
        dropout = params.get('dropout', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        epochs = params.get('epochs', 50)
        
        # Create validation set if not provided
        if X_val is None or y_val is None:
            val_split = VALIDATION_SPLIT
        else:
            val_split = 0.0  # We'll use the provided validation set
        
        # Build the LSTM model
        model = Sequential()
        
        # Add LSTM layers
        if len(units) > 1:
            model.add(LSTM(units[0], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(dropout))
            
            for i in range(1, len(units) - 1):
                model.add(LSTM(units[i], return_sequences=True))
                model.add(Dropout(dropout))
            
            model.add(LSTM(units[-1]))
            model.add(Dropout(dropout))
        else:
            model.add(LSTM(units[0], input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(dropout))
        
        # Output layer
        if classification:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'lstm_checkpoint.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                validation_split=val_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Store training history
        self.training_history = history.history
        
        return model
    
    def train_model(self, classification=True):
        """
        Train a model based on the selected model type
        
        Args:
            classification (bool): Whether to train a classifier or regressor
            
        Returns:
            Trained model
        """
        # Check if data is provided
        if self.data is None:
            raise ValueError("No data provided. Please set data before training.")
        
        # Prepare the data
        is_sequence = self.model_type.lower() == 'lstm'
        X_train, X_test, y_train, y_test, features = self.prepare_data(classification, is_sequence)
        
        # Create validation set for models that support it
        if self.model_type.lower() in ['xgboost', 'lgbm', 'lstm']:
            if is_sequence:
                val_size = int(len(X_train) * VALIDATION_SPLIT)
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]
            else:
                X_train_rs, X_val, y_train_rs, y_val = train_test_split(
                    X_train, y_train, test_size=VALIDATION_SPLIT, random_state=42
                )
                X_train, y_train = X_train_rs, y_train_rs
        else:
            X_val, y_val = None, None
        
        # Train the model based on model type
        if self.model_type.lower() == 'random_forest':
            self.model = self.train_random_forest(X_train, y_train, classification)
        elif self.model_type.lower() == 'xgboost':
            self.model = self.train_xgboost(X_train, y_train, X_val, y_val, classification)
        elif self.model_type.lower() == 'lgbm':
            self.model = self.train_lightgbm(X_train, y_train, X_val, y_val, classification)
        elif self.model_type.lower() == 'lstm':
            self.model = self.train_lstm(X_train, y_train, X_val, y_val, classification)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Evaluate the model
        self.evaluate_model(X_test, y_test, classification)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, classification=True):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test target
            classification (bool): Whether it's a classification or regression problem
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model")
        
        if self.model is None:
            raise ValueError("No model trained yet. Please train a model first.")
        
        # Make predictions
        if self.model_type.lower() == 'lstm':
            y_pred = self.model.predict(X_test)
            if classification:
                y_pred = (y_pred > 0.5).astype(int)
        else:
            if classification:
                y_pred = self.model.predict(X_test)
            else:
                y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        if classification:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            self.metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            logger.info(f"Classification metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            self.metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2)
            }
            
            logger.info(f"Regression metrics: MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return self.metrics
    
    def save_model(self, suffix=None):
        """
        Save the trained model, scaler, and metrics
        
        Args:
            suffix (str): Optional suffix for the model filename
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model trained yet. Please train a model first.")
        
        # Create a timestamp for the model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a filename
        if suffix:
            filename = f"{self.model_type}_{suffix}_{timestamp}"
        else:
            filename = f"{self.model_type}_{timestamp}"
        
        # Save the model
        model_path = os.path.join(self.output_dir, f"{filename}.joblib")
        scaler_path = os.path.join(self.output_dir, f"{filename}_scaler.joblib")
        metrics_path = os.path.join(self.output_dir, f"{filename}_metrics.json")
        
        # Different saving method for LSTM models
        if self.model_type.lower() == 'lstm':
            model_path = os.path.join(self.output_dir, f"{filename}.h5")
            self.model.save(model_path)
        else:
            joblib.dump(self.model, model_path)
        
        # Save the scaler
        joblib.dump(self.scaler, scaler_path)
        
        # Save the metrics and feature importance
        metrics_data = {
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'timestamp': timestamp,
            'parameters': self.model_params
        }
        
        # Add training history for LSTM
        if self.training_history:
            metrics_data['training_history'] = {k: [float(val) for val in v] for k, v in self.training_history.items()}
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        return model_path


class ModelPredictor:
    """
    Class for making predictions with trained models
    """
    
    def __init__(self, model_path=None, scaler_path=None, model_type=None):
        """
        Initialize the model predictor
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
            model_type (str): Type of model ('random_forest', 'xgboost', 'lgbm', 'lstm')
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_type = model_type
        self.model = None
        self.scaler = None
        
        # Load model and scaler if paths are provided
        if model_path:
            self.load_model(model_path)
        
        if scaler_path:
            self.load_scaler(scaler_path)
        
        logger.info("Initialized model predictor")
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Determine model type from file extension if not provided
        if not self.model_type:
            if model_path.endswith('.h5'):
                self.model_type = 'lstm'
            else:
                self.model_type = 'unknown'
        
        # Load the model
        try:
            if self.model_type.lower() == 'lstm':
                self.model = load_model(model_path)
            else:
                self.model = joblib.load(model_path)
            
            self.model_path = model_path
            logger.info(f"Model loaded successfully")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_scaler(self, scaler_path):
        """
        Load a saved scaler
        
        Args:
            scaler_path (str): Path to the saved scaler
            
        Returns:
            Loaded scaler
        """
        logger.info(f"Loading scaler from {scaler_path}")
        
        try:
            self.scaler = joblib.load(scaler_path)
            self.scaler_path = scaler_path
            logger.info(f"Scaler loaded successfully")
            
            return self.scaler
        
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def prepare_features(self, data, sequence=False):
        """
        Prepare features for prediction
        
        Args:
            data (pd.DataFrame): Features DataFrame
            sequence (bool): Whether to prepare sequential data for LSTM
            
        Returns:
            Prepared features
        """
        logger.info("Preparing features for prediction")
        
        if self.scaler is None:
            raise ValueError("No scaler loaded. Please load a scaler before preparing features.")
        
        # Identify feature columns
        exclude_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Symbol', 'day_of_week', 'day_of_month',
            'month', 'quarter', 'year', 'is_month_start', 'is_month_end', 'index', 'datetime', 'date',
            'target', 'future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d'
        ]
        
        features = [col for col in data.columns if col not in exclude_columns and 
                   'future' not in col.lower()]
        
        # Fill NaN values
        data_filled = data[features].fillna(method='ffill').fillna(method='bfill')
        
        # Scale the features
        X_scaled = self.scaler.transform(data_filled)
        
        # Prepare sequence data for LSTM
        if sequence:
            X_seq = []
            seq_length = SEQUENCE_LENGTH
            
            if len(X_scaled) < seq_length:
                raise ValueError(f"Not enough data points for LSTM. Need at least {seq_length} data points.")
            
            # Create sequence
            X_seq.append(X_scaled[-seq_length:])
            X_scaled = np.array(X_seq)
        
        return X_scaled
    
    def predict(self, data, classification=True):
        """
        Make predictions with the loaded model
        
        Args:
            data (pd.DataFrame): Features DataFrame
            classification (bool): Whether it's a classification or regression problem
            
        Returns:
            np.array: Predictions
        """
        logger.info("Making predictions")
        
        if self.model is None:
            raise ValueError("No model loaded. Please load a model before predicting.")
        
        # Prepare features
        is_sequence = self.model_type.lower() == 'lstm'
        X = self.prepare_features(data, is_sequence)
        
        # Make predictions
        if is_sequence:
            y_pred = self.model.predict(X)
            if classification:
                y_pred = (y_pred > 0.5).astype(int)
        else:
            y_pred = self.model.predict(X)
        
        return y_pred

    def predict_proba(self, data):
        """
        Get probability predictions for classification models
        
        Args:
            data (pd.DataFrame): Features DataFrame
            
        Returns:
            np.array: Probability predictions
        """
        logger.info("Making probability predictions")
        
        if self.model is None:
            raise ValueError("No model loaded. Please load a model before predicting.")
        
        if self.model_type.lower() == 'lstm':
            # LSTM models return probabilities directly
            is_sequence = True
            X = self.prepare_features(data, is_sequence)
            return self.model.predict(X)
        else:
            # For other models, use predict_proba if available
            try:
                is_sequence = False
                X = self.prepare_features(data, is_sequence)
                return self.model.predict_proba(X)
            except AttributeError:
                logger.error("Model does not support probability predictions")
                raise ValueError("Model does not support probability predictions")


def find_best_model(input_dir=MODELS_DIR, metric='accuracy'):
    """
    Find the best saved model based on a metric
    
    Args:
        input_dir (str): Directory with saved models
        metric (str): Metric to use for comparison
        
    Returns:
        tuple: Path to the best model and its scaler
    """
    logger.info(f"Finding best model based on {metric}")
    
    # Get all metrics files
    metrics_files = [f for f in os.listdir(input_dir) if f.endswith('_metrics.json')]
    
    if not metrics_files:
        logger.warning("No model metrics files found")
        return None, None
    
    best_score = -float('inf')
    best_model_path = None
    best_scaler_path = None
    
    for metrics_file in metrics_files:
        # Load metrics
        with open(os.path.join(input_dir, metrics_file), 'r') as f:
            metrics_data = json.load(f)
        
        # Check if the metric exists
        if 'metrics' in metrics_data and metric in metrics_data['metrics']:
            score = metrics_data['metrics'][metric]
            
            # For MSE and RMSE, lower is better
            if metric in ['mse', 'rmse']:
                score = -score
            
            if score > best_score:
                best_score = score
                
                # Get corresponding model and scaler paths
                file_prefix = metrics_file.replace('_metrics.json', '')
                model_path = None
                
                # Check for model file (can be .joblib or .h5)
                joblib_path = os.path.join(input_dir, f"{file_prefix}.joblib")
                h5_path = os.path.join(input_dir, f"{file_prefix}.h5")
                
                if os.path.exists(joblib_path):
                    model_path = joblib_path
                elif os.path.exists(h5_path):
                    model_path = h5_path
                
                scaler_path = os.path.join(input_dir, f"{file_prefix}_scaler.joblib")
                
                if model_path and os.path.exists(model_path) and os.path.exists(scaler_path):
                    best_model_path = model_path
                    best_scaler_path = scaler_path
    
    if best_model_path:
        logger.info(f"Best model found: {best_model_path} with {metric} = {best_score}")
    else:
        logger.warning(f"No models found with metric {metric}")
    
    return best_model_path, best_scaler_path


def main():
    """
    Main function for model training and evaluation
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate models for Nifty 50 trading algorithm')
    parser.add_argument('--input', '-i', type=str, default=FEATURES_DATA_DIR, 
                        help='Input directory containing features data')
    parser.add_argument('--output', '-o', type=str, default=MODELS_DIR,
                        help='Output directory for trained models')
    parser.add_argument('--model-type', '-m', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'lgbm', 'lstm'],
                        help='Type of model to train')
    parser.add_argument('--regression', '-r', action='store_true',
                        help='Train a regression model instead of classification')
    parser.add_argument('--target', '-t', type=str, default=TARGET_VARIABLE,
                        help='Target variable for prediction')
    parser.add_argument('--horizon', '-d', type=int, default=PREDICTION_HORIZON,
                        help='Prediction horizon in days')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='Input features file (if not specified, will use the latest file in the input directory)')
    
    args = parser.parse_args()
    
    logger.info("Starting model training")
    
    # Find the input file if not specified
    input_file = args.file
    if not input_file:
        # Use the latest features file in the input directory
        feature_files = [f for f in os.listdir(args.input) if f.startswith('features_') and f.endswith('.csv')]
        if feature_files:
            # Sort by modification time (latest first)
            feature_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.input, x)), reverse=True)
            input_file = os.path.join(args.input, feature_files[0])
            logger.info(f"Using latest features file: {input_file}")
        else:
            logger.error("No features files found in the input directory")
            return
    
    # Load the data
    data = pd.read_csv(input_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded data with shape {data.shape}")
    
    # Initialize the model trainer
    trainer = ModelTrainer(
        data=data,
        target_variable=args.target,
        prediction_horizon=args.horizon,
        model_type=args.model_type,
        output_dir=args.output
    )
    
    # Train the model
    model = trainer.train_model(classification=not args.regression)
    
    # Save the model
    model_path = trainer.save_model()
    
    logger.info("Model training complete")
    
    return model_path


if __name__ == "__main__":
    main()
