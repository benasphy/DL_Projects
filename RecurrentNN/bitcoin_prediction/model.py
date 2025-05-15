import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from datetime import datetime, timedelta
import os

class BitcoinPredictionModel:
    def __init__(self, model_type='lstm', sequence_length=60, use_attention=False):
        """Initialize Bitcoin price prediction model
        
        Args:
            model_type (str): Type of RNN cell to use ('lstm', 'gru', 'bilstm')
            sequence_length (int): Number of time steps to use for prediction
            use_attention (bool): Whether to use attention mechanism
        """
        # Enable Metal GPU but disable mixed precision for compatibility
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"Metal GPU enabled: {physical_devices}")
                
                # Note: Mixed precision is disabled due to compatibility issues with CudnnRNN
                # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception as e:
            print(f"GPU configuration error: {e}")
            print("Using CPU instead")
            
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.data = None
        self.features = None
        self.feature_names = []
        
    def download_data(self, symbol='BTC-USD', period='2y'):
        """Download Bitcoin price data
        
        Args:
            symbol (str): Symbol to download data for
            period (str): Period to download data for (e.g. '2y' for 2 years)
            
        Returns:
            DataFrame: Bitcoin price data
        """
        # Download data
        data = yf.download(symbol, period=period)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
        
        # Add date as a column
        data['Date'] = data.index
        
        return data
    
    def add_technical_indicators(self, data):
        """Add technical indicators to data
        
        Args:
            data (DataFrame): Price data
            
        Returns:
            DataFrame: Price data with technical indicators
        """
        # Print column names for debugging
        print(f"Columns in data: {data.columns}")
        
        # Check if we have MultiIndex columns and handle them
        if isinstance(data.columns, pd.MultiIndex):
            # Convert MultiIndex columns to flat columns
            data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
            print(f"Flattened columns: {data.columns}")
        
        # Find the close price column
        close_col = None
        for col in data.columns:
            if 'close' in str(col).lower():
                close_col = col
                break
        
        if close_col is None:
            raise ValueError(f"Could not find close price column in {data.columns}")
            
        print(f"Using close price column: {close_col}")
        close_series = data[close_col]
        
        # Calculate RSI
        data['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()
        
        # Calculate MACD
        macd = ta.trend.MACD(close_series)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close_series)
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()
        data['Bollinger_Width'] = bollinger.bollinger_wband()
        
        # Calculate moving averages
        data['MA_5'] = ta.trend.SMAIndicator(close_series, window=5).sma_indicator()
        data['MA_20'] = ta.trend.SMAIndicator(close_series, window=20).sma_indicator()
        data['MA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()
        
        # Calculate exponential moving averages
        data['EMA_12'] = ta.trend.EMAIndicator(close_series, window=12).ema_indicator()
        data['EMA_26'] = ta.trend.EMAIndicator(close_series, window=26).ema_indicator()
        
        # Find high price column
        high_col = None
        for col in data.columns:
            if 'high' in str(col).lower():
                high_col = col
                break
        
        if high_col is None:
            print(f"Warning: Could not find high price column in {data.columns}")
            high_series = close_series  # Fallback to close price
        else:
            print(f"Using high price column: {high_col}")
            high_series = data[high_col]
        
        # Find low price column
        low_col = None
        for col in data.columns:
            if 'low' in str(col).lower():
                low_col = col
                break
        
        if low_col is None:
            print(f"Warning: Could not find low price column in {data.columns}")
            low_series = close_series  # Fallback to close price
        else:
            print(f"Using low price column: {low_col}")
            low_series = data[low_col]
        
        # Calculate ATR
        data['ATR'] = ta.volatility.AverageTrueRange(high_series, low_series, close_series).average_true_range()
        
        # Find volume column
        volume_col = None
        for col in data.columns:
            if 'volume' in str(col).lower():
                volume_col = col
                break
        
        if volume_col is None:
            print(f"Warning: Could not find volume column in {data.columns}")
            # Create a dummy volume series
            volume_series = pd.Series(np.ones(len(close_series)), index=close_series.index)
        else:
            print(f"Using volume column: {volume_col}")
            volume_series = data[volume_col]
            
        # Calculate volume indicators
        data['Volume_MA_5'] = ta.trend.SMAIndicator(volume_series, window=5).sma_indicator()
        data['Volume_MA_20'] = ta.trend.SMAIndicator(volume_series, window=20).sma_indicator()
        
        # Calculate price momentum
        data['Price_Change'] = close_series.pct_change()
        data['Price_Change_5'] = close_series.pct_change(periods=5)
        
        # Store the column names for later use
        self.column_mapping = {
            'close': close_col,
            'high': high_col,
            'low': low_col,
            'volume': volume_col
        }
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def prepare_data(self, data, target_column=None, test_size=0.2, include_features=True):
        """Prepare data for training
        
        Args:
            data (DataFrame): Price data with technical indicators
            target_column (str): Column to predict (if None, will use close price column)
            test_size (float): Fraction of data to use for testing
            include_features (bool): Whether to include technical indicators as features
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        # Use the stored column mapping to get the correct target column
        if target_column is None and hasattr(self, 'column_mapping'):
            target_column = self.column_mapping.get('close', 'Close')
            print(f"Using target column: {target_column}")
        elif target_column is None:
            # Try to find a close price column
            for col in data.columns:
                if 'close' in str(col).lower():
                    target_column = col
                    break
            if target_column is None:
                target_column = 'Close'  # Default fallback
            print(f"Using target column: {target_column}")
        self.data = data.copy()
        
        # Define features using column mapping if available
        if hasattr(self, 'column_mapping'):
            # Use the detected column names
            price_features = []
            for col_type in ['open', 'high', 'low', 'close', 'volume']:
                if col_type in self.column_mapping and self.column_mapping[col_type] is not None:
                    price_features.append(self.column_mapping[col_type])
            
            # If we couldn't find all columns, print a warning
            if len(price_features) < 5:
                print(f"Warning: Could only find {len(price_features)} price columns: {price_features}")
                # Try to find any missing columns in the data
                for col in data.columns:
                    col_lower = str(col).lower()
                    if 'open' in col_lower and 'open' not in [c.lower() for c in price_features]:
                        price_features.append(col)
                    elif 'high' in col_lower and 'high' not in [c.lower() for c in price_features]:
                        price_features.append(col)
                    elif 'low' in col_lower and 'low' not in [c.lower() for c in price_features]:
                        price_features.append(col)
                    elif 'close' in col_lower and 'close' not in [c.lower() for c in price_features]:
                        price_features.append(col)
                    elif 'volume' in col_lower and 'volume' not in [c.lower() for c in price_features]:
                        price_features.append(col)
        else:
            # Fallback to default column names
            price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Check if these columns exist in the data
            for col in price_features.copy():
                if col not in data.columns:
                    print(f"Warning: Column {col} not found in data. Removing from features.")
                    price_features.remove(col)
        
        # Print the price features we're using
        print(f"Using price features: {price_features}")
        
        technical_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'MA_5', 'MA_20', 'MA_50', 'EMA_12', 'EMA_26',
            'ATR', 'Volume_MA_5', 'Volume_MA_20', 'Price_Change', 'Price_Change_5'
        ]
        
        if include_features:
            self.feature_names = price_features + technical_features
        else:
            self.feature_names = price_features
            
        # Filter feature names to only include columns that exist in the data
        valid_feature_names = []
        for feature in self.feature_names:
            if feature in data.columns:
                valid_feature_names.append(feature)
            else:
                print(f"Warning: Feature {feature} not found in data. Skipping.")
        
        self.feature_names = valid_feature_names
        print(f"Using features: {self.feature_names}")
        
        # Extract features
        self.features = data[self.feature_names].values
        
        # Scale features
        self.features = self.feature_scaler.fit_transform(self.features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(self.features) - self.sequence_length):
            X.append(self.features[i:i+self.sequence_length])
            # Scale the target separately to easily inverse transform later
            y.append(data[target_column].values[i+self.sequence_length])
            
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Scale target
        y = self.scaler.fit_transform(y)
        
        # Split data
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            Model: LSTM model
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            Model: GRU model
        """
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        return model
    
    def build_bilstm_model(self, input_shape):
        """Build Bidirectional LSTM model
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            Model: Bidirectional LSTM model
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=50)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        return model
    
    def build_attention_model(self, input_shape):
        """Build attention-based model
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            Model: Attention-based model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = LSTM(units=50, return_sequences=True)(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        
        lstm2 = LSTM(units=50, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Attention layer
        attention = tf.keras.layers.Attention()([lstm2, lstm2])
        
        # Combine attention with LSTM output
        concat = Concatenate()([lstm2, attention])
        
        # Final LSTM layer
        lstm3 = LSTM(units=50)(concat)
        lstm3 = Dropout(0.2)(lstm3)
        
        # Output layer
        outputs = Dense(units=1)(lstm3)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_model(self):
        """Build model based on model_type
        
        Returns:
            Model: RNN model
        """
        input_shape = (self.sequence_length, len(self.feature_names))
        
        if self.model_type == 'lstm':
            if self.use_attention:
                self.model = self.build_attention_model(input_shape)
            else:
                self.model = self.build_lstm_model(input_shape)
        elif self.model_type == 'gru':
            self.model = self.build_gru_model(input_shape)
        elif self.model_type == 'bilstm':
            self.model = self.build_bilstm_model(input_shape)
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        return self.model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train model
        
        Args:
            X_train (array): Training data
            y_train (array): Training targets
            X_test (array): Testing data
            y_test (array): Testing targets
            epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
            
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
            
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X_test):
        """Make predictions
        
        Args:
            X_test (array): Test data
            
        Returns:
            array: Predictions
        """
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model
        
        Args:
            X_test (array): Test data
            y_test (array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        predictions = self.predict(X_test)
        
        # Inverse transform targets
        y_test = self.scaler.inverse_transform(y_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def plot_predictions(self, X_test, y_test, dates=None):
        """Plot predictions vs actual values
        
        Args:
            X_test (array): Test data
            y_test (array): Test targets
            dates (array): Dates for test data
            
        Returns:
            Figure: Matplotlib figure
        """
        # Make predictions
        predictions = self.predict(X_test)
        
        # Inverse transform targets
        y_test = self.scaler.inverse_transform(y_test)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Print shapes for debugging
        print(f"Dates shape: {dates.shape if dates is not None else 'None'}")
        print(f"y_test shape: {y_test.shape}")
        print(f"predictions shape: {predictions.shape}")
        
        # Ensure shapes match
        if dates is not None:
            # Reshape y_test if needed
            if len(y_test.shape) > 1 and y_test.shape[1] == 1:
                y_test = y_test.flatten()
            
            # Reshape predictions if needed
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            # Ensure dates and y_test have the same length
            min_len = min(len(dates), len(y_test), len(predictions))
            print(f"Using min_len: {min_len}")
            
            ax.plot(dates[:min_len], y_test[:min_len], label='Actual')
            ax.plot(dates[:min_len], predictions[:min_len], label='Predicted')
        else:
            # Reshape if needed
            if len(y_test.shape) > 1 and y_test.shape[1] == 1:
                y_test = y_test.flatten()
            
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
                
            ax.plot(y_test, label='Actual')
            ax.plot(predictions, label='Predicted')
            
        ax.set_title('Bitcoin Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        
        return fig
    
    def plot_training_history(self):
        """Plot training history
        
        Returns:
            Figure: Matplotlib figure
        """
        if self.history is None:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        return fig
    
    def save_model(self, filepath):
        """Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load model from file
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = tf.keras.models.load_model(filepath)
        
    def forecast_future(self, days=30):
        """Forecast future prices
        
        Args:
            days (int): Number of days to forecast
            
        Returns:
            array: Forecasted prices
        """
        if self.data is None or self.features is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
            
        # Get the last sequence
        last_sequence = self.features[-self.sequence_length:]
        last_sequence = last_sequence.reshape(1, self.sequence_length, len(self.feature_names))
        
        # Initialize forecasted prices
        forecasted_prices = []
        
        # Create a copy of the last sequence for iterative forecasting
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict the next price
            prediction = self.model.predict(current_sequence)
            
            # Inverse transform the prediction
            prediction = self.scaler.inverse_transform(prediction)[0, 0]
            forecasted_prices.append(prediction)
            
            # Update the sequence for the next prediction
            # This is a simplified approach and would need more complex logic
            # to update all features in a real application
            new_row = current_sequence[0, -1, :].copy()
            
            # Find the index of the close price column in the feature names
            close_index = None
            
            # If we have column mapping, use it to find the close column index
            if hasattr(self, 'column_mapping') and 'close' in self.column_mapping:
                close_col = self.column_mapping['close']
                if close_col in self.feature_names:
                    close_index = self.feature_names.index(close_col)
            
            # If we couldn't find it through column mapping, try to find it by name
            if close_index is None:
                # Try to find a column that contains 'close' in its name
                for i, feature in enumerate(self.feature_names):
                    if 'close' in str(feature).lower():
                        close_index = i
                        break
            
            # If we still couldn't find it, default to index 3 (traditional position of Close)
            if close_index is None:
                print("Warning: Could not find close price column in feature names. Using default index 3.")
                print(f"Feature names: {self.feature_names}")
                close_index = 3 if len(self.feature_names) > 3 else 0
            
            print(f"Using close price at index {close_index} for forecasting")
            
            # Update the close price in the new row
            new_row[close_index] = self.scaler.transform([[prediction]])[0, 0]
            
            # Shift the sequence and add the new row
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
            
        return np.array(forecasted_prices)
    
    def get_feature_importance(self):
        """Get feature importance using a simple perturbation method
        
        Returns:
            dict: Feature importance scores
        """
        if self.model is None or self.features is None:
            raise ValueError("Model not trained or data not prepared.")
            
        # Get a sample of test data
        X_sample = self.features[-100:].reshape(1, -1, len(self.feature_names))
        
        # Get baseline prediction
        baseline_pred = self.model.predict(X_sample)
        
        # Calculate feature importance
        importance = {}
        
        for i, feature in enumerate(self.feature_names):
            # Create a perturbed version of the data
            X_perturbed = X_sample.copy()
            X_perturbed[0, :, i] = np.mean(X_perturbed[0, :, i])
            
            # Get prediction with perturbed feature
            perturbed_pred = self.model.predict(X_perturbed)
            
            # Calculate importance as the change in prediction
            importance[feature] = np.abs(baseline_pred - perturbed_pred)[0, 0]
            
        # Normalize importance
        total = sum(importance.values())
        for feature in importance:
            importance[feature] /= total
            
        return importance
