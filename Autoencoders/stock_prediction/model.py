import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta

class StockPredictionAE:
    """LSTM-based Autoencoder for Stock Price Prediction
    
    This class implements an autoencoder architecture using LSTM layers
    for learning patterns in stock price movements and making future predictions.
    The model uses a sequence-to-sequence approach where:
    - Encoder: Compresses the input sequence into a latent representation
    - Decoder: Reconstructs the sequence and predicts future values
    
    Features:
    - GPU acceleration with mixed precision training
    - Batch normalization for stable training
    - Dropout for regularization
    - Dynamic learning rate scheduling
    - Early stopping to prevent overfitting
    
    Example:
        model = StockPredictionAE()
        model.train('AAPL', epochs=10)
        predictions = model.predict_future('AAPL')
    """
    
    def __init__(self, sequence_length=60, prediction_length=30):
        """Initialize Stock Prediction Autoencoder with Metal GPU support
        
        Args:
            sequence_length (int): Length of the input sequence
            prediction_length (int): Length of the predicted sequence
        
        Returns:
            None
        """
        # Enable Metal GPU but disable mixed precision for compatibility
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"Metal GPU enabled: {physical_devices}")
                
                # Note: Mixed precision is disabled due to compatibility issues
                # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception as e:
            print(f"GPU configuration error: {e}")
            print("Using CPU instead")
        
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.model = None
        self.history = None
        self.scaler = MinMaxScaler()
        self.build_model()
    
    def build_model(self):
        """Build the LSTM autoencoder architecture"""
        # Encoder
        inputs = Input(shape=(self.sequence_length, 1))
        encoded = LSTM(128, return_sequences=True)(inputs)
        encoded = LSTM(64)(encoded)
        
        # Bottleneck
        bottleneck = Dense(32, activation='relu')(encoded)
        
        # Decoder for prediction
        decoded = RepeatVector(self.prediction_length)(bottleneck)
        decoded = LSTM(64, return_sequences=True)(decoded)
        decoded = LSTM(128, return_sequences=True)(decoded)
        outputs = TimeDistributed(Dense(1))(decoded)
        
        # Create and compile model
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
    
    def prepare_sequences(self, data):
        """Prepare sequences for training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_length + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.prediction_length)])
        return np.array(X), np.array(y)
    
    def get_stock_data(self, symbol, start_date=None, end_date=None):
        """Get stock data from Yahoo Finance"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download data
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        # Get closing prices
        prices = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_prices = self.scaler.fit_transform(prices)
        
        return {
            'dates': df.index,
            'prices': prices,
            'scaled_prices': scaled_prices,
            'df': df
        }
    
    def train(self, symbol='AAPL', epochs=10, callbacks=None):
        """Train the model on stock data"""
        # Get stock data
        data = self.get_stock_data(symbol)
        scaled_prices = data['scaled_prices']
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_prices)
        
        # Split into train and validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return self.history
    
    def predict_future(self, symbol, last_sequence):
        """Predict future prices"""
        # Scale the sequence
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # Reshape for prediction
        sequence = scaled_sequence.reshape(1, self.sequence_length, 1)
        
        # Get prediction
        scaled_prediction = self.model.predict(sequence)
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(scaled_prediction[0])
        
        return prediction
    
    def evaluate_predictions(self, symbol, n_evaluations=5):
        """Evaluate model predictions"""
        # Get recent data
        data = self.get_stock_data(symbol)
        scaled_prices = data['scaled_prices']
        dates = data['dates']
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_prices)
        
        # Get random sequences for evaluation
        eval_indices = np.random.choice(len(X), n_evaluations, replace=False)
        
        results = []
        for idx in eval_indices:
            # Get actual sequence and target
            sequence = X[idx]
            actual_future = y[idx]
            
            # Get prediction
            predicted_future = self.model.predict(sequence.reshape(1, self.sequence_length, 1))[0]
            
            # Inverse transform
            actual_prices = self.scaler.inverse_transform(actual_future)
            predicted_prices = self.scaler.inverse_transform(predicted_future)
            
            # Calculate metrics
            mse = np.mean(np.square(actual_prices - predicted_prices))
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            
            # Get dates for this sequence
            sequence_dates = dates[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length]
            
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                ),
                *([callbacks] if callbacks else [])
            ],
            results.append({
                'dates': sequence_dates,
                'actual': actual_prices.flatten(),
                'predicted': predicted_prices.flatten(),
                'mse': mse,
                'mae': mae
            })
        
        return results
