import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class EnergyConsumptionModel:
    def __init__(self, sequence_length=24, forecast_horizon=24, use_attention=False):
        """Initialize Energy Consumption Prediction model
        
        Args:
            sequence_length (int): Number of time steps to use for prediction
            forecast_horizon (int): Number of time steps to predict
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
            
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.use_attention = use_attention
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.data = None
        self.features = None
        self.feature_names = []
        self.is_multivariate = False
        
    def load_data(self, filepath, date_column='date', target_column='energy_consumption'):
        """Load energy consumption data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            date_column (str): Name of date column
            target_column (str): Name of target column
            
        Returns:
            DataFrame: Energy consumption data
        """
        # Load data
        data = pd.read_csv(filepath)
        
        # Convert date column to datetime
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Set date column as index
        data = data.set_index(date_column)
        
        # Sort by date
        data = data.sort_index()
        
        # Store data
        self.data = data
        
        return data
    
    def add_time_features(self, data):
        """Add time-based features to data
        
        Args:
            data (DataFrame): Energy consumption data
            
        Returns:
            DataFrame: Data with time features
        """
        # Extract time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        data['year'] = data.index.year
        
        # Add cyclical features for hour, day of week, day of month, and month
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_month_sin'] = np.sin(2 * np.pi * data['day_of_month'] / 31)
        data['day_of_month_cos'] = np.cos(2 * np.pi * data['day_of_month'] / 31)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            data[f'lag_{lag}'] = data['energy_consumption'].shift(lag)
            
        # Add rolling mean features
        for window in [3, 6, 12, 24, 48]:
            data[f'rolling_mean_{window}'] = data['energy_consumption'].rolling(window=window).mean()
            
        # Add rolling std features
        for window in [12, 24, 48]:
            data[f'rolling_std_{window}'] = data['energy_consumption'].rolling(window=window).std()
            
        # Add day type (weekday/weekend)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Add holiday flag (placeholder - would need actual holiday data)
        data['is_holiday'] = 0
        
        # Drop rows with NaN values
        data = data.dropna()
        
        return data
    
    def add_weather_features(self, data, weather_data):
        """Add weather features to data
        
        Args:
            data (DataFrame): Energy consumption data
            weather_data (DataFrame): Weather data
            
        Returns:
            DataFrame: Data with weather features
        """
        # Merge data with weather data
        data = data.merge(weather_data, left_index=True, right_index=True, how='left')
        
        # Forward fill missing weather data
        data = data.ffill()
        
        return data
    
    def detect_anomalies(self, data, target_column='energy_consumption', window=24, threshold=3):
        """Detect anomalies in energy consumption data
        
        Args:
            data (DataFrame): Energy consumption data
            target_column (str): Name of target column
            window (int): Window size for rolling statistics
            threshold (float): Threshold for anomaly detection
            
        Returns:
            DataFrame: Data with anomaly flag
        """
        # Calculate rolling mean and standard deviation
        data['rolling_mean'] = data[target_column].rolling(window=window).mean()
        data['rolling_std'] = data[target_column].rolling(window=window).std()
        
        # Calculate z-score
        data['z_score'] = (data[target_column] - data['rolling_mean']) / data['rolling_std']
        
        # Flag anomalies
        data['is_anomaly'] = (np.abs(data['z_score']) > threshold).astype(int)
        
        # Drop temporary columns
        data = data.drop(['rolling_mean', 'rolling_std', 'z_score'], axis=1)
        
        return data
    
    def prepare_data(self, data, target_column='energy_consumption', test_size=0.2, include_features=True):
        """Prepare data for training
        
        Args:
            data (DataFrame): Energy consumption data
            target_column (str): Column to predict
            test_size (float): Fraction of data to use for testing
            include_features (bool): Whether to include additional features
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        self.data = data.copy()
        
        # Define features
        if include_features:
            # Exclude non-feature columns
            exclude_columns = ['energy_consumption', 'is_anomaly']
            self.feature_names = [col for col in data.columns if col not in exclude_columns]
            self.is_multivariate = True
        else:
            self.feature_names = [target_column]
            self.is_multivariate = False
            
        # Extract features and target
        features = data[self.feature_names].values
        target = data[target_column].values.reshape(-1, 1)
        
        # Scale features
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        features = self.feature_scaler.fit_transform(features)
        
        # Scale target
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        target = self.scaler.fit_transform(target)
        
        # Create sequences for single-step prediction
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:i+self.sequence_length])
            y.append(target[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y for single-step prediction if forecast_horizon is 1
        if self.forecast_horizon == 1:
            y = y.reshape(-1, 1)
        
        # Split data
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def build_lstm_model(self, input_shape, output_shape):
        """Build LSTM model
        
        Args:
            input_shape (tuple): Shape of input data
            output_shape (int): Shape of output data
            
        Returns:
            Model: LSTM model
        """
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=output_shape))
        
        return model
    
    def build_gru_model(self, input_shape, output_shape):
        """Build GRU model
        
        Args:
            input_shape (tuple): Shape of input data
            output_shape (int): Shape of output data
            
        Returns:
            Model: GRU model
        """
        model = Sequential()
        model.add(GRU(units=64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=output_shape))
        
        return model
    
    def build_attention_model(self, input_shape, output_shape):
        """Build attention-based model
        
        Args:
            input_shape (tuple): Shape of input data
            output_shape (int): Shape of output data
            
        Returns:
            Model: Attention-based model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = LSTM(units=64, return_sequences=True)(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        
        lstm2 = LSTM(units=64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Attention layer
        attention = tf.keras.layers.Attention()([lstm2, lstm2])
        
        # Combine attention with LSTM output
        concat = Concatenate()([lstm2, attention])
        
        # Final LSTM layer
        lstm3 = LSTM(units=64)(concat)
        lstm3 = Dropout(0.2)(lstm3)
        
        # Output layer
        outputs = Dense(units=output_shape)(lstm3)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_model(self, model_type='lstm'):
        """Build model based on model_type
        
        Args:
            model_type (str): Type of model to build
            
        Returns:
            Model: RNN model
        """
        input_shape = (self.sequence_length, len(self.feature_names))
        output_shape = self.forecast_horizon
        
        if model_type == 'lstm':
            if self.use_attention:
                self.model = self.build_attention_model(input_shape, output_shape)
            else:
                self.model = self.build_lstm_model(input_shape, output_shape)
        elif model_type == 'gru':
            self.model = self.build_gru_model(input_shape, output_shape)
        
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
        
        # Reshape predictions for inverse transform if needed
        if self.forecast_horizon == 1:
            predictions = predictions.reshape(-1, 1)
        else:
            # For multi-step forecasting, reshape to 2D for inverse transform
            original_shape = predictions.shape
            predictions = predictions.reshape(-1, 1)
            
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        # Reshape back to original shape if needed
        if self.forecast_horizon > 1:
            predictions = predictions.reshape(original_shape)
            
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
        
        # Reshape y_test for inverse transform if needed
        if self.forecast_horizon == 1:
            y_test_reshaped = y_test.reshape(-1, 1)
        else:
            # For multi-step forecasting, reshape to 2D for inverse transform
            original_shape = y_test.shape
            y_test_reshaped = y_test.reshape(-1, 1)
            
        # Inverse transform targets
        y_test_inv = self.scaler.inverse_transform(y_test_reshaped)
        
        # Reshape back to original shape if needed
        if self.forecast_horizon > 1:
            y_test_inv = y_test_inv.reshape(original_shape)
            
        # Calculate metrics for each forecast step
        metrics = {}
        
        if self.forecast_horizon == 1:
            # Single-step forecast
            mse = mean_squared_error(y_test_inv, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_inv, predictions)
            mape = np.mean(np.abs((y_test_inv - predictions) / y_test_inv)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
        else:
            # Multi-step forecast
            for i in range(self.forecast_horizon):
                step_y_test = y_test_inv[:, i]
                step_pred = predictions[:, i]
                
                mse = mean_squared_error(step_y_test, step_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(step_y_test, step_pred)
                mape = np.mean(np.abs((step_y_test - step_pred) / step_y_test)) * 100
                
                metrics[f'step_{i+1}'] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
                
            # Calculate average metrics across all steps
            avg_mse = np.mean([metrics[f'step_{i+1}']['mse'] for i in range(self.forecast_horizon)])
            avg_rmse = np.mean([metrics[f'step_{i+1}']['rmse'] for i in range(self.forecast_horizon)])
            avg_mae = np.mean([metrics[f'step_{i+1}']['mae'] for i in range(self.forecast_horizon)])
            avg_mape = np.mean([metrics[f'step_{i+1}']['mape'] for i in range(self.forecast_horizon)])
            
            metrics['average'] = {
                'mse': avg_mse,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'mape': avg_mape
            }
            
        return metrics
    
    def plot_predictions(self, X_test, y_test, dates=None, num_samples=100):
        """Plot predictions vs actual values
        
        Args:
            X_test (array): Test data
            y_test (array): Test targets
            dates (array): Dates for test data
            num_samples (int): Number of samples to plot
            
        Returns:
            Figure: Matplotlib figure
        """
        # Make predictions
        predictions = self.predict(X_test)
        
        # Reshape y_test for inverse transform if needed
        if self.forecast_horizon == 1:
            y_test_reshaped = y_test.reshape(-1, 1)
        else:
            # For multi-step forecasting, reshape to 2D for inverse transform
            original_shape = y_test.shape
            y_test_reshaped = y_test.reshape(-1, 1)
            
        # Inverse transform targets
        y_test_inv = self.scaler.inverse_transform(y_test_reshaped)
        
        # Reshape back to original shape if needed
        if self.forecast_horizon > 1:
            y_test_inv = y_test_inv.reshape(original_shape)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if self.forecast_horizon == 1:
            # Single-step forecast
            if dates is not None:
                ax.plot(dates[-num_samples:], y_test_inv[-num_samples:], label='Actual')
                ax.plot(dates[-num_samples:], predictions[-num_samples:], label='Predicted')
            else:
                ax.plot(y_test_inv[-num_samples:], label='Actual')
                ax.plot(predictions[-num_samples:], label='Predicted')
        else:
            # Multi-step forecast - plot the first sample with all forecast steps
            sample_idx = 0
            
            # Create x-axis values (forecast steps)
            steps = np.arange(self.forecast_horizon)
            
            ax.plot(steps, y_test_inv[sample_idx], label='Actual')
            ax.plot(steps, predictions[sample_idx], label='Predicted')
            ax.set_xlabel('Forecast Step')
            
        ax.set_title('Energy Consumption Prediction')
        ax.set_ylabel('Energy Consumption')
        ax.legend()
        
        return fig
    
    def plot_forecast(self, X_test, y_test, sample_idx=0):
        """Plot multi-step forecast for a single sample
        
        Args:
            X_test (array): Test data
            y_test (array): Test targets
            sample_idx (int): Index of sample to plot
            
        Returns:
            Figure: Matplotlib figure
        """
        if self.forecast_horizon == 1:
            raise ValueError("This method is only for multi-step forecasting")
            
        # Make predictions
        predictions = self.predict(X_test)
        
        # Reshape y_test for inverse transform
        original_shape = y_test.shape
        y_test_reshaped = y_test.reshape(-1, 1)
            
        # Inverse transform targets
        y_test_inv = self.scaler.inverse_transform(y_test_reshaped)
        
        # Reshape back to original shape
        y_test_inv = y_test_inv.reshape(original_shape)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create x-axis values (forecast steps)
        steps = np.arange(self.forecast_horizon)
        
        ax.plot(steps, y_test_inv[sample_idx], label='Actual')
        ax.plot(steps, predictions[sample_idx], label='Predicted')
        ax.set_title(f'Multi-step Forecast (Sample {sample_idx})')
        ax.set_xlabel('Forecast Step')
        ax.set_ylabel('Energy Consumption')
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
        
    def detect_seasonal_patterns(self, data, target_column='energy_consumption'):
        """Detect seasonal patterns in energy consumption data
        
        Args:
            data (DataFrame): Energy consumption data
            target_column (str): Name of target column
            
        Returns:
            dict: Seasonal patterns
        """
        # Resample data to different frequencies
        hourly_avg = data[target_column].groupby(data.index.hour).mean()
        daily_avg = data[target_column].groupby(data.index.dayofweek).mean()
        monthly_avg = data[target_column].groupby(data.index.month).mean()
        
        # Create seasonal patterns dictionary
        seasonal_patterns = {
            'hourly': hourly_avg.to_dict(),
            'daily': daily_avg.to_dict(),
            'monthly': monthly_avg.to_dict()
        }
        
        return seasonal_patterns
    
    def plot_seasonal_patterns(self, data, target_column='energy_consumption'):
        """Plot seasonal patterns in energy consumption data
        
        Args:
            data (DataFrame): Energy consumption data
            target_column (str): Name of target column
            
        Returns:
            Figure: Matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot hourly pattern
        hourly_avg = data[target_column].groupby(data.index.hour).mean()
        axes[0].plot(hourly_avg.index, hourly_avg.values)
        axes[0].set_title('Hourly Pattern')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Energy Consumption')
        axes[0].set_xticks(range(0, 24, 2))
        
        # Plot daily pattern
        daily_avg = data[target_column].groupby(data.index.dayofweek).mean()
        axes[1].plot(daily_avg.index, daily_avg.values)
        axes[1].set_title('Daily Pattern')
        axes[1].set_xlabel('Day of Week')
        axes[1].set_ylabel('Energy Consumption')
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Plot monthly pattern
        monthly_avg = data[target_column].groupby(data.index.month).mean()
        axes[2].plot(monthly_avg.index, monthly_avg.values)
        axes[2].set_title('Monthly Pattern')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Energy Consumption')
        axes[2].set_xticks(range(1, 13))
        axes[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.tight_layout()
        
        return fig
