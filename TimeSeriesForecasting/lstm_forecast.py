"""
Advanced LSTM for time series forecasting
Inspired by FeedForwardNN/mnist_recognition structure
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

class LSTMTimeSeriesForecaster:
    def __init__(self, seq_len=10, lstm_units=32):
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.model = self.build_model()
        self.history = None
        self.data = None

    def build_model(self):
        model = tf.keras.Sequential([
            layers.LSTM(self.lstm_units, input_shape=(self.seq_len, 1)),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def generate_sine_data(self, n_samples=1000):
        # Synthetic sine wave + noise
        x = np.arange(n_samples)
        y = np.sin(0.02 * x) + 0.1 * np.random.randn(n_samples)
        return y

    def load_series(self, series):
        self.data = np.asarray(series).flatten()

    def make_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(data[i+self.seq_len])
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)
        return X, y

    def train(self, epochs=10, batch_size=32, use_synthetic=True, uploaded_series=None):
        if use_synthetic or uploaded_series is None:
            data = self.generate_sine_data(1100)
        else:
            data = np.asarray(uploaded_series).flatten()
        X, y = self.make_dataset(data[:1000])
        X_val, y_val = self.make_dataset(data[1000:])
        self.history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return self.history

    def forecast(self, input_seq, steps=1):
        # input_seq: shape (seq_len,)
        seq = input_seq.copy().reshape(1, self.seq_len, 1)
        preds = []
        for _ in range(steps):
            pred = self.model.predict(seq)[0, 0]
            preds.append(pred)
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, 0] = pred
        return np.array(preds)

    def plot_training_history(self):
        if self.history is None:
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['mae'], label='Training')
        ax1.plot(self.history.history['val_mae'], label='Validation')
        ax1.set_title('Mean Absolute Error')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE')
        ax1.legend()
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Loss (MSE)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        plt.tight_layout()
        return fig

    def plot_forecast(self, true_series, pred_series, start_idx=0):
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(start_idx, start_idx + len(pred_series))
        ax.plot(np.arange(len(true_series)), true_series, label='True Series')
        ax.plot(idx, pred_series, label='Forecast', marker='o')
        ax.set_title('LSTM Forecast vs True Series')
        ax.legend()
        plt.tight_layout()
        return fig
