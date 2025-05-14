import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class WeatherModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def build_model(self, input_shape):
        """Build the neural network architecture"""
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1)  # Linear activation for regression
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
    def prepare_data(self, data):
        """Prepare weather data for training"""
        # Expected columns: temp, humidity, pressure, wind_speed, wind_direction, rain, cloud_cover
        features = ['temp', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'rain', 'cloud_cover']
        X = data[features].values
        y = data['next_day_temp'].values.reshape(-1, 1)
        
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        X_scaled, y_scaled = self.prepare_data(data)
        
        if self.model is None:
            self.build_model(X_scaled.shape[1])
            
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        
        return self.history
    
    def predict(self, features):
        """Make temperature prediction"""
        features_scaled = self.scaler_X.transform(features)
        pred_scaled = self.model.predict(features_scaled)
        return self.scaler_y.inverse_transform(pred_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_scaled = self.scaler_X.transform(X_test)
        y_scaled = self.scaler_y.transform(y_test.reshape(-1, 1))
        return self.model.evaluate(X_scaled, y_scaled)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # MAE plot
        ax1.plot(self.history.history['mae'], label='Training')
        ax1.plot(self.history.history['val_mae'], label='Validation')
        ax1.set_title('Model MAE')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Squared Error')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_vs_actual(self, y_true, y_pred):
        """Plot prediction vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Prediction vs Actual')
        return plt.gcf()
    
    def save_model(self, path):
        """Save the model and scalers"""
        self.model.save(f"{path}_model")
        np.save(f"{path}_scaler_X.npy", [self.scaler_X.mean_, self.scaler_X.scale_])
        np.save(f"{path}_scaler_y.npy", [self.scaler_y.mean_, self.scaler_y.scale_])
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model and scalers"""
        instance = cls()
        instance.model = tf.keras.models.load_model(f"{path}_model")
        
        # Load scalers
        scaler_X_params = np.load(f"{path}_scaler_X.npy")
        scaler_y_params = np.load(f"{path}_scaler_y.npy")
        
        instance.scaler_X.mean_ = scaler_X_params[0]
        instance.scaler_X.scale_ = scaler_X_params[1]
        instance.scaler_y.mean_ = scaler_y_params[0]
        instance.scaler_y.scale_ = scaler_y_params[1]
        
        return instance
