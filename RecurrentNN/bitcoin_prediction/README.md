# Bitcoin Price Prediction with RNNs

This project implements a deep learning model for predicting Bitcoin prices using Recurrent Neural Networks (LSTM and GRU) with technical indicators and attention mechanisms.

## Features

- **Multiple RNN Architectures**: Choose between LSTM and GRU models
- **Technical Indicators**: Integration of RSI, MACD, Bollinger Bands, and moving averages
- **Attention Mechanism**: Optional attention layer for improved performance
- **Interactive Dashboard**: Streamlit web interface for data visualization and predictions
- **Customizable Parameters**: Adjust sequence length, prediction horizon, and model hyperparameters
- **Historical Data**: Automatic fetching of Bitcoin price data using yfinance
- **Performance Metrics**: Comprehensive evaluation with MSE, RMSE, MAE, and MAPE

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

3. In the web interface:
   - Configure model parameters in the sidebar
   - Load historical Bitcoin data
   - Train the model with your chosen configuration
   - Visualize predictions and performance metrics
   - Make new predictions on recent data

## Model Architecture

The model uses a sequence-to-one architecture where a sequence of historical prices and technical indicators is used to predict future prices. The core architecture consists of:

1. Input layer for time series data and technical indicators
2. LSTM or GRU layers (configurable)
3. Optional attention mechanism
4. Dense output layer for price prediction

## Data Pipeline

1. Historical Bitcoin price data is fetched using yfinance
2. Technical indicators are calculated using the ta library
3. Data is normalized using MinMaxScaler
4. Sequences are created with a sliding window approach
5. Data is split into training and testing sets

## Requirements

See requirements.txt for a complete list of dependencies.
