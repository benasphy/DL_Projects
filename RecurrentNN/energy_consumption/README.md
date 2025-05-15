# Energy Consumption Prediction with RNNs

This project implements a deep learning model for forecasting energy consumption using Recurrent Neural Networks (LSTM and GRU) with seasonal pattern recognition and anomaly detection.

## Features

- **Multiple RNN Architectures**: Choose between LSTM and GRU models
- **Multi-step Forecasting**: Predict energy consumption for multiple future time steps
- **Seasonal Pattern Recognition**: Identify daily, weekly, and yearly consumption patterns
- **Weather & Time Features**: Incorporate external factors that influence energy usage
- **Anomaly Detection**: Identify unusual consumption patterns
- **Interactive Dashboard**: Streamlit web interface for data visualization and predictions
- **Customizable Parameters**: Adjust sequence length, forecast horizon, and model hyperparameters

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
   - Load energy consumption data or use sample data
   - Analyze seasonal patterns and anomalies
   - Train the model with your chosen configuration
   - Visualize predictions and performance metrics
   - Make new forecasts with different parameters

## Model Architecture

The model uses a sequence-to-sequence architecture for multi-step forecasting or a sequence-to-one architecture for single-step forecasting. The core architecture consists of:

1. Input layer for energy consumption data and additional features
2. LSTM or GRU layers (configurable)
3. Optional attention mechanism
4. Dense output layer(s) for energy consumption prediction

## Data Pipeline

1. Energy consumption data is loaded from CSV or generated synthetically
2. Time features are extracted (hour of day, day of week, month, etc.)
3. Weather features are incorporated if available
4. Data is normalized using MinMaxScaler
5. Sequences are created with a sliding window approach
6. Data is split into training and testing sets

## Anomaly Detection

The anomaly detection system uses a statistical approach:

1. Calculate rolling mean and standard deviation over a configurable window
2. Compute z-score for each data point
3. Flag points with z-scores exceeding a threshold as anomalies
4. Visualize anomalies on the energy consumption timeline

## Requirements

See requirements.txt for a complete list of dependencies.
