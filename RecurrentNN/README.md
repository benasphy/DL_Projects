# Recurrent Neural Network Projects

This repository contains a collection of advanced projects implementing Recurrent Neural Networks (RNNs), including LSTM, GRU, and attention-based architectures for various time series and sequential data tasks.

## Projects Overview

### 1. Bitcoin Price Prediction
A deep learning application for forecasting Bitcoin prices using historical data and technical indicators.

**Key Features:**
- LSTM and GRU architectures for time series prediction
- Technical indicators integration (RSI, MACD, Bollinger Bands, moving averages)
- Attention mechanisms for improved model performance
- Interactive Streamlit dashboard for visualization and predictions
- Backtesting capabilities to evaluate model performance
- Customizable prediction horizons (1-day, 7-day, 30-day forecasts)

**Technologies:**
- TensorFlow/Keras for model implementation
- yfinance for Bitcoin historical data
- ta library for technical indicators
- Streamlit for interactive web interface
- Plotly for advanced visualizations

### 2. Energy Consumption Prediction
A forecasting system for energy consumption with seasonal pattern recognition and anomaly detection.

**Key Features:**
- Multi-step forecasting for short and long-term energy predictions
- Seasonal pattern recognition (daily, weekly, yearly cycles)
- Weather and time feature integration
- Anomaly detection for unusual consumption patterns
- Interactive visualization of predictions and historical patterns
- Customizable sequence length and forecast horizon

**Technologies:**
- TensorFlow/Keras for model implementation
- Streamlit for interactive web interface
- Plotly and Matplotlib for data visualization
- scikit-learn for data preprocessing and evaluation

### 3. Sentiment Analysis
A text classification system for analyzing sentiment in reviews, social media posts, and other text content.

**Key Features:**
- Multiple RNN architectures (LSTM, BiLSTM, Attention-based)
- Text preprocessing pipeline with NLTK
- Word embedding representations
- Multi-class classification (positive, negative, neutral)
- Attention visualization to interpret model decisions
- Batch prediction capabilities for large datasets

**Technologies:**
- TensorFlow/Keras for model implementation
- NLTK for text preprocessing
- Streamlit for interactive web interface
- WordCloud for text visualization
- Hugging Face datasets integration for sample data

### 4. Text Generation
A creative text generation system that can produce human-like text in various styles.

**Key Features:**
- Character-level and word-level text generation
- Multiple RNN architectures (LSTM, GRU, Attention-based)
- Temperature-based sampling for controlling creativity
- Interactive text generation with custom seeds
- Support for different literary styles and genres
- Exploration of temperature effects on text generation

**Technologies:**
- TensorFlow/Keras for model implementation
- Streamlit for interactive web interface
- Custom sampling algorithms for text generation
- Visualization of model training and generation process

## Common Features Across Projects

- **GPU Acceleration**: All models support Metal GPU acceleration for faster training
- **Mixed Precision Training**: Implemented for improved performance
- **Interactive Web Interfaces**: Streamlit apps for easy interaction
- **Visualization Tools**: Comprehensive data and model performance visualization
- **Modular Architecture**: Well-structured code for easy extension and modification

## Requirements

Each project folder contains its own `requirements.txt` file with the specific dependencies needed for that project.

## Getting Started

To run any of these projects:

1. Navigate to the specific project folder:
   ```bash
   cd RecurrentNN/<project_folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

Each project follows a similar structure:

```
project_folder/
├── app.py              # Streamlit web application
├── model.py            # Implementation of the RNN models
├── requirements.txt    # Project-specific dependencies
└── README.md           # Project-specific documentation
```

## Model Architectures

All projects implement multiple RNN architectures:

1. **LSTM (Long Short-Term Memory)**
   - Specialized RNN that can learn long-term dependencies
   - Uses gates to control information flow (input, forget, output gates)
   - Effective for capturing long-range patterns in sequential data

2. **GRU (Gated Recurrent Unit)**
   - Simplified version of LSTM with fewer parameters
   - Often performs similarly to LSTM but trains faster
   - Good for smaller datasets or when computational resources are limited

3. **Attention Mechanisms**
   - Helps models focus on relevant parts of the input sequence
   - Improves performance for longer sequences
   - Provides interpretability by showing which parts of the input are important

## Future Enhancements

Planned future enhancements for these projects include:

- Integration with more data sources
- Transformer-based models for comparison
- Deployment options for production environments
- Transfer learning capabilities
- Ensemble methods for improved performance

## Acknowledgments

These projects were developed as educational resources for learning about recurrent neural networks and their applications in various domains.
