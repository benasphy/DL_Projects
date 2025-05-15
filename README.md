# Deep Learning Projects

A collection of interactive deep learning projects built with TensorFlow and Streamlit. Each project showcases different neural network architectures and applications.

## Projects

### Feed Forward Neural Networks
1. **MNIST Recognition**
   - Handwritten digit recognition using CNN
   - Interactive drawing interface
   - Real-time predictions

2. **Weather Forecasting**
   - Time series prediction using LSTM
   - Interactive visualizations
   - Historical weather data analysis

### Autoencoders
1. **Stock Price Prediction**
   - LSTM-based autoencoder for stock price forecasting
   - Real-time data from Yahoo Finance
   - Interactive predictions and evaluation
   - GPU-optimized training

2. **Dimensionality Reduction**
   - Autoencoder for data compression
   - Interactive latent space visualization
   - MNIST dataset exploration
   - GPU-optimized with mixed precision training

## Setup

1. Clone the repository:
```bash
git clone https://github.com/benasphy/DL_Projects.git
cd DL_Projects
```

2. Create a conda environment:
```bash
conda create -n dl-env python=3.10
conda activate dl-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the main menu:
```bash
streamlit run main.py
```

## Features

- **Interactive UI**: Built with Streamlit for easy interaction
- **GPU Acceleration**: Optimized for Metal GPU on Apple Silicon
- **Real-time Visualization**: Using Plotly and Matplotlib
- **Modern Architecture**: Clean code structure with modular design
- **Easy Navigation**: Central menu to access all projects

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- Streamlit 1.31+
- See `requirements.txt` for full list

## Project Structure

```
DL_Projects/
├── main.py                # Main menu
├── requirements.txt       # Project dependencies
├── FeedForwardNN/        # Feed Forward Neural Network projects
│   ├── mnist_recognition/
│   └── weather_forecasting/
└── Autoencoders/         # Autoencoder projects
    ├── stock_prediction/
    └── dimensionality_reduction/
```

## Contributing

Feel free to contribute by:
1. Opening issues
2. Submitting pull requests
3. Adding new projects
4. Improving documentation

## License

MIT License - feel free to use and modify for your own projects!

