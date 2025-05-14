# Weather Forecasting with Neural Networks

A deep learning project that predicts weather conditions using a feedforward neural network. The model is trained on historical weather data and provides next-day temperature predictions through an interactive Streamlit interface.

## Features
- Next-day temperature prediction
- Multiple input features (temperature, humidity, pressure, etc.)
- Interactive visualization of predictions
- Historical data analysis
- Model performance metrics
- Real-time weather data integration

## Model Architecture
- Input layer: 7 neurons (7 weather features)
- Hidden layers:
  - Dense layer (128 neurons, ReLU activation)
  - Dropout (0.2)
  - Dense layer (64 neurons, ReLU activation)
  - Dropout (0.2)
- Output layer: 1 neuron (linear activation)

## Setup and Installation
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Select a location
2. View current weather conditions
3. Get next-day temperature prediction
4. Analyze historical predictions

## Model Performance
- Mean Absolute Error: ~1.5°C
- Root Mean Square Error: ~2.0°C
- R² Score: ~0.85

## Files
- `app.py`: Streamlit web interface
- `model.py`: Neural network implementation
- `requirements.txt`: Project dependencies

## Data Sources
- Historical weather data from public datasets
- Real-time weather data from OpenWeatherMap API

## Deployment
To deploy on Streamlit Cloud:
1. Push to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Set up environment variables for API keys
4. Deploy the app

## License
MIT License
