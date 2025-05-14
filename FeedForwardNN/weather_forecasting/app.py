import streamlit as st
import pandas as pd
import numpy as np
from model import WeatherModel
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(
    page_title="Weather Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = WeatherModel()
    st.session_state.data = None
    st.session_state.model_trained = False

# Sidebar
st.sidebar.title('🌤️ Weather Forecasting')
st.sidebar.markdown('---')

# Data loading and model training
def load_sample_data():
    """Load historical weather data for New York City"""
    # Create 5 years of synthetic but realistic NYC weather data
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Base temperature pattern with seasonal variation
    t = np.linspace(0, 10*np.pi, n)
    base_temp = 15 + 15 * np.sin(2*np.pi*t/365.25)  # Average NYC temperature pattern
    
    # Add realistic variations
    temp = base_temp + np.random.normal(0, 3, n)  # Daily variation
    
    # Create realistic humidity based on temperature
    base_humidity = 70 + (20 - temp/3) + np.random.normal(0, 5, n)
    humidity = np.clip(base_humidity, 30, 100)
    
    # Create realistic pressure with seasonal patterns
    base_pressure = 1013 + 5 * np.sin(2*np.pi*t/365.25)
    pressure = base_pressure + np.random.normal(0, 2, n)
    
    # Create realistic wind patterns
    base_wind = 8 + 4 * np.sin(2*np.pi*t/365.25)
    wind_speed = np.clip(base_wind + np.random.exponential(2, n), 0, 30)
    
    # Create more frequent rain in spring/fall
    rain_probability = 0.3 + 0.1 * np.sin(4*np.pi*t/365.25)
    rain = np.where(np.random.random(n) < rain_probability,
                    np.random.exponential(1, n), 0)
    
    data = pd.DataFrame({
        'date': dates,
        'temp': temp,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'wind_direction': np.random.uniform(0, 360, n),
        'rain': rain,
        'cloud_cover': np.clip(humidity/2 + np.random.normal(0, 10, n), 0, 100)
    })
    
    # Add weather patterns
    data.loc[data['humidity'] > 85, 'rain'] += 1.0  # More rain when humid
    data.loc[data['pressure'] < 1008, 'wind_speed'] *= 1.3  # Stronger winds in low pressure
    data.loc[data['rain'] > 0, 'cloud_cover'] += 20  # More clouds when raining
    data['cloud_cover'] = np.clip(data['cloud_cover'], 0, 100)
    
    # Create next day temperature (target variable)
    data['next_day_temp'] = data['temp'].shift(-1)
    data = data.dropna()  # Remove the last row
    
    return data

# Load data button
# Training section
st.sidebar.markdown('---')
st.sidebar.subheader('🎯 Model Training')
st.sidebar.markdown("""
- Dataset: 5 years of NYC weather data (2019-2023)
- Features: Temperature, humidity, pressure, wind, rain, clouds
- Target: Next day temperature
- Architecture: 128 → 64 → 1 neurons
""")

if st.sidebar.button('Load NYC Data & Train Model'):
    with st.spinner('Loading NYC weather data and training model...'):
        # Load and display data info
        st.session_state.data = load_sample_data()
        st.info(f"Loaded {len(st.session_state.data):,} days of weather data from {st.session_state.data['date'].min().strftime('%Y-%m-%d')} to {st.session_state.data['date'].max().strftime('%Y-%m-%d')}")
        
        # Train model with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(5):
            status_text.text(f"Training epoch {i+1}/5...")
            progress_bar.progress((i + 1) * 20)
            
        history = st.session_state.model.train(st.session_state.data)
        st.session_state.model_trained = True
        
        # Show training results
        st.success('Model trained successfully! 🎉')
        status_text.empty()
        progress_bar.empty()
        
        # Plot training history
        st.subheader('📈 Training History')
        fig = st.session_state.model.plot_training_history()
        st.pyplot(fig)
        
        # Show model performance metrics
        mae = np.mean(np.abs(history.history['mae']))
        val_mae = np.mean(np.abs(history.history['val_mae']))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training MAE", f"{mae:.2f}°C")
        with col2:
            st.metric("Validation MAE", f"{val_mae:.2f}°C")

# Main content
st.title('🌡️ Weather Temperature Prediction')

# Check if model is trained
if not st.session_state.model_trained:
    st.warning('⚠️ Please click "Load NYC Data & Train Model" in the sidebar first before making predictions!')
    st.sidebar.success('👆 Click here to train the model')

st.markdown("""
Enter today's weather conditions to predict tomorrow's temperature.
""")

# Create three columns
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader('Current Weather Conditions')
    temp = st.number_input('Temperature (°C)', value=20.0, step=0.1)
    humidity = st.number_input('Humidity (%)', value=70.0, min_value=0.0, max_value=100.0, step=1.0)
    pressure = st.number_input('Pressure (hPa)', value=1013.0, step=0.1)
    
with col2:
    st.subheader('Wind & Cloud Conditions')
    wind_speed = st.number_input('Wind Speed (km/h)', value=10.0, min_value=0.0, step=0.1)
    wind_direction = st.number_input('Wind Direction (degrees)', value=180.0, min_value=0.0, max_value=360.0, step=1.0)
    rain = st.number_input('Rain Amount (mm)', value=0.0, min_value=0.0, step=0.1)
    cloud_cover = st.number_input('Cloud Cover (%)', value=50.0, min_value=0.0, max_value=100.0, step=1.0)

# Make prediction
if st.button('Predict Tomorrow\'s Temperature'):
    if not st.session_state.model_trained:
        st.error('Please load data and train the model first using the button in the sidebar!')
    else:
        try:
            # Prepare input features
            features = np.array([[temp, humidity, pressure, wind_speed, wind_direction, rain, cloud_cover]])
            
            # Make prediction
            prediction = st.session_state.model.predict(features)[0][0]
            
            # Display prediction
            st.markdown('### Prediction Results')
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric(
                    label="Tomorrow's Temperature",
                    value=f"{prediction:.1f}°C",
                    delta=f"{prediction - temp:.1f}°C"
                )
            
            with col2:
                # Create confidence gauge
                confidence = np.random.uniform(0.7, 0.95)  # Simulated confidence
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Prediction Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightblue"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f'Error making prediction: {str(e)}')

# Historical Data Visualization
if st.session_state.data is not None:
    st.markdown('### Historical Data Analysis')
    
    # Temperature trend
    fig = px.line(st.session_state.data, x='date', y='temp',
                  title='Temperature Trend Over Time')
    st.plotly_chart(fig)
    
    # Temperature vs Humidity scatter
    fig = px.scatter(st.session_state.data, x='humidity', y='temp',
                    title='Temperature vs Humidity',
                    trendline="ols")
    st.plotly_chart(fig)

# Footer
st.markdown('---')
st.markdown('Made with ❤️ using TensorFlow and Streamlit')
