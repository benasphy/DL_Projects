import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from model import EnergyConsumptionModel
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="Energy Consumption Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = EnergyConsumptionModel()
    st.session_state.trained = False
    st.session_state.data = None
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.test_dates = None
    st.session_state.predictions = None
    st.session_state.metrics = None

# Sidebar
st.sidebar.title('⚡ Energy Consumption Prediction')
st.sidebar.markdown('---')

# Model configuration
st.sidebar.subheader('🧠 Model Configuration')

model_type = st.sidebar.selectbox(
    'RNN Architecture',
    ['LSTM', 'GRU'],
    index=0
)

use_attention = st.sidebar.checkbox('Use Attention Mechanism', value=False)

sequence_length = st.sidebar.slider(
    'Sequence Length (Hours)',
    min_value=6,
    max_value=72,
    value=24,
    step=6
)

forecast_horizon = st.sidebar.slider(
    'Forecast Horizon (Hours)',
    min_value=1,
    max_value=48,
    value=24,
    step=1
)

# Data configuration
st.sidebar.markdown('---')
st.sidebar.subheader('📊 Data Configuration')

include_features = st.sidebar.checkbox('Include Weather & Time Features', value=True)

# Training configuration
st.sidebar.markdown('---')
st.sidebar.subheader('🏋️ Training Configuration')

epochs = st.sidebar.slider(
    'Training Epochs',
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

batch_size = st.sidebar.slider(
    'Batch Size',
    min_value=16,
    max_value=128,
    value=32,
    step=16
)

# Main content
st.title('⚡ Energy Consumption Prediction with RNNs')

# Create tabs
data_tab, train_tab, predict_tab, anomaly_tab, explain_tab = st.tabs([
    "📈 Data Analysis", "🏋️ Model Training", "🔮 Consumption Prediction", 
    "🔍 Anomaly Detection", "📚 RNN Explanation"
])

with data_tab:
    st.markdown("### Energy Consumption Data")
    
    # Option to upload data or use sample data
    data_option = st.radio(
        "Choose data source",
        ["Upload CSV file", "Use sample data"]
    )
    
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload energy consumption CSV file", type=["csv"])
        if uploaded_file is not None:
            # Save uploaded file
            with open("uploaded_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load data
            try:
                data = st.session_state.model.load_data("uploaded_data.csv")
                st.success(f"Loaded {len(data)} records of energy consumption data")
                st.session_state.data = data
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        if st.button('Load Sample Data'):
            # Generate sample data
            date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
            data = pd.DataFrame(date_rng, columns=['date'])
            
            # Generate synthetic energy consumption with daily and weekly patterns
            hourly_pattern = np.sin(np.pi * data.index % 24 / 12) + 1
            daily_pattern = 0.3 * np.sin(np.pi * data.index % 168 / 84) + 1
            yearly_pattern = 0.5 * np.sin(np.pi * data.index / 4380) + 1
            random_noise = 0.2 * np.random.randn(len(data))
            
            data['energy_consumption'] = 100 * (hourly_pattern * daily_pattern * yearly_pattern + random_noise)
            data = data.set_index('date')
            
            # Add time features
            if include_features:
                data = st.session_state.model.add_time_features(data)
                
                # Generate synthetic weather data
                data['temperature'] = 15 + 15 * np.sin(np.pi * data.index.dayofyear / 182.5) + 5 * np.random.randn(len(data))
                data['humidity'] = 60 + 20 * np.sin(np.pi * data.index.dayofyear / 182.5) + 10 * np.random.randn(len(data))
                data['cloud_cover'] = 50 + 30 * np.sin(np.pi * data.index.dayofyear / 182.5) + 20 * np.random.randn(len(data))
                data['cloud_cover'] = data['cloud_cover'].clip(0, 100)
            
            st.session_state.data = data
            st.success(f"Generated {len(data)} hours of sample energy consumption data")
    
    # Display data if available
    if st.session_state.data is not None:
        # Show data info
        st.markdown("### Data Overview")
        st.dataframe(st.session_state.data.head())
        
        # Plot energy consumption
        st.markdown("### Energy Consumption Over Time")
        fig = px.line(
            st.session_state.data, 
            x=st.session_state.data.index, 
            y='energy_consumption',
            title='Energy Consumption History'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detect and plot seasonal patterns
        st.markdown("### Seasonal Patterns")
        if st.button('Analyze Seasonal Patterns'):
            fig = st.session_state.model.plot_seasonal_patterns(st.session_state.data)
            st.pyplot(fig)
            
            # Detect anomalies
            st.markdown("### Anomaly Detection")
            data_with_anomalies = st.session_state.model.detect_anomalies(st.session_state.data.copy())
            
            # Plot data with anomalies highlighted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_with_anomalies.index,
                y=data_with_anomalies['energy_consumption'],
                mode='lines',
                name='Energy Consumption'
            ))
            
            # Highlight anomalies
            anomalies = data_with_anomalies[data_with_anomalies['is_anomaly'] == 1]
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['energy_consumption'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(
                title='Energy Consumption with Anomalies',
                xaxis_title='Date',
                yaxis_title='Energy Consumption'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentage of anomalies
            anomaly_percentage = len(anomalies) / len(data_with_anomalies) * 100
            st.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

with train_tab:
    st.markdown("### Train Energy Consumption Prediction Model")
    
    if st.session_state.data is None:
        st.warning("Please load energy consumption data first in the Data Analysis tab")
    else:
        if st.button('Prepare Data & Train Model'):
            with st.spinner('Preparing data...'):
                # Update model parameters
                st.session_state.model = EnergyConsumptionModel(
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                    use_attention=use_attention
                )
                
                # Prepare data
                X_train, y_train, X_test, y_test = st.session_state.model.prepare_data(
                    st.session_state.data,
                    include_features=include_features
                )
                
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Build model
                st.session_state.model.build_model(model_type.lower())
                
                # Show model summary
                st.markdown("### Model Architecture")
                model_summary = []
                st.session_state.model.model.summary(print_fn=lambda x: model_summary.append(x))
                st.code('\n'.join(model_summary))
                
            with st.spinner('Training model...'):
                # Train model
                history = st.session_state.model.train(
                    X_train, y_train, X_test, y_test,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                st.session_state.trained = True
                
                # Plot training history
                st.markdown("### Training History")
                fig = st.session_state.model.plot_training_history()
                st.pyplot(fig)
                
                # Evaluate model
                st.markdown("### Model Evaluation")
                metrics = st.session_state.model.evaluate(X_test, y_test)
                st.session_state.metrics = metrics
                
                # Display metrics
                if forecast_horizon == 1:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{metrics['mse']:.2f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
                    col3.metric("MAE", f"{metrics['mae']:.2f}")
                    col4.metric("MAPE", f"{metrics['mape']:.2f}%")
                else:
                    st.markdown("#### Average Metrics Across All Forecast Steps")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{metrics['average']['mse']:.2f}")
                    col2.metric("RMSE", f"{metrics['average']['rmse']:.2f}")
                    col3.metric("MAE", f"{metrics['average']['mae']:.2f}")
                    col4.metric("MAPE", f"{metrics['average']['mape']:.2f}%")
                    
                    # Show metrics for each forecast step
                    st.markdown("#### Metrics by Forecast Step")
                    step_metrics = pd.DataFrame({
                        'Step': [f"Step {i+1}" for i in range(forecast_horizon)],
                        'MSE': [metrics[f'step_{i+1}']['mse'] for i in range(forecast_horizon)],
                        'RMSE': [metrics[f'step_{i+1}']['rmse'] for i in range(forecast_horizon)],
                        'MAE': [metrics[f'step_{i+1}']['mae'] for i in range(forecast_horizon)],
                        'MAPE': [metrics[f'step_{i+1}']['mape'] for i in range(forecast_horizon)]
                    })
                    st.dataframe(step_metrics)
                
                # Plot predictions
                st.markdown("### Prediction Results")
                if forecast_horizon == 1:
                    fig = st.session_state.model.plot_predictions(X_test, y_test)
                else:
                    fig = st.session_state.model.plot_forecast(X_test, y_test)
                st.pyplot(fig)

with predict_tab:
    st.markdown("### Energy Consumption Prediction")
    
    if not st.session_state.trained:
        st.warning("Please train the model first in the Model Training tab")
    else:
        st.markdown("### Interactive Prediction")
        
        if forecast_horizon == 1:
            st.markdown("#### Single-Step Prediction")
            
            # Make predictions
            predictions = st.session_state.model.predict(st.session_state.X_test)
            
            # Create interactive plot
            fig = go.Figure()
            
            # Get actual values
            y_test = st.session_state.y_test
            if forecast_horizon == 1:
                y_test_reshaped = y_test.reshape(-1, 1)
            else:
                original_shape = y_test.shape
                y_test_reshaped = y_test.reshape(-1, 1)
                
            # Inverse transform targets
            y_test_inv = st.session_state.model.scaler.inverse_transform(y_test_reshaped)
            
            # Reshape back to original shape if needed
            if forecast_horizon > 1:
                y_test_inv = y_test_inv.reshape(original_shape)
            
            # Plot actual vs predicted
            sample_size = min(100, len(predictions))
            
            fig.add_trace(go.Scatter(
                x=list(range(sample_size)),
                y=y_test_inv.flatten()[:sample_size],
                mode='lines',
                name='Actual Consumption'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(sample_size)),
                y=predictions.flatten()[:sample_size],
                mode='lines',
                name='Predicted Consumption'
            ))
            
            fig.update_layout(
                title='Energy Consumption Prediction',
                xaxis_title='Time Step',
                yaxis_title='Energy Consumption',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### Multi-Step Prediction")
            
            # Select sample to visualize
            sample_idx = st.slider(
                'Select sample to visualize',
                min_value=0,
                max_value=len(st.session_state.X_test) - 1,
                value=0
            )
            
            # Make predictions
            predictions = st.session_state.model.predict(st.session_state.X_test)
            
            # Get actual values
            y_test = st.session_state.y_test
            original_shape = y_test.shape
            y_test_reshaped = y_test.reshape(-1, 1)
                
            # Inverse transform targets
            y_test_inv = st.session_state.model.scaler.inverse_transform(y_test_reshaped)
            
            # Reshape back to original shape
            y_test_inv = y_test_inv.reshape(original_shape)
            
            # Create interactive plot
            fig = go.Figure()
            
            # Create x-axis values (forecast steps)
            steps = list(range(1, forecast_horizon + 1))
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=y_test_inv[sample_idx],
                mode='lines+markers',
                name='Actual Consumption'
            ))
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=predictions[sample_idx],
                mode='lines+markers',
                name='Predicted Consumption'
            ))
            
            fig.update_layout(
                title=f'Multi-step Forecast (Sample {sample_idx})',
                xaxis_title='Forecast Step',
                yaxis_title='Energy Consumption',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

with anomaly_tab:
    st.markdown("### Anomaly Detection in Energy Consumption")
    
    if st.session_state.data is None:
        st.warning("Please load energy consumption data first in the Data Analysis tab")
    else:
        st.markdown("""
        Anomaly detection identifies unusual patterns in energy consumption that might indicate:
        - Equipment malfunction
        - Unauthorized usage
        - Data recording errors
        - Unusual weather conditions
        - Special events
        """)
        
        # Anomaly detection parameters
        st.markdown("### Anomaly Detection Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider(
                'Window Size (Hours)',
                min_value=6,
                max_value=72,
                value=24,
                step=6
            )
        
        with col2:
            threshold = st.slider(
                'Anomaly Threshold (Standard Deviations)',
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5
            )
        
        if st.button('Detect Anomalies'):
            # Detect anomalies
            data_with_anomalies = st.session_state.model.detect_anomalies(
                st.session_state.data.copy(),
                window=window_size,
                threshold=threshold
            )
            
            # Plot data with anomalies highlighted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_with_anomalies.index,
                y=data_with_anomalies['energy_consumption'],
                mode='lines',
                name='Energy Consumption'
            ))
            
            # Highlight anomalies
            anomalies = data_with_anomalies[data_with_anomalies['is_anomaly'] == 1]
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['energy_consumption'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(
                title='Energy Consumption with Anomalies',
                xaxis_title='Date',
                yaxis_title='Energy Consumption'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomaly statistics
            st.markdown("### Anomaly Statistics")
            
            # Calculate anomaly percentage
            anomaly_percentage = len(anomalies) / len(data_with_anomalies) * 100
            
            # Calculate average consumption during anomalies vs normal
            avg_anomaly = anomalies['energy_consumption'].mean()
            avg_normal = data_with_anomalies[data_with_anomalies['is_anomaly'] == 0]['energy_consumption'].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")
            col2.metric("Avg. Anomaly Consumption", f"{avg_anomaly:.2f}")
            col3.metric("Avg. Normal Consumption", f"{avg_normal:.2f}")
            
            # Show anomalies in table
            st.markdown("### Detected Anomalies")
            st.dataframe(anomalies)

with explain_tab:
    st.markdown("### Understanding RNNs for Energy Consumption Prediction")
    
    st.markdown("""
    #### How RNNs Work for Energy Consumption Forecasting
    
    Recurrent Neural Networks (RNNs) are particularly well-suited for energy consumption prediction because:
    
    1. **Sequential Data Processing**: Energy consumption data is sequential, with patterns over time
    2. **Memory Capability**: RNNs can remember past consumption patterns
    3. **Variable Input Length**: Can handle different sequence lengths for prediction
    
    #### Types of RNNs Used in This Application
    
    1. **Long Short-Term Memory (LSTM)**
       - Specialized RNN that can learn long-term dependencies
       - Uses gates to control information flow
       - Effective for capturing daily, weekly, and seasonal patterns
    
    2. **Gated Recurrent Unit (GRU)**
       - Simplified version of LSTM with fewer parameters
       - Often performs similarly to LSTM but trains faster
       - Good for smaller datasets
    
    3. **Attention Mechanism**
       - Helps the model focus on relevant parts of the input sequence
       - Improves performance for longer sequences
       - Provides interpretability of which time steps are important
    
    #### Factors Affecting Energy Consumption
    
    This model can incorporate various factors that influence energy consumption:
    
    1. **Time-based Patterns**
       - Hour of day (peak vs. off-peak hours)
       - Day of week (weekday vs. weekend)
       - Month of year (seasonal variations)
       - Holidays and special events
    
    2. **Weather Conditions**
       - Temperature (heating and cooling needs)
       - Humidity
       - Cloud cover (lighting needs)
       - Precipitation
    
    3. **Building Characteristics**
       - Size and type
       - Insulation quality
       - HVAC system efficiency
       - Occupancy patterns
    
    #### Applications of Energy Consumption Prediction
    
    1. **Energy Management**
       - Optimize energy usage
       - Reduce peak demand
       - Plan for capacity needs
    
    2. **Cost Reduction**
       - Time-of-use pricing optimization
       - Demand response programs
       - Budget planning
    
    3. **Anomaly Detection**
       - Identify equipment malfunctions
       - Detect unauthorized usage
       - Find energy waste
    
    4. **Sustainability**
       - Reduce carbon footprint
       - Meet energy efficiency goals
       - Support renewable energy integration
    """)
    
    # Add diagram of LSTM/GRU architecture
    st.markdown("### RNN Architecture for Energy Forecasting")
    
    st.image("https://miro.medium.com/max/1400/1*qn9Lz0QQEsrSKTK4BCYYeQ.png", 
             caption="LSTM/GRU Architecture for Time Series Forecasting", 
             use_column_width=True)
    
    # Add explanation of multi-step forecasting
    st.markdown("### Multi-step Forecasting Approach")
    
    st.markdown("""
    This application supports both single-step and multi-step forecasting:
    
    1. **Single-step Forecasting**
       - Predicts energy consumption for the next time step only
       - More accurate for immediate predictions
       - Used for real-time monitoring
    
    2. **Multi-step Forecasting**
       - Predicts energy consumption for multiple future time steps
       - Allows for longer-term planning
       - Accuracy typically decreases with forecast horizon
       
    The multi-step approach uses a direct strategy where the model outputs multiple values at once, one for each step in the forecast horizon.
    """)
    
    # Add explanation of anomaly detection
    st.markdown("### Anomaly Detection Methodology")
    
    st.markdown("""
    The anomaly detection in this application uses a statistical approach:
    
    1. Calculate rolling mean and standard deviation over a window of time
    2. Compute z-score for each point (how many standard deviations from the mean)
    3. Flag points with z-scores exceeding a threshold as anomalies
    
    This method is effective for detecting:
    - Sudden spikes or drops in consumption
    - Sustained unusual patterns
    - Seasonal anomalies when compared to typical patterns
    """)

# Show selected model in sidebar
st.sidebar.markdown('---')
st.sidebar.subheader('🧠 Selected Model')

if model_type == 'LSTM':
    st.sidebar.markdown("""
    **LSTM Architecture:**
    - Long Short-Term Memory cells
    - 3 stacked LSTM layers
    - Dropout for regularization
    """)
elif model_type == 'GRU':
    st.sidebar.markdown("""
    **GRU Architecture:**
    - Gated Recurrent Units
    - 3 stacked GRU layers
    - Dropout for regularization
    """)

if use_attention:
    st.sidebar.markdown("""
    **Attention Mechanism:**
    - Self-attention on sequence
    - Helps focus on important time steps
    - Improves prediction accuracy
    """)

# Add disclaimer
st.sidebar.markdown('---')
st.sidebar.info("""
**Note**: This app is for educational purposes.
Actual energy forecasting may require additional features and domain expertise.
""")
