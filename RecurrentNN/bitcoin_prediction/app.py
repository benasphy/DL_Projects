import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from model import BitcoinPredictionModel
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = BitcoinPredictionModel()
    st.session_state.trained = False
    st.session_state.data = None
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.test_dates = None
    st.session_state.predictions = None
    st.session_state.metrics = None
    st.session_state.forecast = None

# Sidebar
st.sidebar.title('🪙 Bitcoin Price Prediction')
st.sidebar.markdown('---')

# Model configuration
st.sidebar.subheader('🧠 Model Configuration')

model_type = st.sidebar.selectbox(
    'RNN Architecture',
    ['LSTM', 'GRU', 'BiLSTM'],
    index=0
)

use_attention = st.sidebar.checkbox('Use Attention Mechanism', value=False)

sequence_length = st.sidebar.slider(
    'Sequence Length (Days)',
    min_value=10,
    max_value=100,
    value=60,
    step=5
)

# Data configuration
st.sidebar.markdown('---')
st.sidebar.subheader('📊 Data Configuration')

period = st.sidebar.selectbox(
    'Data Period',
    ['1y', '2y', '5y', 'max'],
    index=1
)

include_features = st.sidebar.checkbox('Include Technical Indicators', value=True)

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
st.title('🪙 Bitcoin Price Prediction with RNNs')

# Create tabs
data_tab, train_tab, predict_tab, forecast_tab, explain_tab = st.tabs([
    "📈 Data Analysis", "🏋️ Model Training", "🔮 Price Prediction", 
    "🚀 Future Forecast", "📚 RNN Explanation"
])

with data_tab:
    st.markdown("### Bitcoin Price Data")
    
    if st.button('Load Bitcoin Data'):
        with st.spinner('Downloading Bitcoin data...'):
            # Download data
            data = st.session_state.model.download_data(period=period)
            
            # Add technical indicators if selected
            if include_features:
                data = st.session_state.model.add_technical_indicators(data)
                
            st.session_state.data = data
            
            # Show data info
            st.success(f"Downloaded {len(data)} days of Bitcoin price data")
            
            # Display data
            st.dataframe(data.tail(10))
            
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
            
            # Get the correct column names from the model's column mapping
            close_col = st.session_state.model.column_mapping.get('close', 'Close')
            volume_col = st.session_state.model.column_mapping.get('volume', 'Volume')
            
            # Plot price chart
            fig = px.line(
                data, 
                x=data.index, 
                y=close_col,
                title='Bitcoin Price History'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot volume (if available)
            if volume_col in data.columns:
                fig = px.bar(
                    data,
                    x=data.index,
                    y=volume_col,
                    title='Bitcoin Trading Volume'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if include_features:
                # Show technical indicators
                st.markdown("### Technical Indicators")
                
                # Create tabs for different indicator groups
                price_tab, momentum_tab, volatility_tab = st.tabs([
                    "Price Indicators", "Momentum Indicators", "Volatility Indicators"
                ])
                
                with price_tab:
                    # Get the correct close column name
                    close_col = st.session_state.model.column_mapping.get('close', 'Close')
                    
                    # Plot moving averages
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[close_col], name='Close Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_5'], name='5-day MA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='20-day MA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='50-day MA'))
                    fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot exponential moving averages
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[close_col], name='Close Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_12'], name='12-day EMA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_26'], name='26-day EMA'))
                    fig.update_layout(title='Exponential Moving Averages', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)
                
                with momentum_tab:
                    # Plot RSI
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
                    fig.add_shape(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30,
                                line=dict(color="green",width=2, dash="dash"))
                    fig.add_shape(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70,
                                line=dict(color="red",width=2, dash="dash"))
                    fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot MACD
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal Line'))
                    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Diff'], name='MACD Histogram'))
                    fig.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='Value')
                    st.plotly_chart(fig, use_container_width=True)
                
                with volatility_tab:
                    # Get the correct close column name
                    close_col = st.session_state.model.column_mapping.get('close', 'Close')
                    
                    # Plot Bollinger Bands
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[close_col], name='Close Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], name='Upper Band'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], name='Lower Band'))
                    fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot ATR
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['ATR'], name='ATR'))
                    fig.update_layout(title='Average True Range (ATR)', xaxis_title='Date', yaxis_title='ATR')
                    st.plotly_chart(fig, use_container_width=True)

with train_tab:
    st.markdown("### Train Bitcoin Price Prediction Model")
    
    if st.session_state.data is None:
        st.warning("Please load Bitcoin data first in the Data Analysis tab")
    else:
        if st.button('Prepare Data & Train Model'):
            with st.spinner('Preparing data...'):
                # Update model parameters
                st.session_state.model = BitcoinPredictionModel(
                    model_type=model_type.lower(),
                    sequence_length=sequence_length,
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
                
                # Get test dates - account for sequence_length offset
                # The test set starts after sequence_length from the end of the training set
                # So we need to adjust the dates accordingly
                sequence_length = st.session_state.model.sequence_length
                test_size = len(X_test)  # Use the actual length of X_test
                
                # Print debug info
                print(f"X_test shape: {X_test.shape}, test_size: {test_size}")
                print(f"Data index length: {len(st.session_state.data.index)}")
                
                # Calculate the correct starting index for test dates
                # We need to account for the sequence_length offset
                start_idx = len(st.session_state.data) - test_size - sequence_length
                end_idx = start_idx + test_size
                
                # Get the dates that correspond to the predictions (which are sequence_length steps ahead)
                st.session_state.test_dates = st.session_state.data.index[start_idx+sequence_length:end_idx+sequence_length]
                
                print(f"Test dates length: {len(st.session_state.test_dates)}")
                print(f"Test dates range: {st.session_state.test_dates[0]} to {st.session_state.test_dates[-1]}")
                
                # If there's still a mismatch, just use the last test_size dates
                if len(st.session_state.test_dates) != test_size:
                    print(f"Warning: Date length mismatch. Adjusting test_dates to match X_test length.")
                    st.session_state.test_dates = st.session_state.data.index[-test_size:]
                
                # Build model
                st.session_state.model.build_model()
                
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
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{metrics['mse']:.2f}")
                col2.metric("RMSE", f"{metrics['rmse']:.2f}")
                col3.metric("MAE", f"{metrics['mae']:.2f}")
                col4.metric("MAPE", f"{metrics['mape']:.2f}%")
                
                # Make predictions
                predictions = st.session_state.model.predict(X_test)
                st.session_state.predictions = predictions
                
                # Plot predictions
                st.markdown("### Prediction Results")
                fig = st.session_state.model.plot_predictions(X_test, y_test, st.session_state.test_dates)
                st.pyplot(fig)
                
                if include_features:
                    # Get feature importance
                    st.markdown("### Feature Importance")
                    importance = st.session_state.model.get_feature_importance()
                    
                    # Plot feature importance
                    fig = px.bar(
                        x=list(importance.keys()),
                        y=list(importance.values()),
                        title='Feature Importance',
                        labels={'x': 'Feature', 'y': 'Importance'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

with predict_tab:
    st.markdown("### Bitcoin Price Prediction Results")
    
    if not st.session_state.trained:
        st.warning("Please train the model first in the Model Training tab")
    else:
        # Display metrics
        st.markdown("### Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{st.session_state.metrics['mse']:.2f}")
        col2.metric("RMSE", f"{st.session_state.metrics['rmse']:.2f}")
        col3.metric("MAE", f"{st.session_state.metrics['mae']:.2f}")
        col4.metric("MAPE", f"{st.session_state.metrics['mape']:.2f}%")
        
        # Plot predictions
        st.markdown("### Prediction vs Actual Prices")
        
        # Get actual values
        y_test = st.session_state.y_test
        y_test = st.session_state.model.scaler.inverse_transform(y_test)
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.test_dates,
            y=y_test.flatten(),
            mode='lines',
            name='Actual Price'
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state.test_dates,
            y=st.session_state.predictions.flatten(),
            mode='lines',
            name='Predicted Price'
        ))
        fig.update_layout(
            title='Bitcoin Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate error
        error = y_test.flatten() - st.session_state.predictions.flatten()
        
        # Plot error
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.test_dates,
            y=error,
            mode='lines',
            name='Prediction Error'
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state.test_dates,
            y=np.zeros(len(error)),
            mode='lines',
            name='Zero Error',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='Prediction Error',
            xaxis_title='Date',
            yaxis_title='Error (USD)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot error distribution
        fig = px.histogram(
            error,
            title='Error Distribution',
            labels={'value': 'Error (USD)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

with forecast_tab:
    st.markdown("### Future Bitcoin Price Forecast")
    
    if not st.session_state.trained:
        st.warning("Please train the model first in the Model Training tab")
    else:
        # Forecast days
        forecast_days = st.slider(
            'Forecast Horizon (Days)',
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
        
        if st.button('Generate Forecast'):
            with st.spinner('Generating forecast...'):
                # Generate forecast
                forecast = st.session_state.model.forecast_future(days=forecast_days)
                st.session_state.forecast = forecast
                
                # Generate dates for forecast
                last_date = st.session_state.data.index[-1]
                
                # Create a proper date range using pandas
                from pandas.tseries.offsets import DateOffset
                forecast_dates = pd.date_range(
                    start=last_date + DateOffset(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                # Plot forecast
                fig = go.Figure()
                
                # Add historical data
                # Use the column mapping to get the correct 'Close' column
                close_col = None
                if hasattr(st.session_state.model, 'column_mapping') and 'close' in st.session_state.model.column_mapping:
                    close_col = st.session_state.model.column_mapping['close']
                else:
                    # Try to find a column that might be the close price
                    for col in st.session_state.data.columns:
                        if 'close' in str(col).lower():
                            close_col = col
                            break
                    # If still not found, use the first column as a fallback
                    if close_col is None and len(st.session_state.data.columns) > 0:
                        close_col = st.session_state.data.columns[0]
                        st.warning(f"Could not find a 'Close' column. Using {close_col} instead.")
                
                # Print debug info
                print(f"Using close column: {close_col}")
                print(f"Available columns: {st.session_state.data.columns.tolist()}")
                
                if close_col is not None:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.data.index[-60:],
                        y=st.session_state.data[close_col][-60:],
                        mode='lines',
                        name='Historical Price'
                    ))
                else:
                    st.error("Could not find a suitable price column in the data.")
                    st.stop()
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast,
                    mode='lines',
                    name='Forecasted Price',
                    line=dict(dash='dash')
                ))
                
                # Add vertical line at current date
                # Convert to string to avoid timestamp issues
                last_date_str = last_date.strftime('%Y-%m-%d')
                
                fig.add_shape(
                    type="line",
                    x0=last_date_str,
                    y0=0,
                    x1=last_date_str,
                    y1=1,
                    yref="paper",
                    line=dict(color="green", width=2, dash="dash"),
                )
                
                # Add annotation for the vertical line
                fig.add_annotation(
                    x=last_date_str,
                    y=1.05,
                    yref="paper",
                    text="Today",
                    showarrow=False,
                    font=dict(color="green")
                )
                
                fig.update_layout(
                    title=f'Bitcoin Price Forecast for Next {forecast_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price': forecast
                })
                st.dataframe(forecast_df)
                
                # Calculate potential return
                # Use the same close_col that was used for plotting
                if close_col is not None:
                    current_price = st.session_state.data[close_col].iloc[-1]
                    final_price = forecast[-1]
                    potential_return = (final_price - current_price) / current_price * 100
                else:
                    st.error("Could not calculate potential return: no suitable price column found.")
                    potential_return = 0
                
                # Display potential return
                st.markdown(f"### Potential Return in {forecast_days} Days")
                st.metric(
                    "Forecasted Return",
                    f"{potential_return:.2f}%",
                    delta=f"{final_price - current_price:.2f} USD"
                )
                
                # Warning about forecast accuracy
                st.warning("""
                **Disclaimer**: This forecast is for educational purposes only. Cryptocurrency prices are highly volatile
                and influenced by many factors not captured in this model. Do not use this forecast for financial decisions.
                """)

with explain_tab:
    st.markdown("### Understanding Recurrent Neural Networks for Time Series")
    
    st.markdown("""
    #### How RNNs Work for Time Series Prediction
    
    Recurrent Neural Networks (RNNs) are a class of neural networks designed to work with sequential data by maintaining an internal state (memory) that captures information about previous inputs in the sequence.
    
    **Key Components:**
    
    1. **Input Sequence**: A series of data points over time (e.g., daily Bitcoin prices)
    2. **Hidden State**: Internal memory that passes information from one step to the next
    3. **Output**: Prediction for the next time step
    
    **RNN Architectures Used in This App:**
    """)
    
    # Create tabs for different architectures
    lstm_tab, gru_tab, attention_tab = st.tabs([
        "LSTM", "GRU", "Attention Mechanism"
    ])
    
    with lstm_tab:
        st.markdown("""
        #### Long Short-Term Memory (LSTM)
        
        LSTM networks are a special kind of RNN designed to address the vanishing gradient problem, allowing them to learn long-term dependencies.
        
        **Key Features:**
        - **Memory Cell**: Long-term memory component
        - **Input Gate**: Controls what new information to store
        - **Forget Gate**: Controls what information to discard
        - **Output Gate**: Controls what information to output
        
        **Advantages:**
        - Can capture long-term dependencies
        - Resistant to vanishing gradient problem
        - Effective for various time series tasks
        
        **Architecture Diagram:**
        """)
        
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png", 
                 caption="LSTM Cell Architecture", 
                 use_column_width=True)
    
    with gru_tab:
        st.markdown("""
        #### Gated Recurrent Unit (GRU)
        
        GRU is a simplified version of LSTM with fewer parameters, making it faster to train while maintaining good performance.
        
        **Key Features:**
        - **Update Gate**: Combines input and forget gates from LSTM
        - **Reset Gate**: Controls how much of the previous state to forget
        
        **Advantages:**
        - Fewer parameters than LSTM (faster training)
        - Often performs similarly to LSTM
        - Better for smaller datasets or when computational resources are limited
        
        **Architecture Diagram:**
        """)
        
        st.image("https://miro.medium.com/max/1400/1*FpRS0C3EHQnELVaWRvb8bg.png", 
                 caption="GRU Cell Architecture", 
                 use_column_width=True)
    
    with attention_tab:
        st.markdown("""
        #### Attention Mechanism
        
        Attention mechanisms allow the model to focus on different parts of the input sequence when making predictions, improving performance for many tasks.
        
        **Key Features:**
        - **Query, Key, Value**: Components used to compute attention weights
        - **Attention Weights**: Determine the importance of each time step
        - **Context Vector**: Weighted sum of values based on attention weights
        
        **Advantages:**
        - Helps model focus on relevant parts of the sequence
        - Improves performance for long sequences
        - Provides interpretability (which time steps are important)
        
        **Architecture Diagram:**
        """)
        
        st.image("https://miro.medium.com/max/1400/1*Ua5DNHDSaYhZaGDPA9-bwg.png", 
                 caption="Attention Mechanism", 
                 use_column_width=True)
    
    st.markdown("""
    #### Technical Indicators for Bitcoin Price Prediction
    
    This app uses various technical indicators to improve prediction accuracy:
    
    1. **Trend Indicators**:
       - Moving Averages (MA)
       - Exponential Moving Averages (EMA)
       - Moving Average Convergence Divergence (MACD)
    
    2. **Momentum Indicators**:
       - Relative Strength Index (RSI)
       - Price Rate of Change
    
    3. **Volatility Indicators**:
       - Bollinger Bands
       - Average True Range (ATR)
    
    4. **Volume Indicators**:
       - Volume Moving Average
       - Volume Change
    
    #### Challenges in Bitcoin Price Prediction
    
    - **High Volatility**: Bitcoin prices can change dramatically in short periods
    - **External Factors**: News, regulations, market sentiment affect prices
    - **Market Manipulation**: Large traders can influence prices
    - **Limited History**: Cryptocurrency markets are relatively new
    
    #### Model Evaluation Metrics
    
    - **Mean Squared Error (MSE)**: Average of squared differences between predictions and actual values
    - **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same unit as the target
    - **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values
    - **Mean Absolute Percentage Error (MAPE)**: Average percentage difference between predictions and actual values
    """)

# Show model architecture in sidebar
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
elif model_type == 'BiLSTM':
    st.sidebar.markdown("""
    **Bidirectional LSTM Architecture:**
    - Processes sequences in both directions
    - 3 stacked BiLSTM layers
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
**Disclaimer**: This app is for educational purposes only. 
Cryptocurrency prices are highly volatile and predictions should not be used for financial decisions.
""")
