import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from model import StockPredictionAE
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(
    page_title="Stock Price Prediction Autoencoder",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = StockPredictionAE()

if 'trained' not in st.session_state:
    st.session_state.trained = False

# Main content
st.title('📈 Stock Price Prediction Autoencoder')

# Create tabs
train_tab, prediction_tab, evaluation_tab = st.tabs([
    "🎓 Training", "🔮 Prediction", "📊 Evaluation"
])

# Training tab
with train_tab:
    st.markdown("### Model Training")
    
    if not st.session_state.trained:
        col1, col2 = st.columns(2)
        with col1:
            stock_symbol = st.text_input("Stock Symbol", "AAPL", help="e.g., AAPL for Apple, GOOGL for Google")
            epochs = st.slider("Number of epochs", 5, 50, 10)
        
        if st.button('🏃 Train Model', type='primary'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback to update progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f'Training progress: {(progress * 100):.1f}% (Epoch {epoch + 1}/{epochs})')
                    if logs:
                        status_text.text(f'Training progress: {(progress * 100):.1f}% (Epoch {epoch + 1}/{epochs}) - Loss: {logs["loss"]:.4f}')
            
            with st.spinner(f'Training model on {stock_symbol} data...'):
                history = st.session_state.model.train(
                    symbol=stock_symbol,
                    epochs=epochs,
                    callbacks=[ProgressCallback()]
                )
                st.session_state.trained = True
                st.session_state.stock_symbol = stock_symbol
                st.session_state.eval_results = st.session_state.model.evaluate_predictions(stock_symbol)
                progress_bar.progress(1.0)
                status_text.success('✅ Model trained successfully!')
                st.rerun()
    else:
        st.success(f'✅ Model is trained on {st.session_state.stock_symbol} stock data!')
        
        # Plot training history
        history = st.session_state.model.history.history
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history['loss'], name='Training Loss'))
        fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
        fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig)

# Prediction tab
with prediction_tab:
    if not st.session_state.trained:
        st.warning('⚠️ Please train the model first!')
    else:
        st.markdown("### Future Price Prediction")
        
        # Get latest stock data
        symbol = st.session_state.stock_symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        # Prepare last sequence
        last_sequence = df['Close'].values[-60:].reshape(-1, 1)
        
        # Get prediction
        prediction = st.session_state.model.predict_future(symbol, last_sequence)
        
        # Create dates for prediction
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(prediction)+1)[1:]
        
        # Plot results
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Prediction
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=prediction.flatten(),
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
        
        # Show prediction details
        st.markdown("#### Predicted Prices")
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': prediction.flatten()
        })
        st.dataframe(pred_df)

# Evaluation tab
with evaluation_tab:
    if not st.session_state.trained:
        st.warning('⚠️ Please train the model first!')
    else:
        st.markdown("### Model Evaluation")
        
        results = st.session_state.eval_results
        
        # Show average metrics
        avg_mse = np.mean([r['mse'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average MSE", f"{avg_mse:.2f}")
        with col2:
            st.metric("Average MAE", f"{avg_mae:.2f}")
        
        # Plot sample predictions
        st.markdown("#### Sample Predictions")
        
        for i, result in enumerate(results):
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=result['dates'],
                y=result['actual'],
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=result['dates'],
                y=result['predicted'],
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Sample Prediction {i+1}',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig)
            
            # Show metrics for this prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MSE", f"{result['mse']:.2f}")
            with col2:
                st.metric("MAE", f"{result['mae']:.2f}")
            
            st.markdown("---")
