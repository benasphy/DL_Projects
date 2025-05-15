"""
Streamlit app for LSTM Time Series Forecasting
"""
import streamlit as st
from lstm_forecast import LSTMTimeSeriesForecaster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="LSTM Time Series Forecasting", layout="wide")
st.title("📈 LSTM Time Series Forecasting Demo")

# Sidebar controls
epochs = st.sidebar.slider("Epochs", min_value=2, max_value=100, value=10, step=2)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=2)
use_synthetic = st.sidebar.checkbox("Use Synthetic Sine Data", value=True)
run_training = st.sidebar.button('🚀 Train LSTM')

# Data upload
uploaded_file = st.sidebar.file_uploader("Upload CSV for Forecasting (single column)", type=["csv"]) if not use_synthetic else None
uploaded_series = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    uploaded_series = df.iloc[:, 0].values

# Model session state
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = LSTMTimeSeriesForecaster()
    st.session_state.trained = False
    st.session_state.last_series = None

# Training section
if run_training:
    with st.spinner(f'Training LSTM for {epochs} epochs...'):
        st.session_state.lstm_model = LSTMTimeSeriesForecaster()
        st.session_state.lstm_model.train(epochs=epochs, batch_size=batch_size, use_synthetic=use_synthetic, uploaded_series=uploaded_series)
        st.session_state.trained = True
        if uploaded_series is not None:
            st.session_state.last_series = uploaded_series
        elif use_synthetic:
            st.session_state.last_series = st.session_state.lstm_model.generate_sine_data(1100)
    st.success('Training complete!')

# Tabs for visualization
train_tab, forecast_tab = st.tabs(["📈 Training History", "🔮 Forecast Demo"])

with train_tab:
    st.markdown("### Training History (MAE & Loss)")
    if st.session_state.trained:
        fig = st.session_state.lstm_model.plot_training_history()
        if fig:
            st.pyplot(fig)
        else:
            st.info("No training history available.")
    else:
        st.info("Train the LSTM to see learning curves.")

with forecast_tab:
    st.markdown("### Try Interactive Forecasting")
    if st.session_state.trained and st.session_state.last_series is not None:
        seq_len = st.session_state.lstm_model.seq_len
        last_series = st.session_state.last_series
        input_seq = last_series[-seq_len:]
        steps = st.slider("Forecast Steps Ahead", min_value=1, max_value=50, value=10)
        pred = st.session_state.lstm_model.forecast(input_seq, steps=steps)
        fig2 = st.session_state.lstm_model.plot_forecast(last_series, np.concatenate([input_seq, pred]), start_idx=len(last_series)-seq_len)
        st.pyplot(fig2)
        st.write(f"Forecasted values: {np.round(pred, 3).tolist()}")
    else:
        st.info("Train the LSTM to try forecasting.")

# Footer
st.markdown('---')
st.markdown('Made with 🧬 TensorFlow, Streamlit, and LSTM')
