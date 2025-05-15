"""
Advanced Streamlit app for GAN MNIST digit generation
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import GANModel

st.set_page_config(page_title="MNIST GAN Generator", layout="wide")
st.title("🧬 GAN for MNIST Digit Generation")

# Sidebar controls
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=5000, value=1000, step=100)
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=2)
sample_interval = st.sidebar.slider("Sample Interval", min_value=50, max_value=1000, value=200, step=50)
num_samples = st.sidebar.slider("Generated Digits", min_value=5, max_value=20, value=10)

# GAN session state
if 'gan_model' not in st.session_state:
    st.session_state.gan_model = GANModel()
    st.session_state.trained = False

# Training section
if st.sidebar.button('🚀 Train GAN'):
    with st.spinner(f'Training GAN for {epochs} epochs...'):
        st.session_state.gan_model = GANModel()
        st.session_state.gan_model.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
        st.session_state.trained = True
    st.success('Training complete!')

# Tabs for visualization
train_tab, samples_tab = st.tabs(["📈 Training History", "🖼️ Generated Digits"])

with train_tab:
    st.markdown("### Training Loss Curves")
    if st.session_state.trained:
        fig = st.session_state.gan_model.plot_training_history()
        st.pyplot(fig)
    else:
        st.info("Train the GAN to see loss curves.")

with samples_tab:
    st.markdown(f"### Generated MNIST Digits ({num_samples})")
    if st.session_state.trained:
        fig = st.session_state.gan_model.plot_generated_samples(num_samples=num_samples)
        st.pyplot(fig)
    else:
        st.info("Train the GAN to generate digits.")

# Footer
st.markdown('---')
st.markdown('Made with 🧬 TensorFlow, Streamlit, and GANs')
