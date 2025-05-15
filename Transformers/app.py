"""
Advanced Streamlit app for Seq2Seq Transformer Toy Example
"""
import streamlit as st
from transformer_toy import Seq2SeqTransformerModel
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Transformer Toy Seq2Seq", layout="wide")
st.title("🧬 Transformer Toy: Sequence-to-Sequence Demo")

# Sidebar controls
epochs = st.sidebar.slider("Epochs", min_value=2, max_value=50, value=10, step=2)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
run_training = st.sidebar.button('🚀 Train Transformer')

# Model session state
if 'transformer_model' not in st.session_state:
    st.session_state.transformer_model = Seq2SeqTransformerModel()
    st.session_state.trained = False

# Training section
if run_training:
    with st.spinner(f'Training Transformer for {epochs} epochs...'):
        st.session_state.transformer_model = Seq2SeqTransformerModel()
        st.session_state.transformer_model.train(epochs=epochs, batch_size=batch_size)
        st.session_state.trained = True
    st.success('Training complete!')

# Tabs for visualization
train_tab, demo_tab = st.tabs(["📈 Training History", "🔢 Sequence Prediction Demo"])

with train_tab:
    st.markdown("### Training History (Accuracy & Loss)")
    if st.session_state.trained:
        fig = st.session_state.transformer_model.plot_training_history()
        if fig:
            st.pyplot(fig)
        else:
            st.info("No training history available.")
    else:
        st.info("Train the Transformer to see learning curves.")

with demo_tab:
    st.markdown("### Try Sequence-to-Sequence Prediction")
    if st.session_state.trained:
        # Generate a random test sequence
        vocab_size = st.session_state.transformer_model.vocab_size
        max_len = st.session_state.transformer_model.max_len
        input_seq = np.random.randint(1, vocab_size, size=(max_len,))
        st.write(f"Input Sequence: {input_seq.tolist()}")
        pred_seq = st.session_state.transformer_model.predict(input_seq)
        st.write(f"Predicted Output: {pred_seq.tolist()}")
        st.write(f"Expected Output (reverse): {input_seq[::-1].tolist()}")
    else:
        st.info("Train the Transformer to try prediction.")

# Footer
st.markdown('---')
st.markdown('Made with 🧬 TensorFlow, Streamlit, and Transformers')
