"""
Advanced Streamlit app for GNN Node Classification Toy Example
"""
import streamlit as st
from gnn_toy import GNNNodeClassifier
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="GNN Toy Node Classification", layout="wide")
st.title("🧬 GNN Toy: Node Classification Demo")

# Sidebar controls
epochs = st.sidebar.slider("Epochs", min_value=2, max_value=50, value=10, step=2)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
run_training = st.sidebar.button('🚀 Train GNN')

# Model session state
if 'gnn_model' not in st.session_state:
    st.session_state.gnn_model = GNNNodeClassifier()
    st.session_state.trained = False

# Training section
if run_training:
    with st.spinner(f'Training GNN for {epochs} epochs...'):
        st.session_state.gnn_model = GNNNodeClassifier()
        st.session_state.gnn_model.train(epochs=epochs, batch_size=batch_size)
        st.session_state.trained = True
    st.success('Training complete!')

# Tabs for visualization
train_tab, demo_tab = st.tabs(["📈 Training History", "🔢 Node Prediction Demo"])

with train_tab:
    st.markdown("### Training History (Accuracy & Loss)")
    if st.session_state.trained:
        fig = st.session_state.gnn_model.plot_training_history()
        if fig:
            st.pyplot(fig)
        else:
            st.info("No training history available.")
    else:
        st.info("Train the GNN to see learning curves.")

with demo_tab:
    st.markdown("### Try Node Classification Prediction")
    if st.session_state.trained:
        # Generate a random node feature vector
        input_dim = st.session_state.gnn_model.input_dim
        x = np.random.randn(1, input_dim)
        st.write(f"Input Node Features: {x.flatten().round(2).tolist()}")
        pred_class = st.session_state.gnn_model.predict(x)[0]
        st.write(f"Predicted Class: {int(pred_class)}")
        st.write(f"Expected Class (sum(features)>0): {int(x.sum() > 0)}")
    else:
        st.info("Train the GNN to try prediction.")

# Footer
st.markdown('---')
st.markdown('Made with 🧬 TensorFlow, Streamlit, and GNNs')
