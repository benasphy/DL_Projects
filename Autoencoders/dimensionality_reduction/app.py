import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from model import DimensionalityReductionAE

st.set_page_config(
    page_title="Dimensionality Reduction Autoencoder",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = DimensionalityReductionAE()

if 'trained' not in st.session_state:
    st.session_state.trained = False

# Main content
st.title('📊 Dimensionality Reduction Autoencoder')

# Create tabs
train_tab, visualization_tab, reconstruction_tab = st.tabs([
    "🎓 Training", "🔍 Latent Space", "🎨 Reconstruction"
])

# Training tab
with train_tab:
    st.markdown("### Model Training")
    
    if not st.session_state.trained:
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Number of epochs", 5, 50, 10)
        with col2:
            encoding_dim = st.select_slider(
                "Encoding dimension",
                options=[2, 4, 8, 16, 32, 64],
                value=32
            )
        
        if st.button('🏃 Train Model', type='primary'):
            # Reinitialize model with new encoding dimension
            st.session_state.model = DimensionalityReductionAE(encoding_dim=encoding_dim)
            
            with st.spinner('Training model... This may take a few minutes...'):
                history = st.session_state.model.train(epochs=epochs)
                st.session_state.trained = True
                
                # Get evaluation results
                st.session_state.reconstruction = st.session_state.model.evaluate_reconstruction()
                st.session_state.latent_viz = st.session_state.model.visualize_latent_space()
                
                st.success('✅ Model trained successfully!')
                st.rerun()
    else:
        st.success('✅ Model is trained and ready!')
        
        # Plot training history
        history = st.session_state.reconstruction['history']
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history['loss'], name='Training Loss'))
        fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
        fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig)

# Latent space visualization tab
with visualization_tab:
    if not st.session_state.trained:
        st.warning('⚠️ Please train the model first!')
    else:
        st.markdown("### Latent Space Visualization")
        
        # Get t-SNE visualization data
        viz_data = st.session_state.latent_viz
        
        # Create interactive scatter plot
        fig = px.scatter(
            x=viz_data['embeddings'][:, 0],
            y=viz_data['embeddings'][:, 1],
            color=viz_data['labels'].astype(str),
            title='t-SNE visualization of latent space',
            labels={'color': 'Digit'},
            width=800,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Show sample images when hovering
        st.markdown("#### Sample Images from Latent Space")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            idx = np.random.randint(0, len(viz_data['labels']))
            col.image(
                viz_data['original_images'][idx],
                caption=f"Digit: {viz_data['labels'][idx]}",
                width=100
            )

# Reconstruction tab
with reconstruction_tab:
    if not st.session_state.trained:
        st.warning('⚠️ Please train the model first!')
    else:
        st.markdown("### Image Reconstruction")
        
        # Display reconstruction error
        st.metric("Mean Square Error", f"{st.session_state.reconstruction['mse']:.6f}")
        
        # Show original vs reconstructed images
        st.markdown("#### Original vs Reconstructed Images")
        
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            # Original
            axes[0, i].imshow(st.session_state.reconstruction['original'][i], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', pad=10)
            
            # Reconstructed
            axes[1, i].imshow(st.session_state.reconstruction['reconstructed'][i], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', pad=10)
        
        plt.tight_layout()
        st.pyplot(fig)
