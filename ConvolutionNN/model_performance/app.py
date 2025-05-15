import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from model import CIFAR10Model
from tensorflow.keras.datasets import cifar10

# Page config
st.set_page_config(
    page_title="CNN Model Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = CIFAR10Model()
    st.session_state.trained = False
    st.session_state.eval_results = {}
    st.session_state.model.build_model()

# Sidebar
st.sidebar.title('📊 CNN Model Performance')
st.sidebar.markdown('---')

# Model configuration
st.sidebar.subheader('🧠 Model Configuration')

model_type = st.sidebar.selectbox(
    'CNN Architecture',
    ['standard', 'deep', 'lightweight'],
    index=0
)

data_augmentation = st.sidebar.checkbox('Use Data Augmentation', value=True)
epochs = st.sidebar.slider('Epochs', 1, 20, 5, 1)

# Training section in sidebar
st.sidebar.markdown('---')
st.sidebar.subheader('🎯 Model Training')
st.sidebar.markdown("""
- Dataset: CIFAR-10 (60,000 images, 10 classes)
- Features: 32x32 RGB images
- Architecture: Convolutional Neural Network
""")

if st.sidebar.button('Train CNN Model'):
    with st.spinner('Training CNN Model...'):
        # Build model with selected architecture
        st.session_state.model = CIFAR10Model()
        st.session_state.model.build_model(model_type=model_type)
        
        # Train model with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(epochs):
            status_text.text(f"Training epoch {i+1}/{epochs}...")
            progress_bar.progress((i + 1) / epochs)
        
        # Train model
        history = st.session_state.model.train(
            epochs=epochs,
            use_data_augmentation=data_augmentation
        )
        
        # Evaluate model
        st.session_state.eval_results = st.session_state.model.evaluate_detailed()
        st.session_state.trained = True
        
        # Show training results
        st.success('Model trained successfully! 🎉')
        status_text.empty()
        progress_bar.empty()
        
        # Show model performance metrics
        accuracy = st.session_state.eval_results['accuracy']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Model Type", model_type.capitalize())

# Main content
st.title('📊 CNN Model Performance Analysis')

# Display dataset info
st.markdown("### 📸 CIFAR-10 Dataset")
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
    - 50,000 training images
    - 10,000 testing images
    
    Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """)

with col2:
    # Load a few sample images
    (_, _), (x_test, y_test) = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    
    # Get one image from each class
    samples = []
    for i in range(10):
        idx = np.where(y_test == i)[0][0]
        samples.append(x_test[idx])
    
    for i, (ax, img) in enumerate(zip(axes, samples)):
        ax.imshow(img)
        ax.set_title(class_names[i])
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# Check if model is trained
if not st.session_state.trained:
    st.warning('⚠️ Please click "Train CNN Model" in the sidebar first before analyzing performance!')
    st.sidebar.success('👆 Click here to train the model')

# Show evaluation results if model is trained
if st.session_state.get('trained', False):
    # Create tabs for different visualizations
    train_tab, acc_tab, conf_tab, mistake_tab = st.tabs([
        '📈 Training History', '🏆 Accuracy Analysis',
        '📊 Confusion Matrix', '❌ Common Mistakes'
    ])
    
    with train_tab:
        st.markdown('### 📈 Training History')
        
        # Get history data
        history = st.session_state.eval_results['history']
        
        # Create training history plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history['accuracy'],
            name='Training Accuracy',
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            y=history['val_accuracy'],
            name='Validation Accuracy',
            mode='lines+markers'
        ))
        fig.update_layout(
            title='Model Accuracy Over Time',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            hovermode='x unified'
        )
        st.plotly_chart(fig)
        
        # Loss plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history['loss'],
            name='Training Loss',
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            y=history['val_loss'],
            name='Validation Loss',
            mode='lines+markers'
        ))
        fig.update_layout(
            title='Model Loss Over Time',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified'
        )
        st.plotly_chart(fig)
    
    with acc_tab:
        st.markdown('### 🏆 Accuracy Analysis')
        
        # Overall accuracy
        st.metric(
            'Overall Test Accuracy',
            f"{st.session_state.eval_results['accuracy']:.2%}"
        )
        
        # Per-class accuracy
        fig = px.bar(
            x=st.session_state.eval_results['per_class_names'],
            y=st.session_state.eval_results['per_class_accuracy'],
            labels={'x': 'Class', 'y': 'Accuracy'},
            title='Accuracy by Class'
        )
        st.plotly_chart(fig)
        
        # Show classification report
        st.markdown('#### Detailed Classification Report')
        st.code(st.session_state.eval_results['classification_report'])
    
    with conf_tab:
        st.markdown('### 📊 Confusion Matrix')
        
        # Plot confusion matrix
        fig = st.session_state.model.plot_confusion_matrix()
        st.pyplot(fig)
        
        # Add interpretation
        st.markdown("""
        #### Interpreting the Confusion Matrix:
        - Each row represents the actual class
        - Each column represents the predicted class
        - Numbers show how many test images were classified in each category
        - Diagonal elements represent correct predictions
        - Off-diagonal elements represent misclassifications
        """)
    
    with mistake_tab:
        st.markdown('### ❌ Analysis of Mistakes')
        
        # Show misclassified examples
        st.markdown("#### Misclassified Examples")
        
        misclassified = st.session_state.eval_results['misclassified_samples']
        
        # Display in grid
        cols = st.columns(5)
        for i, sample in enumerate(misclassified):
            col = cols[i % 5]
            col.image(
                sample['image'],
                caption=f"True: {sample['true']}\nPred: {sample['pred']}",
                use_column_width=True
            )
            
            if i % 5 == 4 and i < len(misclassified) - 1:
                cols = st.columns(5)
