import streamlit as st
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import cv2
from tensorflow.keras.datasets import mnist
from model import MNISTModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64

# Page config
st.set_page_config(
    page_title="MNIST Digit Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = MNISTModel()
    st.session_state.trained = False
    st.session_state.test_accuracy = 0.0
    st.session_state.model.build_model()

# Sidebar
st.sidebar.title('🔢 MNIST Digit Recognition')
st.sidebar.markdown('---')

# Training section in sidebar
if st.sidebar.button('Train CNN Model'):
    with st.spinner('Training Convolutional Neural Network...'):
        history = st.session_state.model.train(epochs=5)
        st.session_state.trained = True
        
        # Calculate test accuracy
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
        test_loss, test_accuracy = st.session_state.model.model.evaluate(x_test, y_test, verbose=0)
        st.session_state.test_accuracy = test_accuracy
        
        st.success(f'Model trained successfully! Test accuracy: {test_accuracy:.2%}')
        
        # Plot training history
        st.sidebar.subheader('📈 Training History')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training', color='#2ecc71')
        ax1.plot(history.history['val_accuracy'], label='Validation', color='#3498db')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training', color='#2ecc71')
        ax2.plot(history.history['val_loss'], label='Validation', color='#3498db')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        st.sidebar.pyplot(fig)

# Main content
st.title('📝 MNIST Digit Recognition - Model Performance')

# Initialize model if not exists
if 'model' not in st.session_state:
    st.session_state.model = MNISTModel()

# Train model if not trained
if not st.session_state.get('trained', False):
    if st.button('🏃 Train Model', type='primary'):
        with st.spinner('Training model... This may take a minute...'):
            history = st.session_state.model.train(epochs=10)
            st.session_state.trained = True
            st.success('✅ Model trained successfully!')
            
            # Get detailed evaluation
            st.session_state.eval_results = st.session_state.model.evaluate_detailed()
        st.rerun()

# Show evaluation results if model is trained
if st.session_state.get('trained', False):
    # Create tabs for different visualizations
    train_tab, acc_tab, conf_tab, mistake_tab, test_tab = st.tabs([
        '📈 Training History', '🏆 Accuracy Analysis',
        '📑 Confusion Matrix', '❌ Common Mistakes',
        '📊 Live Testing'
    ])

    

    
    with train_tab:
        st.markdown('### 📈 Training History')
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
            f"{st.session_state.eval_results['accuracy']:.2%}",
            delta='Good!' if st.session_state.eval_results['accuracy'] > 0.95 else None
        )
        
        # Per-class accuracy
        fig = px.bar(
            x=list(range(10)),
            y=st.session_state.eval_results['per_class_accuracy'],
            labels={'x': 'Digit', 'y': 'Accuracy'},
            title='Accuracy by Digit Class'
        )
        st.plotly_chart(fig)
        
        # Show classification report
        st.markdown('#### Detailed Classification Report')
        st.code(st.session_state.eval_results['classification_report'])
    
    with conf_tab:
        st.markdown('### 📑 Confusion Matrix')
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            st.session_state.eval_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Digit')
        plt.ylabel('True Digit')
        st.pyplot(fig)
        
        # Add interpretation
        st.markdown("""
        #### Interpreting the Confusion Matrix:
        - Each row represents the actual digit (0-9)
        - Each column represents the predicted digit
        - Numbers show how many test images were classified in each category
        - Diagonal elements represent correct predictions
        - Off-diagonal elements represent mistakes
        """)
    
    with mistake_tab:
        st.markdown('### ❌ Analysis of Mistakes')
        
        # Show misclassified examples
        misclassified = st.session_state.eval_results['misclassified_samples']
        cols = st.columns(5)
        for idx, sample in enumerate(misclassified[:5]):
            with cols[idx]:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(sample['image'].reshape(28, 28), cmap='gray')
                ax.axis('off')
                plt.title(f'True: {sample["true"]}\nPred: {sample["pred"]}\nConf: {sample["conf"]:.2%}')
                st.pyplot(fig)
        
        # Add analysis
        st.markdown("""
        #### Common Error Patterns:
        1. Similar-looking digits (e.g., 4 vs 9, 3 vs 8)
        2. Poorly written or ambiguous digits
        3. Unusual writing styles
        """)
    
    with test_tab:
        st.markdown('### 📊 Live Model Testing')
        
        # Add controls for testing
        col1, col2 = st.columns([2, 1])
        with col1:
            num_samples = st.slider('Number of random test samples', 10, 100, 20)
        with col2:
            if st.button('🎡 Run Test', type='primary'):
                # Load test data
                (_, _), (x_test, y_test) = mnist.load_data()
                x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
                
                # Randomly sample test cases
                indices = np.random.choice(len(x_test), num_samples, replace=False)
                x_sample = x_test[indices]
                y_sample = y_test[indices]
                
                # Get predictions
                y_pred = st.session_state.model.model.predict(x_sample, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_sample, y_pred_classes)
                st.metric(
                    'Test Accuracy',
                    f'{accuracy:.2%}',
                    delta=f'{(accuracy - 0.95):.2%} vs target' if accuracy > 0.95 else None
                )
                
                # Show results in a grid
                cols = st.columns(5)
                for idx in range(min(10, num_samples)):
                    with cols[idx % 5]:
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.imshow(x_sample[idx].reshape(28, 28), cmap='gray')
                        ax.axis('off')
                        color = 'green' if y_pred_classes[idx] == y_sample[idx] else 'red'
                        plt.title(
                            f'True: {y_sample[idx]}\nPred: {y_pred_classes[idx]}',
                            color=color
                        )
                        st.pyplot(fig)
                
                # Show accuracy over time
                if 'test_accuracies' not in st.session_state:
                    st.session_state.test_accuracies = []
                st.session_state.test_accuracies.append(accuracy)
                
                # Plot accuracy trend
                fig = px.line(
                    y=st.session_state.test_accuracies,
                    title='Test Accuracy Over Multiple Runs',
                    labels={'x': 'Test Run', 'y': 'Accuracy'}
                )
                fig.add_hline(
                    y=0.95,
                    line_dash='dash',
                    annotation_text='Target Accuracy (95%)',
                    line_color='red'
                )
                st.plotly_chart(fig)
                
                # Show confusion matrix for this batch
                conf_mat = confusion_matrix(y_sample, y_pred_classes)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    conf_mat,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax
                )
                plt.title('Confusion Matrix (Current Batch)')
                plt.xlabel('Predicted Digit')
                plt.ylabel('True Digit')
                st.pyplot(fig)

# Show model architecture in sidebar
st.sidebar.markdown('---')
st.sidebar.subheader('🧠 Model Architecture')
st.sidebar.code("""
Simplified CNN Architecture:
1. Conv2D(32, 3x3) → ReLU → MaxPool
2. Conv2D(64, 3x3) → ReLU → MaxPool
3. Dense(128) → ReLU → Dropout(0.5)
4. Dense(10) → Softmax
""")

st.sidebar.markdown("""
#### 🔍 Model Features:
- Smaller kernels (3x3) for better detail capture
- Strong regularization with 50% dropout
- Noise injection during training
- MNIST-style preprocessing
- Automatic digit centering
""")

# Footer
st.markdown('---')
st.markdown('Made with  using TensorFlow and Streamlit')
