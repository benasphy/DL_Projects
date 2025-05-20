# Set environment variable to disable TensorFlow plugins before any imports
import streamlit as st
import subprocess
import os

st.set_page_config(
    page_title="Deep Learning Projects",
    page_icon="🧠",
    layout="wide"
)

# Define project structure
projects = {
    "Feed Forward Neural Networks": {
        "MNIST Recognition": "FeedForwardNN/mnist_recognition/app.py",
        "Weather Forecasting": "FeedForwardNN/weather_forecasting/app.py"
    },
    "Autoencoders": {
        "Stock Price Prediction": "Autoencoders/stock_prediction/app.py",
        "Dimensionality Reduction": "Autoencoders/dimensionality_reduction/app.py"
    },
    "Convolutional Neural Networks": {
        "model_performance": "ConvolutionNN/model_performance/app.py"
    },
    "Recurrent Neural Networks": {
        "Bitcoin Price Prediction": "RecurrentNN/bitcoin_prediction/app.py",
        "Energy Consumption Prediction": "RecurrentNN/energy_consumption/app.py",
        "Sentiment Analysis": "RecurrentNN/sentiment_analysis/app.py",
        "Text Generation": "RecurrentNN/text_generation/app.py"
    },
    "Generative Adversarial Networks": {
        "MNIST Generation": "GenerativeAdversarialNetwork/app.py"
    },
    "Neural Evolution": {
        "Cartpole": "NEAT/app.py"
    },
    "Transformers": {
        "Toy Example": "Transformers/app.py"
    },
    "Graph Neural Networks": {
        "Toy Example": "GraphNeuralNetworks/app.py"
    },
    "Time Series Forecasting": {
        "LSTM Forecast": "TimeSeriesForecasting/app.py"
    }
}



def main():
    st.title("🧠 Deep Learning Projects Hub")
    
    # Sidebar for project selection
    st.sidebar.title("Project Selection")
    
    # Category selection
    category = st.sidebar.selectbox(
        "Select Category",
        list(projects.keys()),
        format_func=lambda x: f"📁 {x}"
    )
    
    # Project selection
    project = st.sidebar.selectbox(
        "Select Project",
        list(projects[category].keys()),
        format_func=lambda x: f"📊 {x}"
    )
    
    # Display project description
    st.markdown(f"### Selected Project: {project}")
    
    # Project descriptions
    descriptions = {
        "MNIST Recognition": """
        **Convolutional Neural Network for Digit Recognition**
        - Uses MNIST dataset
        - Achieves >98% accuracy
        - Features live testing and performance metrics
        """,
        "Weather Forecasting": """
        **Time Series Prediction with Neural Networks**
        - Predicts temperature and humidity
        - Uses historical weather data
        - Features interactive visualizations
        """,
        "Stock Price Prediction": """
        **LSTM Autoencoder for Stock Price Forecasting**
        - Predicts future stock prices
        - Uses historical market data
        - Features interactive predictions and evaluation
        """,
        "Dimensionality Reduction": """
        **Autoencoder for Data Compression**
        - Reduces high-dimensional data
        - Visualizes latent space
        - Features interactive data exploration
        """,
        "Bitcoin Price Prediction": """
        **RNN/LSTM Model for Cryptocurrency Forecasting**
        - Predicts Bitcoin price movements
        - Incorporates technical indicators (RSI, MACD, etc.)
        - Features attention mechanisms and multiple architectures
        - Interactive visualization of predictions and performance
        """,
        "Energy Consumption Prediction": """
        **RNN-based Energy Forecasting System**
        - Predicts energy consumption patterns
        - Identifies seasonal trends and anomalies
        - Incorporates weather and time features
        - Interactive visualization of predictions and patterns
        """,
        "Sentiment Analysis": """
        **RNN-based Text Classification**
        - Analyzes sentiment in text data (positive/negative/neutral)
        - Uses LSTM, BiLSTM, and attention mechanisms
        - Features word embedding visualization
        - Interactive prediction on custom text inputs
        """,
        "Text Generation": """
        **Creative Text Generation with RNNs**
        - Generates human-like text in various styles
        - Supports character-level and word-level generation
        - Temperature-based sampling for creativity control
        - Interactive generation with custom seed texts
        """,
        "model_performance": """
        **CNN Model Performance Analysis**
        - Evaluates and compares different CNN architectures
        - Visualizes model performance metrics
        - Analyzes training and validation curves
        - Interactive comparison of different models
        """,
        "MNIST Generation": """
        **Generative Adversarial Network for MNIST**
        - Generates new handwritten digit images
        - Trains a GAN on the MNIST dataset
        - Visualizes generated samples interactively
        """,
        "Cartpole": """
        **NEAT Neuroevolution for Cartpole**
        - Solves the Cartpole balancing task
        - Evolves neural networks using NEAT
        - Visualizes agent performance
        """,
        "Toy Example": """
        **Transformer or GNN Toy Example**
        - Minimal demonstration of transformer or graph neural network
        - Sequence-to-sequence or node classification tasks
        - Placeholder for future advanced demos
        """,
        "LSTM Forecast": """
        **LSTM for Time Series Forecasting**
        - Predicts future values from time series data
        - Uses LSTM neural networks
        - Interactive forecast visualization
        """
    }
    
    st.markdown(descriptions[project])
    
    # Launch button
    if st.button("🚀 Launch Project", type="primary"):
        project_path = os.path.join(os.path.dirname(__file__), projects[category][project])
        
        # Check if we're running in Streamlit Cloud
        if "STREAMLIT_CLOUD" in os.environ:
            st.warning("Running in Streamlit Cloud. No need to kill other apps.")
        else:
            # Only kill other apps if not in Streamlit Cloud
            try:
                subprocess.run(["pkill", "-f", "streamlit run"], check=True)
            except FileNotFoundError:
                st.warning("pkill not found. Skipping process cleanup.")
            except Exception as e:
                st.error(f"Error cleaning up processes: {str(e)}")
        
        # Special handling for text generation to avoid TensorFlow issues
        env = os.environ.copy()
        if project == "Text Generation":
            st.info("Using compatibility mode for Text Generation project due to TensorFlow plugin issues.")
            # Set environment variables to disable TensorFlow plugin loading
            env["TF_DISABLE_PLUGIN_LOADING"] = "1"
        
        # Launch the selected project
        python_path = sys.executable
        cmd = [python_path, "-m", "streamlit", "run", project_path, "--", "--server.port", "8501"]
        
        # Add --server.address=0.0.0.0 for Streamlit Cloud
        if "STREAMLIT_CLOUD" in os.environ:
            cmd.extend(["--", "--server.address=0.0.0.0"])
        
        process = subprocess.Popen(cmd, env=env)
        
        st.success(f"Launching {project}... Please wait a moment.")
        
        # Add iframe to embed the app directly
        st.components.v1.iframe("http://localhost:8501", height=800, scrolling=True)
        
        # Also provide a link to open in new tab
        st.markdown(f"""
        You can also open the project in a new tab: <a href="http://localhost:8501" target="_blank">http://localhost:8501</a>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
