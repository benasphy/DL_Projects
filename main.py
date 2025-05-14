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
        """
    }
    
    st.markdown(descriptions[project])
    
    # Launch button
    if st.button("🚀 Launch Project", type="primary"):
        project_path = os.path.join(os.path.dirname(__file__), projects[category][project])
        
        # Kill any running Streamlit apps
        subprocess.run(["pkill", "-f", "streamlit run"])
        
        # Launch the selected project
        cmd = ["/opt/miniconda3/envs/dl-env/bin/python", "-m", "streamlit", "run", 
               project_path, "--", "--server.port", "8501"]
        
        subprocess.Popen(cmd)
        
        st.success(f"Launching {project}... Please wait a moment.")
        
        # Add link to open in new tab
        st.markdown(f"""
        Project is running at: <a href="http://localhost:8501" target="_blank">http://localhost:8501</a>
        
        Click the link above or copy it to your browser if it doesn't open automatically.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
