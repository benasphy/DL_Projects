import streamlit as st

def show_tf_placeholder():
    st.warning("""
    🚨 This project requires TensorFlow, which is not available in Streamlit Cloud.
    
    To run this project:
    1. Download the code from GitHub
    2. Install TensorFlow locally
    3. Run the project on your machine
    
    Available locally:
    - MNIST Recognition
    - Weather Forecasting
    - Stock Price Prediction
    - Dimensionality Reduction
    - And more...
    """)
    
    st.markdown("""
    ### How to Run Locally
    
    1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    ```
    
    2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
    3. Run the specific project:
    ```bash
    streamlit run <project-path>
    ```
    """)
    
    st.info("Note: TensorFlow projects work best when run locally due to hardware acceleration support.")
