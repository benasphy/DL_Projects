import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="RNN Projects Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define project paths and descriptions
PROJECTS = {
    "Bitcoin Price Prediction": {
        "folder": "bitcoin_prediction",
        "description": "Forecast Bitcoin prices using LSTM/GRU models with technical indicators and attention mechanisms.",
        "icon": "📈"
    },
    "Energy Consumption Prediction": {
        "folder": "energy_consumption",
        "description": "Predict energy consumption patterns with seasonal analysis and anomaly detection using RNNs.",
        "icon": "⚡"
    },
    "Sentiment Analysis": {
        "folder": "sentiment_analysis",
        "description": "Analyze sentiment in text data using LSTM, BiLSTM, and attention-based models.",
        "icon": "😊"
    },
    "Text Generation": {
        "folder": "text_generation",
        "description": "Generate creative text in various styles using character-level and word-level RNN models.",
        "icon": "📝"
    }
}

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .project-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
    st.session_state.current_tab = 'overview'

# Sidebar
with st.sidebar:
    st.title("🧠 RNN Projects")
    st.markdown("---")
    
    # Project selection dropdown
    selected_project = st.selectbox(
        "Select Project",
        list(PROJECTS.keys()),
        format_func=lambda x: f"{PROJECTS[x]['icon']} {x}"
    )
    
    if selected_project != st.session_state.current_project:
        st.session_state.current_project = selected_project
        st.session_state.current_tab = 'overview'
    
    st.markdown("---")
    
    # Tab selection
    tab_options = [
        "📋 Overview",
        "💻 Run App",
        "📊 Model",
        "📝 Code"
    ]
    
    for tab in tab_options:
        tab_key = tab.split()[1].lower()
        if st.button(tab, key=f"tab_{tab_key}", use_container_width=True):
            st.session_state.current_tab = tab_key
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About RNN Projects")
    st.markdown("""
    This dashboard provides access to four different Recurrent Neural Network projects, each demonstrating different applications of RNN architectures.
    
    Select a project from the dropdown menu and use the tabs to explore different aspects of each project.
    """)

# Main content
st.title(f"{PROJECTS[selected_project]['icon']} {selected_project}")
st.markdown(f"*{PROJECTS[selected_project]['description']}*")
st.markdown("---")

# Get project folder path
project_folder = PROJECTS[selected_project]['folder']
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), project_folder)

# Display content based on selected tab
if st.session_state.current_tab == 'overview':
    # Overview tab
    readme_path = os.path.join(project_path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme_content = f.read()
        st.markdown(readme_content)
    else:
        st.warning("README.md not found for this project.")

elif st.session_state.current_tab == 'run':
    # Run App tab
    st.header("Run Application")
    st.markdown(f"Run the {selected_project} application directly from here.")
    
    # Command to run the app
    run_command = f"cd {project_path} && streamlit run app.py"
    st.code(run_command, language="bash")
    
    if st.button("▶️ Launch App", type="primary"):
        st.info("Starting the application... Please wait.")
        # This is where we would normally launch the app, but we can't directly run it within this app
        st.markdown(f"**App is running at: [http://localhost:8501](http://localhost:8501)**")
        st.markdown("*Note: If the app doesn't open automatically, please run the command above in your terminal.*")

elif st.session_state.current_tab == 'model':
    # Model tab
    st.header("Model Architecture")
    
    model_path = os.path.join(project_path, "model.py")
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            model_code = f.read()
        
        # Display model summary
        st.markdown("### Model Summary")
        st.markdown(f"The {selected_project} model implements multiple RNN architectures including LSTM, GRU, and attention mechanisms.")
        
        # Display model code
        with st.expander("View Model Code"):
            st.code(model_code, language="python")
    else:
        st.warning("model.py not found for this project.")

elif st.session_state.current_tab == 'code':
    # Code tab
    st.header("Project Code")
    
    # Find all Python files in the project folder
    python_files = [f for f in os.listdir(project_path) if f.endswith(".py")]
    
    if python_files:
        selected_file = st.selectbox("Select File", python_files)
        file_path = os.path.join(project_path, selected_file)
        
        with open(file_path, "r") as f:
            file_code = f.read()
        
        st.code(file_code, language="python")
    else:
        st.warning("No Python files found in this project.")

# Footer
st.markdown("---")
st.markdown("*Created with Streamlit • RNN Projects Dashboard*")
