import os
import sys
import warnings

class TensorFlowNotAvailable:
    def __init__(self):
        self._warned = False
    
    def __getattr__(self, name):
        if not self._warned:
            warning_msg = """
            🚨 TensorFlow is not available in this environment.
            This project requires TensorFlow, which is not supported in Streamlit Cloud.
            Please try:
            1. Running this project locally
            2. Using a different deployment platform
            3. Selecting a project that doesn't require TensorFlow
            """
            warnings.warn(warning_msg, RuntimeWarning)
            self._warned = True
        raise ImportError("TensorFlow is not available in this environment")

# Add our compatibility layer to sys.modules
sys.modules['tensorflow'] = TensorFlowNotAvailable()
