import os
import subprocess
import sys

def main():
    """
    Launcher script for the RNN Projects Dashboard.
    This script checks if Streamlit is installed and launches the dashboard.
    """
    print("RNN Projects Dashboard Launcher")
    print("===============================")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✓ Streamlit is installed")
    except ImportError:
        print("✗ Streamlit is not installed")
        install = input("Would you like to install Streamlit now? (y/n): ")
        if install.lower() == 'y':
            print("Installing Streamlit...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✓ Streamlit installed successfully")
        else:
            print("Please install Streamlit manually with: pip install streamlit")
            return
    
    # Launch the dashboard
    print("\nLaunching RNN Projects Dashboard...")
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    
    # Run streamlit in a subprocess
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("\nTo manually launch the dashboard, run:")
        print(f"streamlit run {dashboard_path}")

if __name__ == "__main__":
    main()
