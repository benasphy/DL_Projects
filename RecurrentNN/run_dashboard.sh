#!/bin/bash

# RNN Projects Dashboard Launcher
echo "RNN Projects Dashboard Launcher"
echo "==============================="

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "✗ Streamlit is not installed"
    read -p "Would you like to install Streamlit now? (y/n): " INSTALL
    if [[ $INSTALL == "y" || $INSTALL == "Y" ]]; then
        echo "Installing Streamlit..."
        pip install streamlit
        echo "✓ Streamlit installed successfully"
    else
        echo "Please install Streamlit manually with: pip install streamlit"
        exit 1
    fi
else
    echo "✓ Streamlit is installed"
fi

# Launch the dashboard
echo -e "\nLaunching RNN Projects Dashboard..."
streamlit run "$DIR/dashboard.py"
