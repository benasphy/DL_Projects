# MNIST Handwritten Digit Recognition

A deep learning project that recognizes handwritten digits using a feedforward neural network. The model is trained on the MNIST dataset and provides predictions through an interactive Streamlit interface.

## Features
- Handwritten digit recognition (0-9)
- Real-time predictions
- Confidence scores for predictions
- Interactive drawing interface
- Model performance visualization
- Training history plots

## Model Architecture
- Input layer: 784 neurons (28x28 pixels)
- Hidden layers:
  - Dense layer (512 neurons, ReLU activation)
  - Dropout (0.2)
  - Dense layer (256 neurons, ReLU activation)
  - Dropout (0.2)
- Output layer: 10 neurons (softmax activation)

## Setup and Installation
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Draw a digit on the canvas
2. Click "Predict" to see the model's prediction
3. Use "Clear" to reset the canvas
4. View confidence scores for all digits

## Model Performance
- Training accuracy: ~99%
- Validation accuracy: ~98%
- Test accuracy: ~98%

## Files
- `app.py`: Streamlit web interface
- `model.py`: Neural network implementation
- `requirements.txt`: Project dependencies

## Deployment
To deploy on Streamlit Cloud:
1. Push to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy the app

## License
MIT License
