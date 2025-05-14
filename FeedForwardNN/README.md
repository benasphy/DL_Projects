# Deep Learning Projects Collection

This repository contains various deep learning projects implemented using feedforward neural networks. Each project is contained in its own directory with its specific implementation, requirements, and documentation.

## Projects

### 1. MNIST Handwritten Digit Recognition
- Implementation of a feedforward neural network for digit recognition
- Uses the MNIST dataset
- Achieves high accuracy in digit classification
- Interactive Streamlit web interface

### 2. Weather Forecasting
- Weather prediction using historical weather data
- Multiple input features including temperature, humidity, pressure
- Predicts temperature for the next day
- Interactive Streamlit web interface

## Project Structure
```
FeedForwardNN/
├── mnist_recognition/
│   ├── app.py
│   ├── model.py
│   ├── requirements.txt
│   └── README.md
└── weather_forecasting/
    ├── app.py
    ├── model.py
    ├── requirements.txt
    └── README.md
```

## Setup
Each project has its own requirements.txt file. To run a specific project:

1. Navigate to the project directory
2. Create a virtual environment (recommended)
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment
All projects are deployable to Streamlit Cloud. Each project's README contains specific deployment instructions.

## Technologies Used
- Python 3.10+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pandas
- Scikit-learn
