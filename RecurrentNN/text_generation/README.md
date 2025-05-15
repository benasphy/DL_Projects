# Text Generation with RNNs

This project implements a creative text generation system using Recurrent Neural Networks (LSTM, GRU, and Attention-based models) to produce human-like text in various styles. The project includes an improved text generator with enhanced context-awareness and post-processing for more coherent and realistic outputs.

## Features

- **Multiple RNN Architectures**: Choose between LSTM, GRU, and Attention-based models
- **Tokenization Options**: Character-level or word-level text generation
- **Temperature-based Sampling**: Control the creativity and randomness of generated text
- **Interactive Text Generation**: Generate text from custom seed inputs
- **Multiple Literary Styles**: Support for different text styles and genres
- **Temperature Exploration**: Compare text generation with different temperature settings
- **Interactive Dashboard**: Streamlit web interface for model training and text generation
- **Improved Text Generator**: Enhanced context-aware generation with better paragraph structure and dialogue handling
- **Post-processing**: Automatic cleanup of common text generation issues for more polished output

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

3. In the web interface:
   - Configure model parameters in the sidebar
   - Load text data or use sample literary texts
   - Analyze text statistics and character frequencies
   - Train the model with your chosen configuration
   - Generate text with custom seeds and temperature settings
   - Explore the effect of temperature on text generation

## Model Architecture

The project implements three main architectures:

1. **LSTM Model**:
   - Character-level: One-hot encoded input → LSTM layers → Dense output
   - Word-level: Embedding layer → LSTM layers → Dense output

2. **GRU Model**:
   - Character-level: One-hot encoded input → GRU layers → Dense output
   - Word-level: Embedding layer → GRU layers → Dense output

3. **Attention Model**:
   - Character-level: One-hot encoded input → LSTM/GRU with attention → Dense output
   - Word-level: Embedding layer → LSTM/GRU with attention → Dense output

## Data Pipeline

1. Text data is loaded from a file or sample texts
2. Text is preprocessed based on tokenization level
3. Sequences are created from the text
4. For character-level: One-hot encoding is applied
5. For word-level: Tokenization and sequence conversion
6. Data is split into training and testing sets

## Temperature Sampling

The temperature parameter controls the randomness of the generated text:

1. Low temperature (0.1-0.5): More deterministic, repetitive output
2. Medium temperature (0.6-1.0): Balanced creativity and coherence
3. High temperature (1.1-2.0): More random, creative, potentially incoherent output

The sampling formula applies temperature (τ) to the model's output probabilities:
P(token) = exp(logit/τ) / Σ exp(logit/τ)

## Requirements

See requirements.txt for a complete list of dependencies.
