# Sentiment Analysis with RNNs

This project implements a deep learning model for sentiment analysis using Recurrent Neural Networks (LSTM, BiLSTM, and Attention-based models) to classify text as positive, negative, or neutral.

## Features

- **Multiple RNN Architectures**: Choose between LSTM, BiLSTM, and Attention-based models
- **Text Preprocessing Pipeline**: Tokenization, stopword removal, lemmatization
- **Word Embeddings**: Dense vector representations of words
- **Multi-class Classification**: Support for binary or multi-class sentiment analysis
- **Attention Visualization**: Interpret which words influenced the model's decision
- **Batch Prediction**: Process multiple texts at once
- **Interactive Dashboard**: Streamlit web interface for data visualization and predictions

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
   - Load sentiment data or use sample data
   - Analyze data distribution and word frequencies
   - Train the model with your chosen configuration
   - Visualize model performance and confusion matrix
   - Make predictions on new text inputs

## Model Architecture

The project implements three main architectures:

1. **LSTM Model**:
   - Word embedding layer
   - LSTM layers with dropout
   - Dense output layer with sigmoid/softmax activation

2. **BiLSTM Model**:
   - Word embedding layer
   - Bidirectional LSTM layers with dropout
   - Dense output layer with sigmoid/softmax activation

3. **Attention Model**:
   - Word embedding layer
   - Bidirectional LSTM with attention mechanism
   - Global pooling layers
   - Dense output layer with sigmoid/softmax activation

## Data Pipeline

1. Text data is loaded from CSV or sample datasets
2. Text is preprocessed (lowercase, remove URLs, punctuation, stopwords)
3. Text is tokenized and converted to sequences
4. Sequences are padded to a fixed length
5. Data is split into training and testing sets

## Text Preprocessing

The text preprocessing pipeline includes:

1. Converting text to lowercase
2. Removing URLs, mentions, and hashtags
3. Removing punctuation and numbers
4. Removing stopwords (common words like "the", "is", etc.)
5. Lemmatization (reducing words to their base form)
6. Tokenization (splitting text into individual words)

## Requirements

See requirements.txt for a complete list of dependencies.
