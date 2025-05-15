import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import tensorflow as tf

# Configure TensorFlow to handle GPU properly
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Metal GPU enabled: {physical_devices}")
        
        # Note: Mixed precision is disabled due to compatibility issues with CudnnRNN
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception as e:
    print(f"GPU configuration error: {e}")
    print("Using CPU instead")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Concatenate, GlobalAveragePooling1D, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle
import json

# Disable mixed precision due to compatibility issues with CudnnRNN
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

class TextGenerationModel:
    """Text Generation Model
    
    This model generates text using a recurrent neural network.
    """
    
    def __init__(self, max_words=10000, max_sequence_length=30, embedding_dim=100, 
                 use_attention=False, char_mode=False):
        """Initialize Text Generation Model
        
        Args:
            max_words (int): Maximum number of words in vocabulary
            max_sequence_length (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            use_attention (bool): Whether to use attention
            char_mode (bool): Whether to use character-level model
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.char_mode = char_mode
        self.model = None
        self.tokenizer = None
        self.history = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None
    
    def prepare_data(self, text_data):
        """Prepare data for training
        
        Args:
            text_data (str): Text data to prepare
            
        Returns:
            tuple: (X, y) training data
        """
        if self.char_mode:
            return self._prepare_data_char_level(text_data)
        else:
            return self._prepare_data_word_level(text_data)
    
    def _prepare_data_char_level(self, text_data):
        """Prepare data for character-level model
        
        Args:
            text_data (str): Text data to prepare
            
        Returns:
            tuple: (X, y) training data
        """
        # Get unique characters
        chars = sorted(list(set(text_data)))
        self.vocab_size = len(chars)
        
        # Create character to index mapping
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        
        # Create sequences
        sequences = []
        for i in range(0, len(text_data) - self.max_sequence_length):
            seq = text_data[i:i + self.max_sequence_length]
            target = text_data[i + self.max_sequence_length]
            sequences.append((seq, target))
        
        # Create training data
        X = np.zeros((len(sequences), self.max_sequence_length, self.vocab_size), dtype=np.bool_)
        y = np.zeros((len(sequences), self.vocab_size), dtype=np.bool_)
        
        for i, (seq, target) in enumerate(sequences):
            for t, char in enumerate(seq):
                X[i, t, self.char_to_idx[char]] = 1
            y[i, self.char_to_idx[target]] = 1
        
        return X, y
    
    def _prepare_data_word_level(self, text_data):
        """Prepare data for word-level model
        
        Args:
            text_data (str): Text data to prepare
            
        Returns:
            tuple: (X, y) training data
        """
        # Tokenize text
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts([text_data])
        
        # Create sequences
        sequences = []
        words = text_data.split()
        for i in range(0, len(words) - self.max_sequence_length):
            seq = words[i:i + self.max_sequence_length]
            target = words[i + self.max_sequence_length]
            sequences.append((seq, target))
        
        # Create training data
        X = []
        y = []
        
        for seq, target in sequences:
            X.append(self.tokenizer.texts_to_sequences([' '.join(seq)])[0])
            target_idx = self.tokenizer.texts_to_sequences([[target]])[0]
            if target_idx:  # Skip if target is not in vocabulary
                y.append(target_idx[0])
        
        # Filter out sequences with missing targets
        X = [x for i, x in enumerate(X) if i < len(y)]
        
        # Convert to numpy arrays
        X = np.array(X)
        
        # One-hot encode targets
        self.vocab_size = min(self.max_words, len(self.tokenizer.word_index) + 1)
        y_onehot = np.zeros((len(y), self.vocab_size))
        for i, target in enumerate(y):
            if target < self.vocab_size:
                y_onehot[i, target] = 1
        
        return X, y_onehot
    
    def build_lstm_model(self):
        """Build LSTM model for text generation
        
        Returns:
            Model: LSTM model
        """
        model = Sequential()
        
        if self.char_mode:
            # Character-level model
            model.add(LSTM(128, input_shape=(self.max_sequence_length, self.vocab_size), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(128))
            model.add(Dropout(0.2))
            model.add(Dense(self.vocab_size, activation='softmax'))
        else:
            # Word-level model
            model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length))
            model.add(LSTM(128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(128))
            model.add(Dropout(0.2))
            model.add(Dense(self.vocab_size, activation='softmax'))
        
        return model
    
    def build_gru_model(self):
        """Build GRU model for text generation
        
        Returns:
            Model: GRU model
        """
        model = Sequential()
        
        if self.char_mode:
            # Character-level model
            model.add(GRU(128, input_shape=(self.max_sequence_length, self.vocab_size), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(128))
            model.add(Dropout(0.2))
            model.add(Dense(self.vocab_size, activation='softmax'))
        else:
            # Word-level model
            model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length))
            model.add(GRU(128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(128))
            model.add(Dropout(0.2))
            model.add(Dense(self.vocab_size, activation='softmax'))
        
        return model
    
    def build_attention_model(self):
        """Build attention-based model for text generation
        
        Returns:
            Model: Attention-based model
        """
        if self.char_mode:
            # Character-level model with attention
            inputs = Input(shape=(self.max_sequence_length, self.vocab_size))
            lstm1 = LSTM(128, return_sequences=True)(inputs)
            lstm1 = Dropout(0.2)(lstm1)
            
            # Attention layer
            attention = tf.keras.layers.Attention()([lstm1, lstm1])
            
            # Combine attention with LSTM output
            concat = Concatenate()([lstm1, attention])
            
            # Global pooling
            avg_pool = GlobalAveragePooling1D()(concat)
            
            # Dense layers
            dense = Dense(128, activation='relu')(avg_pool)
            dense = Dropout(0.2)(dense)
            
            # Output layer
            outputs = Dense(self.vocab_size, activation='softmax')(dense)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
        else:
            # Word-level model with attention
            inputs = Input(shape=(self.max_sequence_length,))
            embedding = Embedding(self.vocab_size, self.embedding_dim)(inputs)
            
            lstm1 = LSTM(128, return_sequences=True)(embedding)
            lstm1 = Dropout(0.2)(lstm1)
            
            # Attention layer
            attention = tf.keras.layers.Attention()([lstm1, lstm1])
            
            # Combine attention with LSTM output
            concat = Concatenate()([lstm1, attention])
            
            # Global pooling
            avg_pool = GlobalAveragePooling1D()(concat)
            
            # Dense layers
            dense = Dense(128, activation='relu')(avg_pool)
            dense = Dropout(0.2)(dense)
            
            # Output layer
            outputs = Dense(self.vocab_size, activation='softmax')(dense)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_model(self, model_type='lstm'):
        """Build model based on model_type
        
        Args:
            model_type (str): Type of model to build
        
        Returns:
            Model: RNN model
        """
        if model_type == 'lstm':
            self.model = self.build_lstm_model()
        elif model_type == 'gru':
            self.model = self.build_gru_model()
        elif model_type == 'attention':
            self.model = self.build_attention_model()
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
        """Train model
        
        Args:
            X_train (array): Training data
            y_train (array): Training targets
            X_test (array): Testing data
            y_test (array): Testing targets
            epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
        
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
            
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def sample_with_temperature(self, preds, temperature=1.0):
        """Sample from predictions with temperature
        
        Args:
            preds (array): Predictions
            temperature (float): Temperature for sampling
            
        Returns:
            int: Sampled index
        """
        # Apply temperature
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample
        probas = np.random.multinomial(1, preds, 1)
        
        return np.argmax(probas)
    
    def generate_text_char_level(self, seed_text, length=100, temperature=1.0):
        """Generate text at character level
        
        Args:
            seed_text (str): Seed text to start generation
            length (int): Length of text to generate
            temperature (float): Temperature for sampling
            
        Returns:
            str: Generated text
        """
        if not self.char_mode:
            raise ValueError("Model was not trained at character level")
        
        # Ensure seed text is at least max_sequence_length
        if len(seed_text) < self.max_sequence_length:
            seed_text = seed_text.rjust(self.max_sequence_length)
        
        # Take last max_sequence_length characters as seed
        generated_text = seed_text[-self.max_sequence_length:]
        
        # Generate text
        for i in range(length):
            # Prepare input
            x = np.zeros((1, self.max_sequence_length, self.vocab_size))
            for t, char in enumerate(generated_text[-self.max_sequence_length:]):
                if char in self.char_to_idx:
                    x[0, t, self.char_to_idx[char]] = 1
            
            # Predict next character
            preds = self.model.predict(x, verbose=0)[0]
            
            # Sample with temperature
            next_index = self.sample_with_temperature(preds, temperature)
            next_char = self.idx_to_char[next_index]
            
            # Add to generated text
            generated_text += next_char
        
        return generated_text
    
    def generate_text_word_level(self, seed_text, length=50, temperature=1.0):
        """Generate text at word level
        
        Args:
            seed_text (str): Seed text to start generation
            length (int): Number of words to generate
            temperature (float): Temperature for sampling
            
        Returns:
            str: Generated text
        """
        if self.char_mode:
            raise ValueError("Model was not trained at word level")
        
        # Tokenize seed text
        seed_seq = self.tokenizer.texts_to_sequences([seed_text])[0]
        
        # Ensure seed sequence is at most max_sequence_length
        if len(seed_seq) > self.max_sequence_length:
            seed_seq = seed_seq[-self.max_sequence_length:]
        
        # Generate text
        generated_text = seed_text
        
        for i in range(length):
            # Pad sequence
            padded_seq = pad_sequences([seed_seq], maxlen=self.max_sequence_length)
            
            # Predict next word
            preds = self.model.predict(padded_seq, verbose=0)[0]
            
            # Sample with temperature
            next_index = self.sample_with_temperature(preds, temperature)
            
            # Convert index to word
            next_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == next_index:
                    next_word = word
                    break
            
            # Add to generated text
            if next_word:
                generated_text += " " + next_word
            
            # Update seed sequence
            seed_seq.append(next_index)
            if len(seed_seq) > self.max_sequence_length:
                seed_seq = seed_seq[-self.max_sequence_length:]
        
        return generated_text
    
    def generate_text(self, seed_text, length=100, temperature=1.0):
        """Generate text using the appropriate method
        
        Args:
            seed_text (str): Seed text to start generation
            length (int): Length of text to generate
            temperature (float): Temperature for sampling
            
        Returns:
            str: Generated text
        """
        if self.char_mode:
            return self.generate_text_char_level(seed_text, length, temperature)
        else:
            return self.generate_text_word_level(seed_text, length, temperature)
    
    def plot_history(self):
        """Plot training history
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax[0].plot(self.history.history['accuracy'])
        ax[0].plot(self.history.history['val_accuracy'])
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        ax[1].plot(self.history.history['loss'])
        ax[1].plot(self.history.history['val_loss'])
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Validation'], loc='upper left')
        
        return fig, ax
    
    def save_model(self, filepath):
        """Save model to file
        
        Args:
            filepath (str): Path to save model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save config
        config = {
            'max_words': self.max_words,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'use_attention': self.use_attention,
            'char_mode': self.char_mode,
            'vocab_size': self.vocab_size
        }
        
        if self.char_mode:
            config['char_to_idx'] = self.char_to_idx
            config['idx_to_char'] = {int(k): v for k, v in self.idx_to_char.items()}
        
        config_path = os.path.join(os.path.dirname(filepath), 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Save tokenizer
        if not self.char_mode and self.tokenizer is not None:
            tokenizer_json = self.tokenizer.to_json()
            tokenizer_path = os.path.join(os.path.dirname(filepath), 'tokenizer.json')
            with open(tokenizer_path, 'w') as f:
                f.write(tokenizer_json)
    
    def load_model(self, filepath):
        """Load model from file
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = tf.keras.models.load_model(filepath)
        
        # Load config
        config_path = os.path.join(os.path.dirname(filepath), 'config.json')
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.max_words = config['max_words']
        self.max_sequence_length = config['max_sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.use_attention = config['use_attention']
        self.char_mode = config['char_mode']
        self.vocab_size = config['vocab_size']
        
        if self.char_mode:
            self.char_to_idx = config['char_to_idx']
            self.idx_to_char = config['idx_to_char']
        else:
            # Load tokenizer
            tokenizer_path = os.path.join(os.path.dirname(filepath), 'tokenizer.json')
            with open(tokenizer_path, 'r') as f:
                tokenizer_json = f.read()
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
