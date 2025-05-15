"""
Advanced Seq2Seq Transformer model for toy sequence-to-sequence data
Inspired by FeedForwardNN/mnist_recognition structure
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Seq2SeqTransformerModel:
    def __init__(self, vocab_size=50, embed_dim=32, num_heads=2, ff_dim=64, max_len=10):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        inputs = keras.Input(shape=(self.max_len,))
        x = layers.Embedding(self.vocab_size, self.embed_dim)(inputs)
        x = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)(x, x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(self.ff_dim, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def generate_toy_data(self, n_samples=1000):
        # Toy task: input is a random sequence, target is reversed sequence
        X = np.random.randint(1, self.vocab_size, size=(n_samples, self.max_len))
        y = np.flip(X, axis=1)
        y = y[..., np.newaxis]  # For sparse_categorical_crossentropy
        return X, y

    def train(self, epochs=10, batch_size=32):
        X, y = self.generate_toy_data(1000)
        X_val, y_val = self.generate_toy_data(200)
        self.history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return self.history

    def evaluate(self):
        X_test, y_test = self.generate_toy_data(200)
        return self.model.evaluate(X_test, y_test)

    def predict(self, input_seq):
        pred = self.model.predict(input_seq[np.newaxis, :])
        return np.argmax(pred, axis=-1)[0]

    def plot_training_history(self):
        if self.history is None:
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        plt.tight_layout()
        return fig
