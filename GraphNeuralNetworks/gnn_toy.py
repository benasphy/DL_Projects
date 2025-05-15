"""
Advanced GNN for node classification on toy graph data
Inspired by FeedForwardNN/mnist_recognition structure
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class GNNNodeClassifier:
    def __init__(self, input_dim=10, hidden_dim=16, num_classes=2, num_nodes=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        # Simple message-passing GNN: X -> Dense -> Dense
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = layers.Dense(self.hidden_dim, activation='relu')(inputs)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def generate_toy_graph(self):
        # Generate random node features and labels
        X = np.random.randn(self.num_nodes, self.input_dim)
        # Simple rule: sum(features) > 0 -> class 1, else 0
        y = (X.sum(axis=1) > 0).astype(int)
        return X, y

    def train(self, epochs=10, batch_size=16):
        X, y = self.generate_toy_graph()
        X_val, y_val = self.generate_toy_graph()
        self.history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return self.history

    def evaluate(self):
        X_test, y_test = self.generate_toy_graph()
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.argmax(pred, axis=-1)

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
