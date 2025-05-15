"""
Simple autoencoder for MNIST
"""
import tensorflow as tf
from tensorflow.keras import layers

# Encoder
encoder = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
])

# Decoder
decoder = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(784, activation='sigmoid'),
    layers.Reshape((28, 28))
])

if __name__ == "__main__":
    print("This is a minimal autoencoder scaffold. Add training code to proceed.")
