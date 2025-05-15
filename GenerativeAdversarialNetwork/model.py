"""
Advanced GAN Model for MNIST digit generation
Inspired by FeedForwardNN/mnist_recognition structure
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class GANModel:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        self.history = {'d_loss': [], 'g_loss': []}

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')
        return gan

    def load_data(self):
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = np.expand_dims(x_train, axis=-1)
        return x_train

    def train(self, epochs=1000, batch_size=128, sample_interval=100):
        x_train = self.load_data()
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(1, epochs+1):
            # Train Discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_imgs = x_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)
            # Save history
            self.history['d_loss'].append(d_loss[0])
            self.history['g_loss'].append(g_loss if isinstance(g_loss, float) else g_loss[0])
            if epoch % sample_interval == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} | D loss: {d_loss[0]:.4f} | G loss: {g_loss:.4f}")
        return self.history

    def generate_samples(self, num_samples=10):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        return gen_imgs

    def plot_training_history(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(self.history['d_loss'], label='Discriminator Loss')
        ax.plot(self.history['g_loss'], label='Generator Loss')
        ax.set_title('GAN Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_generated_samples(self, num_samples=10):
        gen_imgs = self.generate_samples(num_samples)
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
        for i in range(num_samples):
            axes[i].imshow(gen_imgs[i].squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        return fig
