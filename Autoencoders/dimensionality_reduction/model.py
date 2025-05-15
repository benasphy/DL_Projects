import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.manifold import TSNE

class DimensionalityReductionAE:
    def __init__(self, encoding_dim=32):
        """Initialize Dimensionality Reduction Autoencoder with Metal GPU support"""
        # Enable Metal GPU but disable mixed precision for compatibility
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"Metal GPU enabled: {physical_devices}")
                
                # Note: Mixed precision is disabled due to compatibility issues
                # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception as e:
            print(f"GPU configuration error: {e}")
            print("Using CPU instead")
        
        self.encoding_dim = encoding_dim
        self.input_dim = 784  # 28x28 pixels
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.build_model()
    
    def build_model(self):
        """Build the autoencoder architecture with improved layers"""
        # Input layer
        input_img = Input(shape=(self.input_dim,))
        
        # Encoder layers with batch normalization and dropout
        x = Dense(512, activation='relu')(input_img)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        x = Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        x = Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Latent space
        encoded = Dense(self.encoding_dim, activation='relu', name='encoder_output')(x)
        
        # Decoder layers with batch normalization
        x = Dense(128, activation='relu')(encoded)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Output layer
        decoded = Dense(self.input_dim, activation='sigmoid')(x)
        
        # Create models
        self.model = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)
        
        # Create decoder model
        decoder_input = Input(shape=(self.encoding_dim,))
        x = Dense(128, activation='relu')(decoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        decoder_output = Dense(self.input_dim, activation='sigmoid')(x)
        self.decoder = Model(decoder_input, decoder_output)
        
        # Use Adam optimizer with learning rate schedule
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.9
            )
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['mae']
        )
    
    def train(self, epochs=10):
        """Train the autoencoder with data augmentation"""
        # Load and preprocess MNIST data
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        
        # Flatten and normalize the data
        x_train = x_train.reshape((len(x_train), self.input_dim))
        x_test = x_test.reshape((len(x_test), self.input_dim))
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        
        # Add noise for robustness
        noise_factor = 0.1
        x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
        x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
        x_train_noisy = tf.clip_by_value(x_train_noisy, 0., 1.)
        x_test_noisy = tf.clip_by_value(x_test_noisy, 0., 1.)
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train the model with larger batch size for GPU
        self.history = self.model.fit(
            x_train_noisy, x_train,
            epochs=epochs,
            batch_size=512,  # Increased for GPU
            shuffle=True,
            validation_data=(x_test_noisy, x_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def encode_data(self, data):
        """Encode data to lower dimension"""
        # Reshape and normalize if needed
        if len(data.shape) == 3:
            data = data.reshape((len(data), self.input_dim))
        if data.max() > 1:
            data = data.astype('float32') / 255.
        
        return self.encoder.predict(data)
    
    def decode_data(self, encoded_data):
        """Decode data back to original dimension"""
        decoded = self.model.predict(encoded_data)
        return decoded.reshape(-1, 28, 28)
    
    def visualize_latent_space(self, n_samples=1000):
        """Create t-SNE visualization of the latent space"""
        # Load test data
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
        
        # Encode the data
        x_test_reshaped = x_test.reshape((len(x_test), self.input_dim))
        x_test_reshaped = x_test_reshaped.astype('float32') / 255.
        encoded_data = self.encode_data(x_test_reshaped)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        encoded_tsne = tsne.fit_transform(encoded_data)
        
        return {
            'embeddings': encoded_tsne,
            'labels': y_test,
            'original_images': x_test
        }
    
    def evaluate_reconstruction(self, n_samples=10):
        """Evaluate reconstruction quality"""
        # Load test data
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test[:n_samples]
        
        # Get reconstructions
        x_test_reshaped = x_test.reshape((len(x_test), self.input_dim))
        x_test_reshaped = x_test_reshaped.astype('float32') / 255.
        reconstructed = self.model.predict(x_test_reshaped)
        reconstructed = reconstructed.reshape(-1, 28, 28)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(x_test/255. - reconstructed))
        
        return {
            'original': x_test,
            'reconstructed': reconstructed,
            'mse': mse,
            'history': self.history.history
        }
