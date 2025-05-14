import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

class MNISTModel:
    def __init__(self):
        """Initialize MNIST model with Metal GPU support"""
        # Enable Metal GPU
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"Metal GPU enabled: {physical_devices}")
        except:
            print("No GPU found, using CPU instead")
        
        self.model = None
        self.history = None
        
    def build_model(self):
        self.model = Sequential([
            # Simple but effective CNN
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                   input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten and Dense
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),  # Stronger dropout
            Dense(10, activation='softmax')
        ])
        
        # Use a more aggressive learning rate
        # Compile model with optimized settings
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def load_and_prepare_data(self):
        """Load and preprocess MNIST dataset"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize and reshape data for CNN
        x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
        x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
        
        # Use larger batch size for GPU
        BATCH_SIZE = 256
        
        return (x_train, y_train), (x_test, y_test)
    
    def train(self, epochs=10):  
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        (x_train, y_train), (x_test, y_test) = self.load_and_prepare_data()
        
        # Add random noise to make model more robust
        noise_factor = 0.1
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
        # Train with optimized settings for Metal GPU
        self.history = self.model.fit(
            x_train_noisy, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=256,  # Larger batch size for GPU
            verbose=1  # Show progress
        )
        
        return self.history
    
    def evaluate_detailed(self):
        """Perform detailed model evaluation"""
        # Load test data
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
        
        # Get predictions
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        class_report = classification_report(y_test, y_pred_classes)
        
        # Get some misclassified examples
        misclassified_idx = np.where(y_pred_classes != y_test)[0]
        misclassified_samples = []
        for idx in misclassified_idx[:10]:  # Get first 10 mistakes
            misclassified_samples.append({
                'image': x_test[idx],
                'true': y_test[idx],
                'pred': y_pred_classes[idx],
                'conf': y_pred[idx][y_pred_classes[idx]]
            })
        
        # Calculate per-class accuracy
        per_class_acc = []
        for i in range(10):
            mask = y_test == i
            class_acc = accuracy_score(y_test[mask], y_pred_classes[mask])
            per_class_acc.append(class_acc)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'misclassified_samples': misclassified_samples,
            'per_class_accuracy': per_class_acc,
            'history': self.history.history
        }
    
    def evaluate(self, x_test=None, y_test=None):
        """Evaluate model performance"""
        if x_test is None or y_test is None:
            _, (x_test, y_test) = self.load_and_prepare_data()
            
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, image):
        # Ensure we have a 2D grayscale image
        if len(image.shape) == 3 and image.shape[-1] > 1:
            # Convert RGB to grayscale if needed
            image = np.mean(image, axis=-1)
        
        # Threshold the image to make it more clean
        _, image = cv2.threshold(image.astype('uint8'), 127, 255, cv2.THRESH_BINARY)
        
        # Add padding to center the digit
        pad_size = 4
        image = np.pad(image, pad_size, mode='constant', constant_values=0)
        
        # Find the bounding box of the digit
        coords = cv2.findNonZero(image)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Crop to the digit plus some padding
            image = image[max(0, y-pad_size):min(image.shape[0], y+h+pad_size),
                         max(0, x-pad_size):min(image.shape[1], x+w+pad_size)]
        
        # Resize to 20x20 and pad to 28x28 to match MNIST style
        image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
        image = np.pad(image, ((4,4), (4,4)), mode='constant', constant_values=0)
        
        # Reshape and normalize
        image = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Get prediction
        prediction = self.model.predict(image, verbose=0)
        
        return prediction[0]
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    def save_model(self, path):
        """Save the model"""
        self.model.save(path)
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model"""
        instance = cls()
        instance.model = tf.keras.models.load_model(path)
        return instance
