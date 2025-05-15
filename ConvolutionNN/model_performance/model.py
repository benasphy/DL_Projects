import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class CIFAR10Model:
    def __init__(self):
        """Initialize CIFAR10 CNN model with GPU support if available"""
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
        
        self.model = None
        self.history = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def build_model(self, model_type='standard'):
        """Build CNN model architecture
        
        Args:
            model_type (str): Type of model architecture ('standard', 'deep', or 'lightweight')
        """
        if model_type == 'standard':
            self.model = Sequential([
                # First convolutional block
                Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                
                # Second convolutional block
                Conv2D(64, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(64, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.3),
                
                # Third convolutional block
                Conv2D(128, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(128, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.4),
                
                # Fully connected layers
                Flatten(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
        elif model_type == 'deep':
            # Deeper architecture with more layers
            self.model = Sequential([
                # First block
                Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                BatchNormalization(),
                Conv2D(64, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                
                # Second block
                Conv2D(128, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(128, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.3),
                
                # Third block
                Conv2D(256, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(256, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(256, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.4),
                
                # Fourth block
                Conv2D(512, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(512, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                
                # Fully connected layers
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
        elif model_type == 'lightweight':
            # Lightweight model for faster training
            self.model = Sequential([
                Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                MaxPooling2D(pool_size=(2, 2)),
                
                Conv2D(32, (3, 3), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                
                Conv2D(64, (3, 3), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def load_and_prepare_data(self, use_data_augmentation=True):
        """Load and preprocess CIFAR-10 dataset
        
        Args:
            use_data_augmentation (bool): Whether to use data augmentation
            
        Returns:
            tuple: Training and test data
        """
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Normalize pixel values to be between 0 and 1
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # One-hot encode target values
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        if use_data_augmentation:
            # Create data generator for training with augmentation
            self.datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            )
            self.datagen.fit(x_train)
        
        return (x_train, y_train), (x_test, y_test)
    
    def train(self, epochs=20, batch_size=64, use_data_augmentation=True, model_type='standard'):
        """Train the CNN model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            use_data_augmentation (bool): Whether to use data augmentation
            model_type (str): Type of model architecture
            
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model(model_type)
            
        # Load and prepare data
        (x_train, y_train), (x_test, y_test) = self.load_and_prepare_data(use_data_augmentation)
        
        # Train the model
        if use_data_augmentation:
            # Train with data augmentation
            self.history = self.model.fit(
                self.datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=1
            )
        else:
            # Train without data augmentation
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=1
            )
        
        return self.history
    
    def evaluate(self, x_test=None, y_test=None):
        """Evaluate model performance
        
        Args:
            x_test (numpy.ndarray, optional): Test images
            y_test (numpy.ndarray, optional): Test labels
            
        Returns:
            tuple: Test loss and accuracy
        """
        if x_test is None or y_test is None:
            _, (x_test, y_test) = self.load_and_prepare_data(use_data_augmentation=False)
            
        return self.model.evaluate(x_test, y_test)
    
    def evaluate_detailed(self):
        """Perform detailed model evaluation
        
        Returns:
            dict: Detailed evaluation metrics and results
        """
        # Load test data
        _, (x_test, y_test) = self.load_and_prepare_data(use_data_augmentation=False)
        
        # Get predictions
        y_pred_prob = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names=self.class_names)
        
        # Get some misclassified examples
        misclassified_idx = np.where(y_pred != y_true)[0]
        misclassified_samples = []
        for idx in misclassified_idx[:10]:  # Get first 10 mistakes
            misclassified_samples.append({
                'image': x_test[idx],
                'true': self.class_names[y_true[idx]],
                'pred': self.class_names[y_pred[idx]],
                'conf': y_pred_prob[idx][y_pred[idx]]
            })
        
        # Calculate per-class accuracy
        per_class_acc = []
        for i in range(10):
            mask = y_true == i
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_acc.append(class_acc)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'misclassified_samples': misclassified_samples,
            'per_class_accuracy': per_class_acc,
            'per_class_names': self.class_names,
            'history': self.history.history if self.history else None
        }
    
    def predict(self, image):
        """Make prediction on a single image
        
        Args:
            image (numpy.ndarray): Input image (32x32x3)
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        # Ensure image has correct shape and normalization
        if image.shape != (32, 32, 3):
            raise ValueError("Image must be 32x32x3 in shape")
            
        # Normalize if needed
        if image.max() > 1.0:
            image = image.astype('float32') / 255.0
            
        # Reshape for model input
        image = np.expand_dims(image, axis=0)
        
        # Get prediction
        prediction = self.model.predict(image, verbose=0)
        
        return prediction[0]
    
    def visualize_feature_maps(self, image, layer_name=None):
        """Visualize feature maps for a given image
        
        Args:
            image (numpy.ndarray): Input image
            layer_name (str, optional): Name of layer to visualize
            
        Returns:
            tuple: Figure and feature maps
        """
        # Ensure image has correct shape
        if image.shape != (32, 32, 3):
            raise ValueError("Image must be 32x32x3 in shape")
            
        # Normalize if needed
        if image.max() > 1.0:
            image = image.astype('float32') / 255.0
            
        # Reshape for model input
        image = np.expand_dims(image, axis=0)
        
        # If no layer specified, use the first convolutional layer
        if layer_name is None:
            for layer in self.model.layers:
                if 'conv' in layer.name:
                    layer_name = layer.name
                    break
        
        # Create a model that will output feature maps
        layer_outputs = [layer.output for layer in self.model.layers if layer.name == layer_name]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Get feature maps
        activations = activation_model.predict(image)
        
        # Plot feature maps
        feature_maps = activations[0]
        n_features = min(16, feature_maps.shape[-1])  # Display at most 16 features
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].imshow(feature_maps[:, :, i], cmap='viridis')
            axes[i].set_title(f'Feature {i+1}')
            axes[i].axis('off')
            
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        return fig, feature_maps
    
    def plot_training_history(self):
        """Plot training history
        
        Returns:
            matplotlib.figure.Figure: Training history plot
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, normalize=False):
        """Plot confusion matrix
        
        Args:
            normalize (bool): Whether to normalize confusion matrix
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        # Get evaluation results
        results = self.evaluate_detailed()
        cm = results['confusion_matrix']
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return plt.gcf()
