"""Unit tests for stock prediction model."""

import unittest
import numpy as np
from Autoencoders.stock_prediction.model import StockPredictionAE

class TestStockPrediction(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model = StockPredictionAE(sequence_length=5, prediction_length=3)
    
    def test_model_initialization(self):
        """Test model initialization and architecture."""
        self.assertEqual(self.model.sequence_length, 5)
        self.assertEqual(self.model.prediction_length, 3)
        self.assertIsNotNone(self.model.model)
        
        # Check model layers
        layers = self.model.model.layers
        self.assertGreater(len(layers), 0)
        
        # Check input shape
        self.assertEqual(layers[0].input_shape[1:], (5, 1))
        
        # Check output shape
        self.assertEqual(layers[-1].output_shape[1:], (3, 1))
    
    def test_data_preprocessing(self):
        """Test data preprocessing functions."""
        # Create dummy data
        dates = np.arange('2020-01-01', '2020-01-11', dtype='datetime64[D]')
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        # Create sequences
        X, y = self.model.create_sequences(prices)
        
        # Check shapes
        self.assertEqual(X.shape[1:], (5, 1))  # sequence_length
        self.assertEqual(y.shape[1:], (3, 1))  # prediction_length
        
        # Check values
        expected_X_first = np.array([[100], [101], [102], [103], [104]])
        expected_y_first = np.array([[105], [106], [107]])
        np.testing.assert_array_equal(X[0], expected_X_first)
        np.testing.assert_array_equal(y[0], expected_y_first)
    
    def test_prediction(self):
        """Test model prediction functionality."""
        # Create dummy input sequence
        input_sequence = np.array([[100], [101], [102], [103], [104]])
        input_sequence = np.expand_dims(input_sequence, 0)  # Add batch dimension
        
        # Get prediction
        prediction = self.model.model.predict(input_sequence)
        
        # Check prediction shape
        self.assertEqual(prediction.shape, (1, 3, 1))  # (batch, prediction_length, features)
        
        # Check prediction values are reasonable
        self.assertTrue(np.all(prediction > 0))  # Stock prices should be positive
    
    def test_model_compilation(self):
        """Test model compilation and optimizer."""
        # Check optimizer
        self.assertIsNotNone(self.model.model.optimizer)
        
        # Check loss function
        self.assertEqual(self.model.model.loss, 'mse')
        
        # Check metrics
        self.assertTrue('mae' in self.model.model.metrics_names)

if __name__ == '__main__':
    unittest.main()
