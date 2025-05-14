"""Unit tests for validation utilities."""

import unittest
from datetime import datetime
from utils.validation import (
    validate_stock_symbol,
    validate_date_range,
    validate_training_params
)

class TestValidation(unittest.TestCase):
    def test_stock_symbol_validation(self):
        # Test valid symbols
        self.assertTrue(validate_stock_symbol("AAPL")[0])
        self.assertTrue(validate_stock_symbol("MSFT")[0])
        self.assertTrue(validate_stock_symbol("GOOGL")[0])
        
        # Test invalid symbols
        self.assertFalse(validate_stock_symbol("")[0])
        self.assertFalse(validate_stock_symbol("123")[0])
        self.assertFalse(validate_stock_symbol("aapl")[0])
        self.assertFalse(validate_stock_symbol("TOO-LONG")[0])
    
    def test_date_range_validation(self):
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Test valid date ranges
        self.assertTrue(validate_date_range("2020-01-01", today)[0])
        self.assertTrue(validate_date_range("1970-01-01", "2020-12-31")[0])
        
        # Test invalid date ranges
        self.assertFalse(validate_date_range("2025-01-01")[0])  # Future date
        self.assertFalse(validate_date_range("1969-12-31")[0])  # Before 1970
        self.assertFalse(validate_date_range("2020-01-01", "2019-12-31")[0])  # End before start
        self.assertFalse(validate_date_range("invalid-date")[0])  # Invalid format
    
    def test_training_params_validation(self):
        # Test valid parameters
        self.assertTrue(validate_training_params(10, 32, 0.001)[0])
        self.assertTrue(validate_training_params(100, 512, 0.1)[0])
        
        # Test invalid parameters
        self.assertFalse(validate_training_params(0, 32, 0.001)[0])  # Invalid epochs
        self.assertFalse(validate_training_params(10, 0, 0.001)[0])  # Invalid batch size
        self.assertFalse(validate_training_params(10, 32, 0)[0])  # Invalid learning rate
        self.assertFalse(validate_training_params(10, 2048, 0.001)[0])  # Batch size too large
        self.assertFalse(validate_training_params(1500, 32, 0.001)[0])  # Too many epochs
        self.assertFalse(validate_training_params(10, 32, 2.0)[0])  # Learning rate too high

if __name__ == '__main__':
    unittest.main()
