"""Validation utilities for deep learning projects."""

import re
import yfinance as yf
from datetime import datetime, timedelta

def validate_stock_symbol(symbol):
    """Validate a stock symbol and check if it exists.
    
    Args:
        symbol (str): Stock symbol to validate (e.g., 'AAPL')
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid, empty string if valid
    """
    if not symbol:
        return False, "Stock symbol cannot be empty"
    
    if not re.match("^[A-Z]{1,5}$", symbol):
        return False, "Invalid symbol format. Must be 1-5 uppercase letters"
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return True, ""
        return False, f"Could not find stock data for symbol: {symbol}"
    except Exception as e:
        return False, f"Error validating stock symbol: {str(e)}"

def validate_date_range(start_date, end_date=None):
    """Validate a date range for time series data.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid, empty string if valid
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start > end:
            return False, "Start date cannot be after end date"
            
        if end > datetime.now():
            return False, "End date cannot be in the future"
            
        if start < datetime(1970, 1, 1):
            return False, "Start date cannot be before 1970"
            
        return True, ""
    except ValueError as e:
        return False, f"Invalid date format: {str(e)}"

def validate_training_params(epochs, batch_size, learning_rate):
    """Validate neural network training parameters.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid, empty string if valid
    """
    if not isinstance(epochs, int) or epochs < 1:
        return False, "Epochs must be a positive integer"
        
    if not isinstance(batch_size, int) or batch_size < 1:
        return False, "Batch size must be a positive integer"
        
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        return False, "Learning rate must be a positive float"
        
    if batch_size > 1024:
        return False, "Batch size too large, maximum is 1024"
        
    if epochs > 1000:
        return False, "Too many epochs, maximum is 1000"
        
    if learning_rate > 1.0:
        return False, "Learning rate too high, should be less than 1.0"
        
    return True, ""
