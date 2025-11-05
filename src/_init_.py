# src/__init__.py
"""
CIS545 Quant Project - Multi-Modal Stock Price Prediction
Data loading and cleaning module
"""

from .config_loader import config
from .data_loader import DataLoader

__version__ = "0.1.0"
__all__ = ['config', 'DataLoader']