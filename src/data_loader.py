# src/data_loader.py
import pandas as pd
import os
from .config_loader import config

class DataLoader:
    def __init__(self):
        self.config = config
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def load_data(self):
        """Load data without cleaning"""
        price_data = self._load_price_data()
        news_data = self._load_news_data()
        
        return {'price_data': price_data, 'news_data': news_data}
    
    def _load_price_data(self):
        """Load price data only"""
        price_config = self.config.get('data_sources.price_data')
        data_path = os.path.join(self.project_root, price_config['path'])
        
        df = pd.read_excel(data_path)
        return df
    
    def _load_news_data(self):
        """Load news data only"""
        news_config = self.config.get('data_sources.news_data')
        data_path = os.path.join(self.project_root, news_config['path'])
        
        df = pd.read_excel(data_path)
        return df