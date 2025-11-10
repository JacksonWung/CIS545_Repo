# src/data_loader.py
import pandas as pd
import os
from .config_loader import config

class DataLoader:
    def __init__(self):
        self.config = config
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def load_data(self):
        """Load all data sources without cleaning"""
        print("Loading data...")
        
        price_data = self._load_price_data()
        news_data = self._load_news_data()
        fundamental_data = self._load_fundamental_data()
        
        return {
            'price_data': price_data, 
            'news_data': news_data,
            'fundamental_data': fundamental_data
        }
    
    def _load_price_data(self):
        """Load price data only"""
        price_config = self.config.get('data_sources.price_data')
        data_path = os.path.join(self.project_root, price_config['path'])
        
        print(f"Loading price data from: {data_path}")
        df = pd.read_excel(data_path, engine='openpyxl')
        print(f"Price data shape: {df.shape}")
        return df
    
    def _load_news_data(self):
        """Load news data only - no cleaning"""
        news_config = self.config.get('data_sources.news_data')
        data_path = os.path.join(self.project_root, news_config['path'])
        
        print(f"Loading news data from: {data_path}")
        df = pd.read_excel(data_path, engine='openpyxl')
        print(f"News data shape: {df.shape}")
        return df
    
    def _load_fundamental_data(self):
        """Load fundamental data"""
        fundamental_config = self.config.get('data_sources.fundamental_data')
        data_path = os.path.join(self.project_root, fundamental_config['path'])
        
        print(f"Loading fundamental data from: {data_path}")
        df = pd.read_excel(data_path, engine='openpyxl')
        print(f"Fundamental data shape: {df.shape}")
        return df