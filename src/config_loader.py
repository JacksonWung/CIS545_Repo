# src/config_loader.py
import yaml
import os

class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, 'config', 'parameters.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load YAML configuration file"""
        print(f"Loading config from: {self.config_path}")
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# Global configuration instance
config = ConfigLoader()