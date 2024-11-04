import yaml
import os
from pathlib import Path

class ConfigHandler:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create directories for models and results if they don't exist"""
        dirs = [
            self.config['defaults']['model_dir'],
            self.config['defaults']['results_dir']
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def get_experiment_config(self, test_id):
        """Get configuration for a specific experiment"""
        if isinstance(test_id, int):
            test_id = f"test_{test_id}"
            
        if test_id not in self.config['experiments']:
            raise ValueError(f"No configuration found for {test_id}")
            
        return self.config['experiments'][test_id]
    
    def get_experiment_ids(self):
        """Get list of all experiment IDs"""
        return list(self.config['experiments'].keys())
    
    def get_model_path(self, test_id):
        """Get path for saving model"""
        if isinstance(test_id, int):
            test_id = f"test_{test_id}"
        return os.path.join(
            self.config['defaults']['model_dir'],
            f'sep_bk_nn_{test_id}.pth'
        )
    
    def get_results_path(self, test_id):
        """Get path for saving results"""
        if isinstance(test_id, int):
            test_id = f"test_{test_id}"
        return os.path.join(
            self.config['defaults']['results_dir'],
            f'results_{test_id}'
        )