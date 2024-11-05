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
    
    
def parse_test_ids(test_ids):
    """
    Parse test IDs including ranges to return a list of integers
    Examples:
        ['1,2,3'] -> [1, 2, 3]
        ['1-3'] -> [1, 2, 3]
        ['1-3', '5', '7-9'] -> [1, 2, 3, 5, 7, 8, 9]
    """
    parsed_ids = set()
    
    # Join all arguments and split by comma
    for part in ','.join(test_ids).split(','):
        if '-' in part:
            # Handle range (e.g., "1-3")
            try:
                start, end = map(int, part.split('-'))
                parsed_ids.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            # Handle single number
            try:
                parsed_ids.add(int(part))
            except ValueError:
                continue
    
    return sorted(list(parsed_ids))

def print_block_separator(text, style='default'):
    """Print a visually distinct block separator with text"""
    terminal_width = 80  # You can also use shutil.get_terminal_size().columns
    
    styles = {
        'default': {
            'top': '╔' + '═' * (terminal_width-2) + '╗',
            'middle': '║' + ' ' * (terminal_width-2) + '║',
            'bottom': '╚' + '═' * (terminal_width-2) + '╝'
        },
        'simple': {
            'top': '┌' + '─' * (terminal_width-2) + '┐',
            'middle': '│' + ' ' * (terminal_width-2) + '│',
            'bottom': '└' + '─' * (terminal_width-2) + '┘'
        },
        'hash': {
            'top': '#' * terminal_width,
            'middle': '#' + ' ' * (terminal_width-2) + '#',
            'bottom': '#' * terminal_width
        }
    }
    
    style_chars = styles.get(style, styles['default'])
    
    # Center the text
    text = f" {text} "  # Add spacing around text
    text_line = '║' + text.center(terminal_width-2) + '║'
    
    print("\n")  # Add some spacing
    print(style_chars['top'])
    print(text_line)
    print(style_chars['bottom'])
    print("\n")  # Add some spacing
