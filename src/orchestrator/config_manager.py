# Configuration manager for orchestrator

import json

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}

    def load_config(self):
        # Load configuration settings
        print("Loading configuration...")
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("Configuration file not found.")

    def save_config(self):
        # Save configuration settings
        print("Saving configuration...")
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
