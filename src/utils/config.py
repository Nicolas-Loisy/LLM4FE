import json
import os
import re
import logging
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

class SingletonConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, config_path):
        self.config_path = config_path
        self.config = {}
        self._load_env()
        self._load_config()

    def _load_env(self):
        """Load environment variables from .env file"""
        try:
            load_dotenv()
            logger.debug("Environment variables loaded")
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            # Process environment variables in the config
            self.config = self._replace_env_variables(self.config)
            logger.debug(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    @staticmethod
    def _replace_env_var_in_string(value):
        # Fonction qui remplace les variables d'environnement dans une chaîne de texte
        env_var_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match):
            env_var = match.group(1)
            return os.getenv(
                env_var, match.group(0)
            )  # Remplace ou laisse ${VAR} tel quel

        return env_var_pattern.sub(replace, value)

    @staticmethod
    def _replace_env_variables(config):
        if isinstance(config, dict):
            for key, value in config.items():
                config[key] = SingletonConfig._replace_env_variables(value)
        elif isinstance(config, list):
            return [SingletonConfig._replace_env_variables(item) for item in config]
        elif isinstance(config, str):
            return SingletonConfig._replace_env_var_in_string(config)
        return config

    def get(self, key, default=None):
        """Get configuration value with environment variable override.
        If the config value is a dict, return the dict (do not override with env).
        """
        value = self.config.get(key, default)
        if isinstance(value, dict):
            return value
        env_value = os.environ.get(key)
        if env_value is not None:
            return env_value
        return value

    def get_file_content(self, key, default=None):
        """Get content of a file specified in the configuration.
        
        Args:
            key: The configuration key that contains the file path
            default: Default value if key is not found or file can't be loaded
            
        Returns:
            String with file content
        """
        file_path = self.get(key, default)
        if not file_path or not isinstance(file_path, str):
            return default
            
        try:
            # Get the project root directory (where config was loaded from)
            base_dir = Path(self.config_path).parent.parent.parent
            full_path = base_dir / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return default
    
    def get_file_content_by_path(self, file_path: str, default=None):
        """Get content of a file by direct path.
        
        Args:
            file_path: The file path relative to project root
            default: Default value if file can't be loaded
            
        Returns:
            String with file content
        """
        if not file_path or not isinstance(file_path, str):
            return default
            
        try:
            # Get the project root directory (where config was loaded from)
            base_dir = Path(self.config_path).parent.parent.parent
            full_path = base_dir / file_path
            
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return default

    def get_with_params(self, key: str, params=None, default=None):
        """Get configuration value and inject parameters.
        
        Args:
            key: The configuration key
            params: Dictionary of parameters to inject into the string
            default: Default value if key is not found
            
        Returns:
            String with parameters injected
        """
        # Special case for file keys
        if key.endswith("_file"):
            value = self.get_file_content(key, default)
        else:
            value = self.get(key, default)
            
        if not isinstance(value, str) or not params:
            return value
            
        try:
            # Use string format to inject parameters
            return value.format(**params)
        except KeyError as e:
            logger.error(f"Missing parameter for string formatting: {e}")
            return value
        except Exception as e:
            logger.error(f"Error during parameter injection: {str(e)}")
            return value


def get_config(config_path=None):
    """Get or initialize the singleton config instance"""
    if config_path is None and SingletonConfig._instance is None:
        raise ValueError("Config path must be provided on first initialization")
    
    return SingletonConfig(config_path)