import json
import os
import re
import logging
from dotenv import load_dotenv

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


def get_config(config_path=None):
    """Get or initialize the singleton config instance"""
    if config_path is None and SingletonConfig._instance is None:
        raise ValueError("Config path must be provided on first initialization")
    
    return SingletonConfig(config_path)