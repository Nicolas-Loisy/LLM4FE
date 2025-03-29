# Configuration manager for orchestrator

import json
import os
from dotenv import load_dotenv
from utils.logger import logger

class Config:
    def __init__(self, config_path, env_path='.env'):
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        self.load_env()

    def load_env(self):
        """Load environment variables from .env file"""
        logger.info(f"Loading environment variables from {self.env_path}")
        try:
            load_dotenv(self.env_path)
            logger.debug(f"Environment variables loaded from {self.env_path}")
        except Exception as e:
            logger.exception(f"Error loading environment variables: {str(e)}")

    def get_env(self, key, default=None):
        """Get environment variable by key"""
        value = os.environ.get(key, default)
        return value

    def load_config(self):
        # Load configuration settings
        logger.info("Loading configuration...")
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.debug(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {str(e)}")
        except Exception as e:
            logger.exception(f"Error loading configuration: {str(e)}")
