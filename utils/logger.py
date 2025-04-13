import logging
import logging.config
import os
from pathlib import Path

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._logger = None
        return cls._instance

    def setup(self, config_path=None):
        """
        Set up the logger using configuration file or basic setup
        
        Args:
            config_file (str): Path to logging.ini file
            level (str): Fallback log level if config_file not found
            log_file (str): Fallback log file path if config_file not found
        """
        if self._logger is not None:
            return
            
        # Try to load config file
        if not config_path:
            # Look for logging.ini in default locations
            search_paths = [
                Path(os.getcwd()) / "logging.ini",
                Path(__file__).parent.parent / "data" / "logs" / "logging.ini",
                Path(__file__).parent.parent / "logging.ini"
            ]
            
            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            # Configure from file
            try:
                logging.config.fileConfig(config_path)
                self._logger = logging.getLogger('LLM4FE')
                self._logger.info(f"Logger configured from {config_path}")
                return
            except Exception as e:
                print(f"Error loading logging config: {e}")
                # Fall through to basic config
        
    # Simple delegate methods for logging
    def debug(self, message):
        if self._logger is None: self.setup()
        self._logger.debug(message)
    
    def info(self, message):
        if self._logger is None: self.setup()
        self._logger.info(message)
    
    def warning(self, message):
        if self._logger is None: self.setup()
        self._logger.warning(message)
    
    def error(self, message):
        if self._logger is None: self.setup()
        self._logger.error(message)
    
    def critical(self, message):
        if self._logger is None: self.setup()
        self._logger.critical(message)
    
    def exception(self, message):
        if self._logger is None: self.setup()
        self._logger.exception(message)

# Create a singleton instance
logger = Logger()
