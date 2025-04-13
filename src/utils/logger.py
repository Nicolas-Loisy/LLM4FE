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

    def setup(self, a_path=None):
        """
        Set up the logger using configuration file or basic setup
        
        Args:
            config_file (str): Path to logging.ini file
        """
        if self._logger is not None:
            return

        if not a_path:
            print(a_path)
            return print("Logger : The logging path does not exist.")
            
        # Try to load config file
        # if not a_path:
        #     # Look for logging.ini in default locations
        #     search_paths = [
        #         Path(os.getcwd()) / "logging.ini",
        #         Path(__file__).parent.parent / "data" / "logs" / "logging.ini",
        #         Path(__file__).parent.parent / "logging.ini"
        #     ]
            
        #     for path in search_paths:
        #         if path.exists():
        #             a_path = str(path)
        #             break

        if a_path and Path(a_path).exists():
            # Configure from file
            try:
                logging.config.fileConfig(a_path)
                self._logger = logging.getLogger('LLM4FE')
                self._logger.info(f"Logger configured from {a_path}")
                return
            except Exception as e:
                print(f"Error loading logging config: {e}")
                # Fall through to basic config
        
    # Simple delegate methods for logging
    def debug(self, message):
        self._logger.debug(message)

    def ji(self):
        print("fuck python")
    
    def info(self, message):
        self._logger.info(message)
    
    def warning(self, message):
        self._logger.warning(message)
    
    def error(self, message):
        self._logger.error(message)
    
    def critical(self, message):
        self._logger.critical(message)
    
    def exception(self, message):
        self._logger.exception(message)

# Create a singleton instance
logger = Logger()

# logger.setup()
# logger.info("Logger initialized.")