import logging
import logging.config
import threading
from pathlib import Path

class SingletonLogger:
    _instance = None
    _lock = threading.Lock() # Thread safe 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, logger_path=None):
        """
        Set up the logger using configuration file or basic setup.
        
        Args:
            logger_path (str): Path to logging.ini file.
        """
        try:
            if logger_path and Path(logger_path).exists():
                logging.config.fileConfig(logger_path)
                self.logger = logging.getLogger('LLM4FE')
                self.logger.info(f"Logger configured from {logger_path}")
        except Exception as e:
            print(f"Error loading logging config: {e}")
        
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)
    
    def exception(self, message):
        self.logger.exception(message)

def get_logger(config_path):
    return SingletonLogger(config_path)