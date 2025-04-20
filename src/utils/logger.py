import logging
import logging.config
import os
from pathlib import Path

class SingletonLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
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
            os.makedirs(Path(__file__).parent.parent.parent / "data" / "logs", exist_ok=True)
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


def init_logger(logger_path):
    get_logger(logger_path)

def get_logger(logger_path):
    return SingletonLogger(logger_path)