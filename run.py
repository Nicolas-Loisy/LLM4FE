from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator
from src.utils.logger import init_logger

if __name__ == "__main__":
    
    # Logging initialization
    logging_path = Path(__file__).parent / "logging.ini"
    init_logger(logger_path=logging_path)

    orchestrator = Orchestrator()