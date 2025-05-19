
if __name__ == "__main__":
    from pathlib import Path
    from src.orchestrator.orchestrator import Orchestrator
    from src.utils.logger import init_logger

    # Logging initialization
    logging_path = Path(__file__).parent / "logging.ini"
    init_logger(logger_path=logging_path)

    orchestrator = Orchestrator(config_path="data/config.json")
    new_dataset = orchestrator.run(dataset_path="data/datasets/data.csv")
    print(new_dataset.head())