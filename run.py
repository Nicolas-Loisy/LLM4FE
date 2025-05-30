if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.orchestrator.orchestrator import Orchestrator
    from src.utils.logger import init_logger
    import pprint
    
    logging_path = os.path.join(Path(__file__).parent, "data", "logs","logging.ini")
    init_logger(logger_path=str(logging_path))

    orchestrator = Orchestrator(config_path="data/config.json")

    description = "This is a sample dataset with various features of health data and other, the target is the 'status' column."

    new_dataset = orchestrator.run(dataset_path="data/datasets/data.csv", dataset_description=description, target_column="status", iterations=1)
    pprint.pprint(new_dataset)