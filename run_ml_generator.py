from pathlib import Path

from src.utils.logger import init_logger
from src.ml_generator.ml_estimator import MachineLearningEstimator

if __name__ == "__main__":
    ml_estimator = MachineLearningEstimator(dataset_path=Path("data/tests/data.csv"), target_col="target")
    ml_score = ml_estimator.run()
    print(ml_score)




