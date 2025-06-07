import os
from pathlib import Path

from src.utils.logger import init_logger
from src.ml_generator.ml_estimator import MachineLearningEstimator

if __name__ == "__main__":
    logging_path = os.path.join(Path(__file__).parent, "data", "logs","logging.ini")
    init_logger(logger_path=str(logging_path))
    
    # Classification Example Test
    ml_estimator = MachineLearningEstimator(dataset_path=Path("data/datasets/dataset_classification_ml_test.csv"), target_col="target")
    ml_results = ml_estimator.run()
    
    print(f"Problem Type: {ml_results['problem_type']}")
    print(f"{ml_results['metric_name']}: {ml_results['score']:.4f}")
    ml_better = ml_estimator.get_best_score(ml_results['score'], 0.1, ml_results['problem_type'])
    print(ml_better)

    # Regression Example Test
    # ml_estimator = MachineLearningEstimator(dataset_path=Path("data/datasets/dataset_regression_ml_test.csv"), target_col="price")
    # ml_results = ml_estimator.run()
    
    # print(f"Problem Type: {ml_results['problem_type']}")
    # print(f"{ml_results['metric_name']}: {ml_results['score']:.4f}")
    # ml_better = ml_estimator.get_best_score(ml_results['score'], 2340000, ml_results['problem_type'])
    # print(ml_better)




