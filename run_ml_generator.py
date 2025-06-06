from pathlib import Path

from src.ml_generator.ml_estimator import MachineLearningEstimator

if __name__ == "__main__":
    ml_estimator = MachineLearningEstimator(dataset_path=Path("data/tests/data.csv"), target_col="target")
    ml_score = ml_estimator.run()
    print(ml_score)
    ml_better = ml_estimator.score_better(2340)
    print(ml_better)




