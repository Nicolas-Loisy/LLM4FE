from pathlib import Path

from src.ml_generator.ml_estimator import MachineLearningEstimator

if __name__ == "__main__":
    init_logger()
    
    ml_estimator = MachineLearningEstimator(dataset_path=Path("data/tests/data.csv"), target_col="target")
    ml_results = ml_estimator.run()
    
    print(f"Problem Type: {ml_results['problem_type']}")
    print(f"{ml_results['metric_name']}: {ml_results['score']:.4f}")
    ml_better = ml_estimator.score_better(2340)
    print(ml_better)




