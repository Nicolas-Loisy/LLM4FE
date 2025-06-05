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

    result = orchestrator.run(
        dataset_path="data/datasets/data.csv", 
        dataset_description=description, 
        target_column="target", 
        iterations=3
    )
    
    print("\n" + "="*50)
    print("ORCHESTRATION RESULTS")
    print("="*50)
    print(f"Final Dataset: {result['final_dataset']}")
    print(f"Final Score: {result['final_score']:.4f}")
    print(f"Best Dataset: {result['best_dataset']}")
    print(f"Best Score: {result['best_score']:.4f} (Version {result['best_version']})")
    print(f"Total Transformations: {result['transformations_count']}")
    print(f"Score History: {[f'{score:.4f}' for score in result['score_history']]}")
    
    print(f"\nScore Evolution:")
    print(f"  Baseline (cleaned): {result['score_history'][0]:.4f}")
    for i, score in enumerate(result['score_history'][1:], 1):
        marker = " ★" if score == result['best_score'] else ""
        print(f"  Version {i}: {score:.4f}{marker}")
    
    print("\nDetailed Results:")
    pprint.pprint(result)