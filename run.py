if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.orchestrator.orchestrator import Orchestrator, IterationType
    from src.utils.logger import init_logger
    import pprint
    
    logging_path = os.path.join(Path(__file__).parent, "data", "logs","logging.ini")
    init_logger(logger_path=str(logging_path))

    orchestrator = Orchestrator(config_path="data/configs/config.json")

    description = "This is a sample dataset with various features of health data and other, the target is the 'status' column."

    # Choose execution mode
    # USE_MULTIPLE_PROMPTS = True  # Set to False for single prompt execution
    USE_MULTIPLE_PROMPTS = False
    
    # Choose iteration type
    ITERATION_TYPE = IterationType.FIXED  # Options: FIXED, SCORE_IMPROVEMENT, PERCENTAGE_IMPROVEMENT
    # ITERATION_TYPE = IterationType.SCORE_IMPROVEMENT
    # ITERATION_TYPE = IterationType.PERCENTAGE_IMPROVEMENT
    
    if USE_MULTIPLE_PROMPTS:
        print("\n" + "="*70)
        print("RUNNING MULTI-PROMPT ORCHESTRATION")
        print("="*70)
        
        result = orchestrator.run_multiple_prompts(
            dataset_path="data/datasets/data.csv", 
            dataset_description=description, 
            target_column="target", 
            max_iterations=5,
            iteration_type=ITERATION_TYPE,
            min_improvement_percentage=1.0  # Only used for PERCENTAGE_IMPROVEMENT
        )
        
        print("\n" + "="*50)
        print("MULTI-PROMPT ORCHESTRATION RESULTS")
        print("="*50)
        print(f"Global Best Prompt: {result['global_best_prompt']}")
        print(f"Global Best Score: {result['global_best_score']:.4f}")
        print(f"Prompts Compared: {len(result['prompt_results'])}")
        print(f"Iterations Summary Path: {result['iterations_summary_path']}")
        
        print(f"\nPrompt Comparison Summary:")
        for prompt_name, summary in result['summary'].items():
            print(f"  {prompt_name}:")
            print(f"    Best Score: {summary['best_score']:.4f}")
            print(f"    Iterations: {summary['iterations']}")
        
        # print(f"\n\nBest Result Details:")
        # if result['global_best_result']:
        #     best = result['global_best_result']
        #     print(f"  Final Dataset: {best['final_dataset']}")
        #     print(f"  Best Dataset: {best['best_dataset']}")
        #     print(f"  Best Score: {best['best_score']:.4f}")
        #     print(f"  Total Iterations: {best['total_iterations']}")

    else:
        print("\n" + "="*70)
        print("RUNNING SINGLE PROMPT ORCHESTRATION")
        print(f"ITERATION TYPE: {ITERATION_TYPE.value.upper()}")
        print("="*70)
        
        if ITERATION_TYPE == IterationType.FIXED:
            result = orchestrator.run(
                dataset_path="data/datasets/data.csv", 
                dataset_description=description, 
                target_column="target", 
                max_iterations=3,
                iteration_type=ITERATION_TYPE
            )
        elif ITERATION_TYPE == IterationType.SCORE_IMPROVEMENT:
            result = orchestrator.run(
                dataset_path="data/datasets/data.csv", 
                dataset_description=description, 
                target_column="target", 
                max_iterations=10,
                iteration_type=ITERATION_TYPE
            )
        elif ITERATION_TYPE == IterationType.PERCENTAGE_IMPROVEMENT:
            result = orchestrator.run(
                dataset_path="data/datasets/data.csv", 
                dataset_description=description, 
                target_column="target", 
                max_iterations=10,
                iteration_type=ITERATION_TYPE,
                min_improvement_percentage=2.0  # Stop if improvement < 2%
            )
        
        print("\n" + "="*50)
        print("ORCHESTRATION RESULTS")
        print("="*50)
        print(f"Iteration Type: {ITERATION_TYPE.value}")
        print(f"Final Dataset: {result['final_dataset']}")
        print(f"Best Dataset: {result['best_dataset']}")
        print(f"Best Score: {result['best_score']:.4f} (Version {result['best_version']})")
        print(f"Total Transformations: {result['total_iterations']}")
        print("Score History:", [f"{score_dict['score']:.4f}" for score_dict in result['iteration_scores']])
        
        print(f"\nScore Evolution:")
        print(f"  Baseline (cleaned): {result['iteration_scores'][0]['score']:.4f}")
        for i, score_dict in enumerate(result['iteration_scores'][1:], 1):
            marker = " ★" if score_dict['score'] == result['best_score'] else ""
            print(f"  Version {i}: {score_dict['score']:.4f}{marker}")
    
    # print("\nDetailed Results:")
    # pprint.pprint(result)