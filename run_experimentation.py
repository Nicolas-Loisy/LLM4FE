if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.orchestrator.orchestrator import Orchestrator, IterationType
    from src.utils.logger import init_logger
    import matplotlib.pyplot as plt
    import numpy as np
    
    logging_path = os.path.join(Path(__file__).parent, "data", "logs","logging.ini")
    init_logger(logger_path=str(logging_path))

    orchestrator = Orchestrator(config_path="data/configs/config.json")

    # dataset_path = "data/datasets/data.csv"
    # description = "This is a sample dataset with various features of health data and other, the target is the 'status' column."

    dataset_path = "data/datasets/Loan_approval_dataset.csv"
    description = "The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. This dataset is commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features. Columns: loan_id - Loan Id, no_of_dependents - Number of Dependents of the Applicant, education - Education of the Applicant, self_employed - Employment Status of the Applicant, income_annum - Annual Income of the Applicant, loan_amount - Loan Amount, loan_term - Loan Term in Years, cibil_score - Credit Score, residential_assets_value - Residential assets, commercial_assets_value - Commercial assets, luxury_assets_value - Luxury assets, bank_asset_value - Bank assets, loan_status - Loan Approval Status."
    target = "loan_status"


    # Choose execution mode
    USE_MULTIPLE_PROMPTS = True  # Set to False for single prompt execution
    # USE_MULTIPLE_PROMPTS = False
    
    # Choose iteration type
    ITERATION_TYPE = IterationType.FIXED
    if USE_MULTIPLE_PROMPTS:
        print("\n" + "="*70)
        print("RUNNING MULTI-PROMPT ORCHESTRATION")
        print("="*70)
        
        result = orchestrator.run_multiple_prompts(
            dataset_path=dataset_path, 
            dataset_description=description, 
            target_column=target, 
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
        
        # Add visualizations
        print(f"\nGenerating visualizations...")
        
        def plot_score_evolution(result):
            """Plot score evolution by iteration for each prompt type"""
            plt.figure(figsize=(12, 8))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            for i, (prompt_name, prompt_result) in enumerate(result['prompt_results'].items()):
                if 'iteration_scores' in prompt_result and prompt_result['iteration_scores']:
                    iterations = [score['iteration'] for score in prompt_result['iteration_scores']]
                    scores = [score['score'] for score in prompt_result['iteration_scores']]
                    
                    # Clean prompt name
                    clean_name = prompt_name.replace('_template.txt', '')
                    
                    color = colors[i % len(colors)]
                    plt.plot(iterations, scores, 
                            marker='o', linewidth=2.5, markersize=6,
                            label=clean_name, color=color)
                    
                    # Add best score marker
                    # best_score = max(scores)
                    # best_iteration = iterations[scores.index(best_score)]
                    # plt.scatter(best_iteration, best_score, 
                    #           s=100, color=color, marker='*', 
                    #           edgecolors='black', linewidth=1, zorder=5)
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Score Evolution by Iteration for Each Prompt Type', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add annotations for best scores
            for i, (prompt_name, prompt_result) in enumerate(result['prompt_results'].items()):
                if 'iteration_scores' in prompt_result and prompt_result['iteration_scores']:
                    scores = [score['score'] for score in prompt_result['iteration_scores']]
                    iterations = [score['iteration'] for score in prompt_result['iteration_scores']]
                    best_score = max(scores)
                    best_iteration = iterations[scores.index(best_score)]
                    
                    plt.annotate(f'{best_score:.4f}', 
                               xy=(best_iteration, best_score),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, ha='left')
            
            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'score_evolution_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Score evolution graph saved as: {filename}")
            plt.show()
            
            return filename
        
        def plot_best_scores(result):
            """Plot best scores comparison per prompt"""
            plt.figure(figsize=(10, 6))
            
            prompt_names = []
            best_scores = []
            
            for prompt_name, summary in result['summary'].items():
                # Clean prompt name
                clean_name = prompt_name.replace('_template.txt', '')
                prompt_names.append(clean_name)
                best_scores.append(summary['best_score'])
            
            x = np.arange(len(prompt_names))
            
            bars = plt.bar(x, best_scores, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            
            # Add value labels on bars
            for bar, score in zip(bars, best_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{score:.4f}', ha='center', va='bottom')
            
            plt.xlabel('Prompt Type', fontsize=12)
            plt.ylabel('Best Score', fontsize=12)
            plt.title('Best Scores Comparison by Prompt Type', fontsize=14, fontweight='bold')
            plt.xticks(x, prompt_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'best_scores_comparison_{timestamp}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.show()
            
            return filename
        
        def plot_iteration_heatmap(result):
            """Plot heatmap of scores per iteration per prompt"""
            plt.figure(figsize=(12, 6))
            
            # Prepare data for heatmap
            prompt_names = list(result['prompt_results'].keys())
            # Clean prompt names
            clean_prompt_names = [name.replace('_template.txt', '') for name in prompt_names]
            
            max_iterations = max(len(result['prompt_results'][p]['iteration_scores']) 
                               for p in prompt_names)
            
            heatmap_data = np.zeros((len(prompt_names), max_iterations))
            
            for i, prompt_name in enumerate(prompt_names):
                scores = result['prompt_results'][prompt_name]['iteration_scores']
                for j, score_data in enumerate(scores):
                    heatmap_data[i, j] = score_data['score']
            
            plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Score')
            plt.yticks(range(len(clean_prompt_names)), clean_prompt_names)
            plt.xlabel('Iteration')
            plt.ylabel('Prompt Type')
            plt.title('Score Heatmap: Prompts vs Iterations')
            
            # Add text annotations
            for i in range(len(prompt_names)):
                for j in range(max_iterations):
                    if heatmap_data[i, j] > 0:
                        plt.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                               ha='center', va='center', fontsize=8, color='white')
            
            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'iteration_heatmap_{timestamp}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.show()
            
            return filename
        
        try:
            from datetime import datetime
            
            score_evolution_file = plot_score_evolution(result)
            best_scores_file = plot_best_scores(result)
            heatmap_file = plot_iteration_heatmap(result)
            
            print(f"\nVisualization files generated:")
            print(f"- {score_evolution_file}")
            print(f"- {best_scores_file}")
            print(f"- {heatmap_file}")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

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

        
        print("\n" + "="*50)
        print("ORCHESTRATION RESULTS")
        print("="*50)
        print(f"Iteration Type: {ITERATION_TYPE.value}")
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
    
    # print("\nDetailed Results:")
    # pprint.pprint(result)