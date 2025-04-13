# Main script to launch the LLM4FE pipeline

import argparse
import os
import sys
from src.orchestrator.orchestrator import Orchestrator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLM4FE: LLM-based Feature Engineering Pipeline')
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                        help='Path to the input dataset CSV file')
    parser.add_argument('--config', '-c', type=str, default='config.json',
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                        help='Number of pipeline iterations to run (default: 1)')
    parser.add_argument('--description', '-s', type=str, default=None,
                        help='Optional description of the dataset for the LLM')
    return parser.parse_args()

def main():
    """Main function to run the pipeline"""
    print("Running the LLM4FE pipeline...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Verify the dataset file exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1
    
    # Verify the config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    try:
        # Create an instance of the Orchestrator with dataset description
        orchestrator = Orchestrator(
            config_path=args.config,
            dataset_description=args.description
        )
        
        # Run the pipeline with a single method call
        result = orchestrator.run(
            dataset_path=args.dataset,
            iterations=args.iterations
        )
        
        # Display results
        if "version" in result:
            print(f"\nPipeline completed successfully! Best model version: {result['version']}")
            print(f"Best model directory: {result['version_dir']}")
            print(f"Best model score: {result.get('scores', {})}")
        else:
            print(f"\nPipeline completed: {result['message']}")
        
        return 0
    
    except Exception as e:
        print(f"Error running the pipeline: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
