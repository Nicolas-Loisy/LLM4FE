import logging
from typing import Optional, Dict, Any, List
import os
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline
from src.data_cleanning.data_cleaner import DataCleaner
# from src.automl.automl_pipeline import AutoMLPipeline
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config_path: str = "data/config.json"):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to the configuration file
        """

        logger.info("Initializing Orchestrator...")
        self.config = get_config(config_path)
        
        # Tracking des versions et transformations
        self.current_version = 0
        self.version_history = []
        
        # Créer les répertoires nécessaires
        self.output_dir = os.path.join("data", "versions")
        self.models_dir = os.path.join(self.output_dir, "models")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def run(
        self, 
        dataset_path: str, 
        dataset_description: Optional[str] = None, 
        target_column: Optional[str] = None,
        iterations: int = 1
    ) -> Dict[str, Any]:
        """
        Main entry point to run the feature engineering pipeline with multiple iterations.

        Args:
            dataset_path: Path to the input dataset
            dataset_description: Optional description of the dataset for feature engineering
            target_column: Optional target column for supervised learning tasks
            iterations: Number of feature engineering iterations to run

        Returns:
            Dictionary with information about all versions and the final dataset
        """
        logger.info(f"Starting LLM4FE orchestration with {iterations} iterations...")
        
        current_dataset_path = dataset_path
        current_description = dataset_description
        all_transformations = []
        
        for i in range(iterations):
            # Incrémenter la version
            self.current_version += 1
            logger.info(f"Starting iteration {i+1}/{iterations} (version {self.current_version})...")
            
            # Initialiser le pipeline FE pour cette itération avec la colonne cible
            fe_pipeline = FeatureEngineeringPipeline(
                dataset_path=current_dataset_path,
                dataset_description=current_description,
                target_column=target_column
            )
            
            # Exécuter une itération de feature engineering
            transformed_dataset, new_transformations, updated_description = fe_pipeline.run()
            
            if transformed_dataset is not None:
                # Sauvegarder le dataset après feature engineering
                fe_output_path = os.path.join(self.output_dir, f"dataset_v{self.current_version}_fe.csv")
                transformed_dataset.to_csv(fe_output_path, index=False)
                
                # Appliquer le nettoyage des données après feature engineering
                logger.info("Applying data cleaning after feature engineering...")
                data_cleaner = DataCleaner()
                cleaned_output_path = os.path.join(self.output_dir, f"dataset_v{self.current_version}.csv")
                final_output_path = data_cleaner.run(
                    input_path=fe_output_path,
                    output_path=cleaned_output_path,
                    threshold=0.8,
                    target_column=target_column
                )
                
                # Charger le dataset final nettoyé
                final_dataset = pd.read_csv(final_output_path)
                
                # Mise à jour de la description si nécessaire
                if updated_description:
                    current_description = updated_description
                
                # Garder une trace de toutes les transformations
                all_transformations.extend(new_transformations)
                
                # Créer l'entrée d'historique pour cette itération
                version_entry = {
                    "version": self.current_version,
                    "input_path": current_dataset_path,
                    "fe_output_path": fe_output_path,
                    "output_path": final_output_path,
                    "new_transformations_count": len(new_transformations),
                    "total_transformations_count": len(all_transformations),
                    "description": updated_description,
                    "timestamp": datetime.now().isoformat(),
                }
                self.version_history.append(version_entry)
                
                # Sauvegarder la configuration
                config_path = self.save_version_config(
                    self.current_version, 
                    all_transformations,
                    updated_description or current_description or "",
                    current_dataset_path,
                    final_output_path,
                    final_dataset,
                    target_column
                )
                
                # Ajouter le chemin de config à l'entrée d'historique
                self.version_history[-1]["config_path"] = config_path
                
                # Mettre à jour pour la prochaine itération
                current_dataset_path = final_output_path
                
                logger.info(f"Iteration {i+1} completed successfully")
            else:
                logger.error(f"Iteration {i+1} failed")
                break
        
        # Sauvegarder les informations globales
        self.save_global_info()
        
        logger.info(f"Completed all {iterations} iterations")
        return {
            'versions': self.version_history,
            'final_dataset': current_dataset_path if transformed_dataset is not None else None,
            'transformations_count': len(all_transformations)
        }

    def save_version_config(
            self, 
            version: int, 
            transformations: List, 
            dataset_description: str, 
            input_path: str, 
            output_path: str, 
            dataset: pd.DataFrame,
            target_column: Optional[str] = None
        ) -> str:
        """
        Save configuration information for a version.
        
        Args:
            version: Version number
            transformations: Applied transformations
            dataset_description: Dataset description
            input_path: Path to input dataset
            output_path: Path to output dataset
            dataset: The transformed dataset
            target_column: Optional target column name
            
        Returns:
            Path to the saved configuration file
        """
        config_filename = f"config_v{version}.json"
        config_path = os.path.join(self.output_dir, config_filename)
        
        # Convert transformations directly to JSON-serializable format
        json_transformations = []
        for t in transformations:
            if hasattr(t, 'model_dump'):
                json_transformations.append(t.model_dump())
            else:
                # Fallback if model_dump is not available
                json_transformations.append(t.__dict__)
                
        # Current version data
        current_version_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "dataset_description": dataset_description,
            "input_file": input_path,
            "output_file": output_path,
            "dataset_shape": {
                "rows": dataset.shape[0],
                "columns": dataset.shape[1]
            },
            "target_column": target_column,
            "transformations": json_transformations,
            "columns": list(dataset.columns)
        }
        
        # Full configuration with version history
        config_data = {
            "current_version": current_version_data,
            "version_history": self.version_history  # Include all previous versions
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration for version {version} to {config_path}")
        return config_path
        
    def save_global_info(self):
        """
        Save global information about all versions
        """
        global_info = {
            'total_versions': self.current_version,
            'versions_summary': {}
        }
        
        for entry in self.version_history:
            version = entry.get('version')
            global_info['versions_summary'][version] = {
                'timestamp': entry.get('timestamp'),
                'description': entry.get('description'),
                'input_path': entry.get('input_path'),
                'output_path': entry.get('output_path'),
                'transformations_count': entry.get('total_transformations_count')
            }
            
        info_path = os.path.join(self.output_dir, "versions_summary.json")
        with open(info_path, 'w') as f:
            json.dump(global_info, f, indent=4)
            
        logger.info(f"Saved global versions info to {info_path}")
