import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages versioning and persistence of datasets and transformations."""
    
    def __init__(self, output_dir: str = "data/versions"):
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, "models")
        self.current_version = 0
        self.version_history = []
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def increment_version(self) -> int:
        """Increment and return the current version number."""
        self.current_version += 1
        return self.current_version
    
    def save_dataset(self, dataset: pd.DataFrame, suffix: str = "") -> str:
        """Save dataset with versioned filename."""
        filename = f"dataset_v{self.current_version}{suffix}.csv"
        output_path = os.path.join(self.output_dir, filename)
        dataset.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
        return output_path
    
    def create_version_entry(
        self,
        input_path: str,
        fe_output_path: str,
        final_output_path: str,
        new_transformations_count: int,
        total_transformations_count: int,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a version history entry."""
        version_entry = {
            "version": self.current_version,
            "input_path": input_path,
            "fe_output_path": fe_output_path,
            "output_path": final_output_path,
            "new_transformations_count": new_transformations_count,
            "total_transformations_count": total_transformations_count,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }
        self.version_history.append(version_entry)
        return version_entry
    
    def save_version_config(
        self,
        transformations: List,
        dataset_description: str,
        input_path: str,
        output_path: str,
        dataset: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> str:
        """Save configuration for current version."""
        config_filename = f"config_v{self.current_version}.json"
        config_path = os.path.join(self.output_dir, config_filename)
        
        # Convert transformations to JSON-serializable format
        json_transformations = []
        for t in transformations:
            if hasattr(t, 'model_dump'):
                json_transformations.append(t.model_dump())
            else:
                json_transformations.append(t.__dict__)
        
        current_version_data = {
            "version": self.current_version,
            "timestamp": datetime.now().isoformat(),
            "dataset_description": dataset_description,
            "input_file": input_path,
            "output_file": output_path,
            "dataset_shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
            "target_column": target_column,
            "transformations": json_transformations,
            "columns": list(dataset.columns)
        }
        
        config_data = {
            "current_version": current_version_data,
            "version_history": self.version_history
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration for version {self.current_version} to {config_path}")
        return config_path
    
    def save_global_summary(self):
        """Save global summary of all versions."""
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
