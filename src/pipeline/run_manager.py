"""
Run Management System for Gravitational Wave Hunter v2.0

This module provides run management capabilities for tracking and reproducing
pipeline executions. It creates unique run directories, tracks metadata,
and enables full reproducibility of experiments.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import json
import hashlib
import subprocess
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)


class RunManager:
    """
    Manages pipeline runs with unique directories and metadata tracking.
    
    This class creates unique run directories, tracks metadata about each run,
    and enables full reproducibility of experiments. It handles version control
    information, configuration snapshots, and run statistics.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for runs, by default "runs/"
    config_path : str, optional
        Path to configuration file, by default "config/download_config.yaml"
        
    Attributes
    ----------
    base_dir : Path
        Base directory for runs
    config_path : Path
        Path to configuration file
    current_run_dir : Optional[Path]
        Current run directory
    run_metadata : Optional[Dict[str, Any]]
        Current run metadata
        
    Examples
    --------
    >>> run_manager = RunManager()
    >>> run_dir = run_manager.create_run()
    >>> print(f"Run directory: {run_dir}")
    >>> run_manager.save_metadata({"status": "completed"})
    """
    
    def __init__(
        self, 
        base_dir: str = "runs/",
        config_path: str = "config/pipeline_clean_config.yaml"
    ) -> None:
        self.base_dir = Path(base_dir)
        self.config_path = Path(config_path)
        self.current_run_dir: Optional[Path] = None
        self.run_metadata: Optional[Dict[str, Any]] = None
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_run(
        self, 
        run_name: Optional[str] = None,
        include_git_hash: bool = True,
        include_timestamp: bool = True
    ) -> Path:
        """
        Create a new unique run directory.
        
        Parameters
        ----------
        run_name : str, optional
            Custom run name, by default None (auto-generated)
        include_git_hash : bool, optional
            Include git hash in run name, by default True
        include_timestamp : bool, optional
            Include timestamp in run name, by default True
            
        Returns
        -------
        Path
            Path to the created run directory
            
        Examples
        --------
        >>> run_manager = RunManager()
        >>> run_dir = run_manager.create_run("experiment_1")
        >>> print(f"Run directory: {run_dir}")
        """
        # Generate run name
        if run_name is None:
            run_name = "run"
            
        # Add timestamp
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{run_name}_{timestamp}"
            
        # Add git hash
        if include_git_hash:
            git_hash = self._get_git_hash()
            if git_hash:
                run_name = f"{run_name}_{git_hash[:8]}"
                
        # Create run directory
        run_dir = self.base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (run_dir / "config").mkdir(exist_ok=True)
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "results").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        
        self.current_run_dir = run_dir
        
        # Initialize metadata
        self.run_metadata = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(),
            "config_path": str(self.config_path),
            "git_hash": self._get_git_hash(),
            "git_branch": self._get_git_branch(),
            "status": "created"
        }
        
        # Save configuration snapshot
        self._save_config_snapshot()
        
        # Save initial metadata
        self.save_metadata()
        
        logger.info(f"Created run directory: {run_dir}")
        return run_dir
        
    def save_metadata(self, additional_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save run metadata to JSON file.
        
        Parameters
        ----------
        additional_metadata : Dict[str, Any], optional
            Additional metadata to merge with existing metadata
        """
        if self.current_run_dir is None:
            raise ValueError("No active run. Call create_run() first.")
            
        if self.run_metadata is None:
            raise ValueError("No run metadata initialized.")
            
        # Merge additional metadata
        if additional_metadata:
            self.run_metadata.update(additional_metadata)
            
        # Add timestamp
        self.run_metadata["last_updated"] = datetime.now().isoformat()
        
        # Convert numpy arrays and other non-serializable types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj) if 'float' in str(type(obj)) else int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_metadata = convert_numpy(self.run_metadata)
        
        # Save to file
        metadata_file = self.current_run_dir / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
            
        logger.info(f"Saved run metadata to {metadata_file}")
        
    def get_run_info(self) -> Dict[str, Any]:
        """
        Get current run information.
        
        Returns
        -------
        Dict[str, Any]
            Current run metadata
        """
        if self.run_metadata is None:
            raise ValueError("No active run. Call create_run() first.")
            
        return self.run_metadata.copy()
        
    def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all runs in the base directory.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of run metadata dictionaries
        """
        runs = []
        
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        runs.append(metadata)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"Could not read metadata for {run_dir}: {e}")
                        
        # Sort by creation time
        runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return runs
        
    def get_run_by_name(self, run_name: str) -> Optional[Dict[str, Any]]:
        """
        Get run metadata by name.
        
        Parameters
        ----------
        run_name : str
            Name of the run to retrieve
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Run metadata if found, None otherwise
        """
        runs = self.list_runs()
        for run in runs:
            if run.get("run_name") == run_name:
                return run
        return None
        
    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
            
    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
            
    def _save_config_snapshot(self) -> None:
        """Save a snapshot of the configuration file."""
        if self.current_run_dir is None:
            return
            
        config_snapshot_path = self.current_run_dir / "config" / "config_snapshot.yaml"
        
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                # Create a placeholder config file with error info
                placeholder_content = f"# CONFIG FILE NOT FOUND\n# Expected path: {self.config_path}\n# This run was created without a valid config file.\n"
                with open(config_snapshot_path, 'w') as f:
                    f.write(placeholder_content)
                self.run_metadata["config_hash"] = "not_found"
                return
            
            # Copy configuration file
            with open(self.config_path, 'r') as f:
                config_content = f.read()
                
            with open(config_snapshot_path, 'w') as f:
                f.write(config_content)
                
            # Calculate config hash
            config_hash = hashlib.sha256(config_content.encode()).hexdigest()
            self.run_metadata["config_hash"] = config_hash
            logger.info(f"Config snapshot saved to {config_snapshot_path}")
            
        except IOError as e:
            logger.error(f"Could not save config snapshot: {e}")
            # Create error file
            error_content = f"# ERROR SAVING CONFIG\n# Error: {e}\n# Original path: {self.config_path}\n"
            try:
                with open(config_snapshot_path, 'w') as f:
                    f.write(error_content)
            except:
                pass  # If we can't even write the error, give up
            
    def add_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Add model information to run metadata.
        
        Parameters
        ----------
        model_info : Dict[str, Any]
            Model architecture and training information
        """
        if self.run_metadata is None:
            raise ValueError("No active run. Call create_run() first.")
            
        if "models" not in self.run_metadata:
            self.run_metadata["models"] = []
            
        self.run_metadata["models"].append({
            "added_at": datetime.now().isoformat(),
            **model_info
        })
        
    def add_training_results(self, results: Dict[str, Any]) -> None:
        """
        Add training results to run metadata.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Training results and metrics
        """
        if self.run_metadata is None:
            raise ValueError("No active run. Call create_run() first.")
            
        self.run_metadata["training_results"] = {
            "added_at": datetime.now().isoformat(),
            **results
        }
        
    def add_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Add evaluation results to run metadata.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Evaluation results and metrics
        """
        if self.run_metadata is None:
            raise ValueError("No active run. Call create_run() first.")
            
        self.run_metadata["evaluation_results"] = {
            "added_at": datetime.now().isoformat(),
            **results
        }
        
    def mark_completed(self, success: bool = True) -> None:
        """
        Mark the current run as completed.
        
        Parameters
        ----------
        success : bool, optional
            Whether the run was successful, by default True
        """
        if self.run_metadata is None:
            raise ValueError("No active run. Call create_run() first.")
            
        self.run_metadata["status"] = "completed" if success else "failed"
        self.run_metadata["completed_at"] = datetime.now().isoformat()
        
        self.save_metadata()
        
    def get_reproducibility_info(self) -> Dict[str, Any]:
        """
        Get information needed to reproduce this run.
        
        Returns
        -------
        Dict[str, Any]
            Reproducibility information
        """
        if self.run_metadata is None:
            raise ValueError("No active run. Call create_run() first.")
            
        return {
            "run_name": self.run_metadata.get("run_name"),
            "config_hash": self.run_metadata.get("config_hash"),
            "git_hash": self.run_metadata.get("git_hash"),
            "git_branch": self.run_metadata.get("git_branch"),
            "created_at": self.run_metadata.get("created_at"),
            "config_snapshot_path": str(self.current_run_dir / "config" / "config_snapshot.yaml")
        }
