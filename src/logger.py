from typing import Dict, Optional, Any
import logging
import os
from datetime import datetime
import sys
import json
from pathlib import Path
from contextlib import contextmanager

# Store loggers in a dict to avoid duplicate configuration
_loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Setup logger with specified configuration.
    
    Args:
        name: Logger name
        verbose: If True, sets level to DEBUG, otherwise INFO
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
        
    logger = logging.getLogger(name)
    logger.propagate = False
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create console handler with custom formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Use shorter format for better readability
    formatter = logging.Formatter(
        '%(levelname)-8s %(name)-12s %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Store logger in cache
    _loggers[name] = logger
    
    return logger

def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Get or create a logger with specified configuration.
    
    Args:
        name: Logger name
        verbose: If True, sets level to DEBUG, otherwise INFO
        
    Returns:
        Logger instance
    """
    # Return cached logger if it exists
    if name in _loggers:
        return _loggers[name]
        
    return setup_logger(name, verbose)

@contextmanager 
def create_progress_bar():
    """Create and manage a progress bar context."""
    try:
        yield None
    finally:
        pass

def log_metrics_table(logger: logging.Logger, metrics: Dict[str, Dict[str, float]], dataset_info: str) -> None:
    """Log metrics in a formatted table.
    
    Args:
        logger: Logger instance to use
        metrics: Dictionary of model metrics
        dataset_info: Information about the dataset used
    """
    for model_name, model_metrics in metrics.items():
        logger.info(f"\n{model_name} - {dataset_info}")
        logger.info("-" * 40)
        for metric, value in model_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric:<15} {value:>8.4f}")

def save_model_results(
    model_name: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
    training_time: float,
    dataset_info: str,
    preprocessing_info: Dict[str, Any],
    optuna_study: Dict[str, Any]
) -> None:
    """Save model results to JSON file.
    
    Args:
        model_name: Name of the model
        metrics: Model performance metrics
        parameters: Model parameters
        training_time: Time taken for training
        dataset_info: Information about the dataset
        preprocessing_info: Preprocessing configuration
        optuna_study: Optuna study results
        
    Raises:
        OSError: If directory creation or file writing fails
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_dir = Path("../results/data") / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "parameters": parameters,
            "training_time": training_time,
            "timestamp": timestamp,
            "dataset_info": dataset_info,
            "preprocessing_info": preprocessing_info,
            "optuna_study": optuna_study
        }
        
        result_path = results_dir / f"{model_name}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to save results: {str(e)}")
        raise

def clear_loggers() -> None:
    """Clear all cached loggers."""
    global _loggers
    _loggers.clear()