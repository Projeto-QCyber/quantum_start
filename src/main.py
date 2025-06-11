import argparse
from typing import Dict, Any, Tuple, List, Optional
from classicaltrainers import DecisionTreeTrainer, NaiveBayesTrainer, BaseTrainer
from quantumtrainers import VQCTrainer
from qiskit.providers.fake_provider import GenericBackendV2
from logger import get_logger, log_metrics_table, create_progress_bar
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import IDSPreprocessor
from scipy.sparse import spmatrix
from dataset import generate_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train and evaluate ML models')
    
    # Model selection
    parser.add_argument('--nb', action='store_true', help='Train NB (Naive Bayes) model')
    parser.add_argument('--dt', action='store_true', help='Train DT (Decision Tree) model')
    parser.add_argument('--vqc', action='store_true', help='Train VQC (Variational Quantum Classifier) model')
    parser.add_argument('--all', action='store_true', help='Train all available models (default)')
    
    # Dataset options
    parser.add_argument('--micro', action='store_true', 
                      help='Use micro dataset for quick testing (100 samples)')
    parser.add_argument('--samples', type=int, default=1000,
                      help='Number of samples for training (ignored if --micro is set)')
    
    # Training parameters
    parser.add_argument('--trials', type=int, default=10,
                      help='Number of optimization trials')
    parser.add_argument('--verbose', action='store_true',
                      help='Show detailed training progress')
    
    # Output options
    parser.add_argument('--no-plots', action='store_true',
                      help='Disable plot generation')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # If no specific model is selected, run all
    if not (args.nb or args.dt or args.vqc):
        args.all = True
    
    return args

def save_model_results(trainer: BaseTrainer, name: str, dataset_info: str) -> None:
    """Save model results directly to JSON file in results directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = Path("../results/data") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "model_name": name,
        "metrics": trainer.metrics[name],
        "parameters": trainer.best_params[name],
        "training_time": trainer.training_times[name],
        "timestamp": timestamp,
        "dataset_info": dataset_info
    }
    
    result_path = results_dir / f"{name}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

def main() -> None:
    args = parse_args()
    
    # Initialize loggers
    results_logger = get_logger("results", verbose=args.verbose)
    debug_logger = get_logger("debug", verbose=args.verbose)
    
    try:
        trainers: List[Tuple[str, BaseTrainer]] = []
        
        # Initialize quantum backend
        backend = GenericBackendV2(num_qubits=8)  # 8 qubits for PCA reduced features
        
        if args.all:
            trainers.extend([
                ('NB', NaiveBayesTrainer(n_trials=args.trials)),
                ('DT', DecisionTreeTrainer(n_trials=args.trials)),
                ('VQC', VQCTrainer(
                    backend=backend,
                    study_name="vqc_study",
                    n_trials=args.trials
                ))
            ])
        else:
            if args.nb:
                trainers.append(('NB', NaiveBayesTrainer(n_trials=args.trials)))
            if args.dt:
                trainers.append(('DT', DecisionTreeTrainer(n_trials=args.trials)))
            if args.vqc:
                trainers.append(('VQC', VQCTrainer(
                    backend=backend,
                    study_name="vqc_study",
                    n_trials=args.trials
                )))
        
        if not trainers:
            results_logger.error("No models selected for training")
            return
        
        # Generate dataset
        X_train, X_test, y_train, y_test = generate_dataset(
            micro=args.micro,
            n_samples=args.samples
        )
        
        dataset_info = "micro dataset" if args.micro else f"{args.samples} samples"
        results_logger.info(f"Training on {dataset_info}")
        
        # Training progress
        with create_progress_bar() as progress:
            for name, trainer in trainers:
                task = progress.add_task(f"Training {name}...", total=None)
                
                trainer.train_model(X_train, X_test, y_train, y_test, 
                                model_name=name)
                progress.update(task, completed=True)
                
                # Log detailed metrics
                log_metrics_table(results_logger, trainer.metrics, dataset_info)
                
                # Save results directly to file
                save_model_results(trainer, name, dataset_info)

        results_logger.info("Pipeline completed successfully")
        
    except Exception as e:
        debug_logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 