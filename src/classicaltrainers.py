from typing import Dict, List, Optional, Any, Union, Type
import numpy as np
import optuna
from optuna.trial import Trial, FrozenTrial
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from qiskit_machine_learning.algorithms import VQC
import time
from logger import get_logger, save_model_results, create_progress_bar
from pathlib import Path
from datetime import datetime
import json
from preprocessing import IDSPreprocessor

results_logger = get_logger("results")

# Type alias for classifiers that implement scikit-learn's estimator interface
SklearnClassifier = Union[DecisionTreeClassifier, GaussianNB]
Classifier = Union[SklearnClassifier, VQC]

def get_model_name(model_type: str, preprocessor: IDSPreprocessor) -> str:
    """Generate standardized model name with preprocessing info."""
    scaler_map = {
        'standard': 'SS',
        'maxabs': 'MAS'
    }
    dim_reduction_map = {
        'pca': 'PCA',
        'none': 'RAW'
    }
    
    scaler_type = scaler_map.get(preprocessor.scaler_type, 'UNK')
    dim_reduction = dim_reduction_map.get(preprocessor.dim_reduction_type, 'UNK')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{scaler_type}_{dim_reduction}_{model_type}_{timestamp}"

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def get_project_path(relative_path: str) -> Path:
    """Get an absolute path relative to the project root."""
    root = get_project_root()
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

class BaseTrainer:
    """Base class for model training with hyperparameter optimization"""
    
    def __init__(self, n_trials: int = 10, class_names: Optional[List[str]] = None) -> None:
        self.n_trials: int = n_trials
        self.class_names: List[str] = class_names or ['Normal', 'Attack']
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.training_times: Dict[str, float] = {}
        self.models: Dict[str, Classifier] = {}
        self.preprocessing_info: Dict[str, Dict[str, Any]] = {}
    
    def _create_model(self, trial: Union[Trial, FrozenTrial]) -> Classifier:
        """Create model with trial parameters - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _evaluate_model(self, model: Classifier, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model using cross-validation or direct evaluation based on model type."""
        if isinstance(model, (DecisionTreeClassifier, GaussianNB)):
            # For scikit-learn models, use cross-validation
            scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            return float(np.mean(scores))
        else:
            # For quantum models, use direct evaluation
            try:
                model.fit(X, y)
                y_pred = model.predict(X)
                return float(f1_score(y, y_pred, average='weighted'))
            except Exception as e:
                results_logger.error(f"Error evaluating model: {str(e)}")
                return float('-inf')
    
    def train_model(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        y_train: np.ndarray, 
        y_test: np.ndarray,
        model_name: str,
        preprocessor: Optional[IDSPreprocessor] = None
    ) -> Classifier:
        """Train and evaluate a model with hyperparameter optimization"""
        start_time = time.time()
        
        # Store preprocessing information if available
        if preprocessor:
            self.preprocessing_info[model_name] = {
                "scaler": {
                    "type": preprocessor.scaler_type,
                    "params": preprocessor.get_scaler_params()
                },
                "dim_reduction": {
                    "type": preprocessor.dim_reduction_type,
                    "params": preprocessor.get_dim_reduction_params()
                }
            }
        
        # Use the utility function for paths
        status_file = get_project_path("results/status.json")
        
        def update_status(status: dict):
            with open(status_file, 'w') as f:
                json.dump({
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    **status
                }, f)
        
        try:
            update_status({
                "stage": "initializing",
                "message": f"Starting {model_name} training",
                "progress": 0.0
            })
            
            # Use the utility function for optuna path
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            study_path = get_project_path(f"results/optuna/{timestamp}_{model_name}.db")
            
            study = optuna.create_study(
                direction="maximize",
                storage=f"sqlite:///{study_path}",
                study_name=f"{model_name}_{timestamp}"
            )
            
            best_score = 0.0
            def objective(trial: Trial) -> float:
                trial_start = time.time()
                model = self._create_model(trial)
                mean_score = self._evaluate_model(model, X_train, y_train)
                
                # Update status with trial progress
                nonlocal best_score
                best_score = max(best_score, mean_score)
                progress = (trial.number + 1) / self.n_trials
                
                update_status({
                    "stage": "optimizing",
                    "message": f"Trial {trial.number + 1}/{self.n_trials}",
                    "progress": progress,
                    "current_score": mean_score,
                    "best_score": best_score,
                    "trial_time": time.time() - trial_start
                })
                
                return mean_score
            
            # Optimize
            study.optimize(objective, n_trials=self.n_trials)
            
            # Final model training
            update_status({
                "stage": "finalizing",
                "message": "Training final model",
                "progress": 0.95
            })
            
            best_params = study.best_params
            model = self._create_model(study.best_trial)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics and store results
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision_micro': float(precision_score(y_test, y_pred, average='micro')),
                'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
                'recall_micro': float(recall_score(y_test, y_pred, average='micro')),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_micro': float(f1_score(y_test, y_pred, average='micro')),
                'f1_weighted': float(f1_score(y_test, y_pred, average='weighted'))
            }
            
            # Calculate total training time
            training_time = time.time() - start_time
            
            # Update final status
            update_status({
                "stage": "completed",
                "message": "Training completed",
                "progress": 1.0,
                "metrics": metrics,
                "training_time": training_time
            })
            
            # Store results
            self.best_params[model_name] = best_params
            self.metrics[model_name] = metrics
            self.training_times[model_name] = training_time
            self.models[model_name] = model
            
            # Save results file
            save_model_results(
                model_name=model_name,
                metrics=metrics,
                parameters=best_params,
                training_time=training_time,
                dataset_info=f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}",
                preprocessing_info=self.preprocessing_info.get(model_name, {}),
                optuna_study={
                    'name': f"{model_name}_{timestamp}",
                    'path': str(study_path),
                    'n_trials': self.n_trials,
                    'best_value': float(study.best_value)
                }
            )
            
            return model
            
        except Exception as e:
            update_status({
                "stage": "error",
                "message": str(e),
                "progress": 0.0
            })
            raise

class DecisionTreeTrainer(BaseTrainer):
    """Decision Tree model trainer with hyperparameter optimization"""
    def _create_model(self, trial: Union[Trial, FrozenTrial]) -> DecisionTreeClassifier:
        params: Dict[str, Any] = {
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
        return DecisionTreeClassifier(**params, random_state=42)

class NaiveBayesTrainer(BaseTrainer):
    """Naive Bayes model trainer with hyperparameter optimization"""
    def _create_model(self, trial: Union[Trial, FrozenTrial]) -> GaussianNB:
        var_smoothing: float = trial.suggest_float('var_smoothing', 1e-11, 1e-7, log=True)
        return GaussianNB(var_smoothing=var_smoothing) 