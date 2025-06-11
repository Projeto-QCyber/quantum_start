import numpy as np
from qiskit.circuit.library import (
    ZZFeatureMap, RealAmplitudes, PauliFeatureMap, 
    ZFeatureMap, EfficientSU2,
    ExcitationPreserving, TwoLocal
)
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import BackendSamplerV2
from qiskit_machine_learning.algorithms import VQC
from typing import Dict, List, Optional, Any, Union, Type, Sequence, Tuple, Iterable, Callable
import optuna
from optuna import Trial
from optuna.trial import FrozenTrial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from qiskit.providers import BackendV2
from qiskit_machine_learning.optimizers import COBYLA, SPSA
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import traceback
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from classicaltrainers import BaseTrainer
from preprocessing import IDSPreprocessor
from logger import get_logger
from dataset import generate_dataset, load_dataset_info

class TestTrial(optuna.trial.BaseTrial):
    """Trial subclass for testing optimizers with fixed parameters."""
    
    def __init__(self, optimizer_type: str):
        self._optimizer_type = optimizer_type
        self._params = {
            'maxiter': 100,
            'learning_rate': 0.01,
            'beta1': 0.9,
            'tol': 1e-5,
            'perturbation': 0.1,
            'regularization': 0.01
        }
        self._user_attrs = {}
        self._system_attrs = {}
        self._datetime_start = datetime.now()
        self._number = 0
    
    def _suggest(self, name: str) -> float:
        return self._params.get(name, 0.01)
    
    def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
        return self._suggest(name)
    
    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
        return int(self._suggest(name))
    
    def suggest_categorical(self, name: str, choices: List[str]) -> str:
        if name == 'optimizer':
            return self._optimizer_type
        return choices[0]
    
    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        return self._suggest(name)
    
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        return self._suggest(name)
    
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        return self._suggest(name)
    
    def report(self, value: float, step: int) -> None:
        pass
    
    def should_prune(self) -> bool:
        return False
    
    def set_user_attr(self, key: str, value: Any) -> None:
        self._user_attrs[key] = value
    
    def set_system_attr(self, key: str, value: Any) -> None:
        self._system_attrs[key] = value
    
    @property
    def params(self) -> Dict[str, Any]:
        return self._params
    
    @property
    def distributions(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        return {}
    
    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self._user_attrs
    
    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self._system_attrs
    
    @property
    def datetime_start(self) -> datetime:
        return self._datetime_start
    
    @property
    def number(self) -> int:
        return self._number

class VQCTrainer(BaseTrainer):
    """Variational Quantum Classifier trainer with hyperparameter optimization"""
    
    # Circuit configurations with explicit dimension handling
    FEATURE_MAPS = {
        'PauliFeatureMap': {
            'class': PauliFeatureMap,
            'uses_qubit_dimension': True,  # feature_dimension = n_qubits
            'params': {
                'paulis': ['Z', 'ZZ'],
                'reps': 2
            }
        },
        'RawFeatureVector': {
            'class': RawFeatureVector,
            'uses_qubit_dimension': False,  # feature_dimension = 2**n_qubits
            'params': {}
        },
        'ZFeatureMap': {
            'class': ZFeatureMap,
            'uses_qubit_dimension': True,  # feature_dimension = n_qubits
            'params': {
                'reps': 2
            }
        },
        'ZZFeatureMap': {
            'class': ZZFeatureMap,
            'uses_qubit_dimension': True,  # feature_dimension = n_qubits
            'params': {
                'reps': 2
            }
        }
    }
    
    ANSATZE = {
        'EfficientSU2': {
            'class': EfficientSU2,
            'params': {
                'reps': 2,
                'entanglement': 'linear'
            }
        },
        'ExcitationPreserving': {
            'class': ExcitationPreserving,
            'params': {
                'reps': 2,
                'entanglement': 'linear'
            }
        },
        'RealAmplitudes': {
            'class': RealAmplitudes,
            'params': {
                'reps': 2,
                'entanglement': 'linear'
            }
        },
        'TwoLocal': {
            'class': TwoLocal,
            'params': {
                'rotation_blocks': ['ry', 'rz'],
                'entanglement_blocks': 'cz',
                'reps': 2
            }
        }
    }
    
    def __init__(self, 
                 backend: BackendV2,
                 study_name: str,
                 storage_path: str = "optuna_studies.db",
                 n_trials: int = 3,
                 class_names: Optional[List[str]] = None):
        """Initialize the VQC trainer with study persistence."""
        super().__init__(n_trials=n_trials, class_names=class_names)
        self.backend = backend
        self.study_name = study_name
        self.storage_path = storage_path

        # Initialize other components
        self.sampler = BackendSamplerV2(backend=self.backend)
        self.n_qubits = None  # Will be set based on data dimensions
        
        # Setup logging
        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing VQCTrainer with study: {study_name}")
        self.logger.info(f"Backend: {backend}, Trials: {n_trials}")

    def _create_optimizer(self, trial: Union[Trial, FrozenTrial, TestTrial]) -> Any:
        """Create optimizer based on trial parameters."""
        optimizer_type = trial.suggest_categorical(
            'optimizer', ['COBYLA', 'SPSA']
        )
        
        optimizer_params: Dict[str, Any] = {}
        
        if optimizer_type == 'COBYLA':
            optimizer_params.update({
                'maxiter': trial.suggest_int('maxiter', 100, 500),
                'tol': trial.suggest_float('tol', 1e-6, 1e-4, log=True)
            })
            optimizer = COBYLA(**optimizer_params)
            
        else:  # SPSA
            optimizer_params.update({
                'maxiter': trial.suggest_int('maxiter', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'perturbation': trial.suggest_float('perturbation', 0.01, 0.1),
                'resamplings': 1,  # Reduce noise in gradient estimation
                'trust_region': True  # Add trust region to prevent large parameter changes
            })
            optimizer = SPSA(**optimizer_params)
        
        return optimizer

    def _validate_optimizer_dimensions(self, optimizer: Any, n_parameters: int) -> None:
        """Validate optimizer's internal state dimensions match parameter count."""
        logger = get_logger(__name__)
        
        if isinstance(optimizer, SPSA):
            # SPSA handles dimensions automatically during minimize
            logger.debug(f"SPSA will initialize with {n_parameters} parameters during training")

    def _prepare_input_data(self, X: np.ndarray, n_qubits: int, feature_map_type: str) -> np.ndarray:
        """Prepare input data according to feature map requirements.
        
        Args:
            X: Input data array
            n_qubits: Number of qubits to use
            feature_map_type: Type of feature map
            
        Returns:
            np.ndarray: Prepared input data
        """
        if feature_map_type == 'RawFeatureVector':
            target_size = 2 ** n_qubits
            if X.shape[1] < target_size:
                # Pad with zeros if needed
                pad_width = ((0, 0), (0, target_size - X.shape[1]))
                X = np.pad(X, pad_width, mode='constant', constant_values=0)
            elif X.shape[1] > target_size:
                # Truncate if too large
                X = X[:, :target_size]
            
            # Normalize to create valid quantum states
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = np.divide(X, row_norms, where=row_norms != 0)
        
        return X

    def _create_model(self, 
                     trial: Optional[Union[Trial, FrozenTrial]] = None,
                     feature_map_type: Optional[str] = None,
                     ansatz_type: Optional[str] = None,
                     n_qubits: Optional[int] = None,
                     callback: Optional[Callable] = None) -> VQC:
        """Create VQC model with specified or trial parameters."""
        logger = get_logger(__name__)
        
        # Get circuit types either from trial or parameters
        if trial is not None:
            feature_map_type = trial.suggest_categorical(
                'feature_map', list(self.FEATURE_MAPS.keys())
            )
            ansatz_type = trial.suggest_categorical(
                'ansatz', list(self.ANSATZE.keys())
            )
        
        if feature_map_type is None or ansatz_type is None:
            raise ValueError("Must provide either trial or specific circuit types")

        # Get feature map configuration
        feature_map_config = self.FEATURE_MAPS[feature_map_type]
        feature_map_class = feature_map_config['class']
        feature_map_params = feature_map_config['params'].copy()
        uses_qubit_dimension = feature_map_config['uses_qubit_dimension']
        
        # Get ansatz configuration
        ansatz_config = self.ANSATZE[ansatz_type]
        ansatz_class = ansatz_config['class']
        ansatz_params = ansatz_config['params'].copy()
        
        # Validate n_qubits is provided
        if n_qubits is None:
            raise ValueError("Must provide n_qubits")
        
        # Create feature map with proper dimension handling
        if uses_qubit_dimension:
            # For quantum feature maps (PauliFeatureMap, ZFeatureMap, etc.)
            # feature_dimension determines the number of qubits
            feature_map = feature_map_class(
                feature_dimension=n_qubits,
                **feature_map_params
            )
            logger.debug(f"Created {feature_map_type} with {n_qubits} qubits")
            ansatz_qubits = n_qubits  # Ansatz matches feature map qubits
        else:
            # For RawFeatureVector, feature_dimension must be 2^n_qubits
            feature_dimension = 2 ** n_qubits
            feature_map = feature_map_class(
                feature_dimension=feature_dimension,
                **feature_map_params
            )
            logger.debug(f"Created {feature_map_type} with {feature_dimension} features ({n_qubits} qubits)")
            ansatz_qubits = n_qubits  # Ansatz uses log2(features) qubits
        
        # Create ansatz with matching number of qubits
        ansatz = ansatz_class(
            num_qubits=ansatz_qubits,
            **ansatz_params
        )
        logger.debug(f"Created {ansatz_type} ansatz with {ansatz_qubits} qubits")
        
        logger.debug(f"Created model with {feature_map_type} feature map and {ansatz_type} ansatz")
        logger.debug(f"Using {n_qubits} qubits")
        
        # Create optimizer and validate dimensions
        optimizer = self._create_optimizer(trial) if trial else SPSA(maxiter=100)
        self._validate_optimizer_dimensions(optimizer, ansatz.num_parameters)
        
        # Create VQC model with validated components
        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss="cross_entropy",
            optimizer=optimizer,
            callback=callback,
            initial_point=np.random.random(ansatz.num_parameters)
        )
        
        return model

    def train_model(self, 
                   X_train: np.ndarray,
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str,
                   preprocessor: Optional[IDSPreprocessor] = None,
                   feature_map: str = 'PauliFeatureMap',
                   ansatz_type: str = 'EfficientSU2',
                   patience: int = 10,
                   min_delta: float = 1e-4
                   ) -> VQC:
        """Train quantum model with hyperparameter optimization and early stopping"""
        logger = get_logger(__name__)
        logger.info(f"Starting model training for {model_name}")
        logger.info(f"Training data shape: {X_train.shape}")
        start_time = time.time()
        
        # Apply preprocessing if provided
        if preprocessor:
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
            X_train = preprocessor.fit_transform(X_train_df)
            X_test = preprocessor.transform(X_test_df)
            self.preprocessing_info[model_name] = {
                "scaler": preprocessor.get_scaler_params(),
                "dim_reduction": preprocessor.get_dim_reduction_params()
            }
        
        # Get feature map configuration
        feature_map_config = self.FEATURE_MAPS[feature_map]
        uses_qubit_dimension = feature_map_config['uses_qubit_dimension']
        
        # Calculate required qubits based on feature map type
        input_size = X_train.shape[1]
        if uses_qubit_dimension:
            # For quantum feature maps, n_qubits = n_features
            n_qubits = input_size
            logger.info(f"Using {n_qubits} qubits (one per feature)")
        else:
            # For RawFeatureVector, n_qubits = log2(n_features)
            n_qubits = int(np.ceil(np.log2(input_size)))
            target_size = 2 ** n_qubits
            if target_size != input_size:
                logger.warning(
                    f"Input size {input_size} is not a power of 2. "
                    f"Data will be padded/truncated to {target_size} features."
                )
            logger.info(f"Using {n_qubits} qubits for {target_size} features")
        
        # Validate backend can support required qubits
        self._validate_backend(n_qubits)
        
        # Prepare input data according to feature map requirements
        X_train = self._prepare_input_data(X_train, n_qubits, feature_map)
        X_test = self._prepare_input_data(X_test, n_qubits, feature_map)

        def objective(trial: Trial) -> float:
            trial_start = time.time()
            logger.info(f"\nStarting trial {trial.number}")
            
            try:
                # Initialize early stopping variables
                best_loss = float('inf')
                patience_counter = 0
                loss_history = []
                
                def callback(weights: np.ndarray, loss: float) -> None:
                    nonlocal best_loss, patience_counter
                    loss_history.append(loss)
                    
                    if loss < best_loss - min_delta:
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        trial.set_user_attr('stopped_early', True)
                
                # Create model with specific feature map and ansatz
                model = self._create_model(
                    trial=trial,
                    feature_map_type=feature_map,
                    ansatz_type=ansatz_type,
                    n_qubits=n_qubits,
                    callback=callback
                )
                
                # Train the model
                model.fit(X_train, y_train)
                
                # If training completed normally
                y_pred = model.predict(X_test)
                
                # Calculate multiple metrics
                scores = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Store the scores and convergence info
                trial.set_user_attr('scores', scores)
                trial.set_user_attr('final_loss', loss_history[-1] if loss_history else float('inf'))
                trial.set_user_attr('n_iterations', len(loss_history))
                trial.set_user_attr('circuit_info', {
                    'n_qubits': n_qubits,
                    'n_features': X_train.shape[1],
                    'feature_map': feature_map,
                    'ansatz': ansatz_type
                })
                
                # Penalize score if stopped early
                if trial.user_attrs.get('stopped_early', False):
                    logger.info(f"Trial {trial.number} stopped early")
                
                # Return combined score
                trial_duration = time.time() - trial_start
                logger.info(f"Trial {trial.number} completed in {trial_duration:.2f}s")
                logger.info(f"Trial {trial.number} scores: {scores}")
                
                return float(0.4 * scores['accuracy'] + 0.4 * scores['f1'] + 
                           0.1 * scores['precision'] + 0.1 * scores['recall'])
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
                return float('-inf')
        
        # Create and run the study with pruning
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=30,
                interval_steps=10
            )
        )
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get best model that didn't stop early
        best_trial = study.best_trial
        best_model = self._create_model(
            trial=best_trial,
            feature_map_type=feature_map,
            ansatz_type=ansatz_type,
            n_qubits=n_qubits
        )
        best_model.fit(X_train, y_train)
        
        # Store results
        self.best_params[model_name] = best_trial.params
        self.metrics[model_name] = best_trial.user_attrs['scores']
        self.training_times[model_name] = time.time() - start_time
        self.models[model_name] = best_model
        
        # Print convergence information
        print(f"\nBest trial convergence info:")
        print(f"Final loss: {best_trial.user_attrs.get('final_loss', 'N/A')}")
        print(f"Number of iterations: {best_trial.user_attrs.get('n_iterations', 'N/A')}")
        print(f"Stopped early: {best_trial.user_attrs.get('stopped_early', False)}")
        print(f"Circuit info: {best_trial.user_attrs.get('circuit_info', {})}")
        
        training_duration = time.time() - start_time
        logger.info(f"\nTraining completed for {model_name} in {training_duration:.2f}s")
        logger.info(f"Best parameters: {self.best_params[model_name]}")
        logger.info(f"Final metrics: {self.metrics[model_name]}")
        
        return best_model

    def test_comprehensive(self, n_qubits: Optional[int] = None) -> Dict[str, Any]:
        """Run comprehensive tests of all feature map and ansatz combinations."""
        logger = get_logger(__name__)
        logger.info("Starting comprehensive test of all combinations...")
        
        # Use class configurations
        feature_maps = list(self.FEATURE_MAPS.keys())
        ansatze = list(self.ANSATZE.keys())
        optimizers = ['COBYLA', 'SPSA']
        results = {}
        
        try:
            # Load micro dataset with quantum-ready dimensions and reduced size
            X_train, X_test, y_train, y_test = generate_dataset(
                micro=True,
                n_components=n_qubits,
                quantum_mode=True if n_qubits else False,
                max_samples=150  # Use smaller dataset
            )
            logger.info("Using reduced UNSW-NB15 micro dataset")
            
            # Load dataset info for dimension details
            dataset_info = load_dataset_info()
            n_features = dataset_info["n_features"]
            
            # Ensure we have a valid number of qubits
            if n_qubits is None:
                n_qubits = dataset_info["preprocessing"]["n_qubits"] or int(np.ceil(np.log2(n_features)))
            
            if not isinstance(n_qubits, int):
                raise ValueError(f"Invalid number of qubits: {n_qubits}")
            
            logger.info(f"Dataset loaded - Features: {n_features}, Base qubits: {n_qubits}")
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            for feature_map in feature_maps:
                # Get feature map configuration
                feature_map_config = self.FEATURE_MAPS[feature_map]
                uses_qubit_dimension = feature_map_config['uses_qubit_dimension']
                
                # Calculate required qubits for this feature map
                if uses_qubit_dimension:
                    # For quantum feature maps, n_qubits = n_features
                    map_qubits = n_features
                    logger.info(f"{feature_map}: Using {map_qubits} qubits (one per feature)")
                else:
                    # For RawFeatureVector, n_qubits = log2(n_features)
                    map_qubits = n_qubits
                    target_size = 2 ** map_qubits
                    logger.info(f"{feature_map}: Using {map_qubits} qubits for {target_size} features")
                
                # Validate backend can support this feature map
                try:
                    self._validate_backend(map_qubits)
                except ValueError as e:
                    logger.error(f"Backend cannot support {feature_map} (needs {map_qubits} qubits): {str(e)}")
                    continue
                
                # Prepare data for this feature map
                X_train_map = self._prepare_input_data(X_train.copy(), map_qubits, feature_map)
                X_test_map = self._prepare_input_data(X_test.copy(), map_qubits, feature_map)
                
                for ansatz in ansatze:
                    for optimizer_type in optimizers:
                        combo_name = f"{feature_map} + {ansatz} + {optimizer_type}"
                        logger.info(f"\nTesting {combo_name}...")
                        
                        try:
                            # Create a test trial for this optimizer
                            trial = TestTrial(optimizer_type)
                            
                            # Create optimizer
                            optimizer = self._create_optimizer(trial)
                            
                            # Create model with specific circuit types
                            model = self._create_model(
                                feature_map_type=feature_map,
                                ansatz_type=ansatz,
                                n_qubits=map_qubits
                            )
                            # Replace default optimizer
                            model.optimizer = optimizer
                            
                            # Test the model
                            start_time = time.time()
                            model.fit(X_train_map, y_train)
                            training_time = time.time() - start_time
                            
                            pred_start = time.time()
                            y_pred = model.predict(X_test_map)
                            pred_time = time.time() - pred_start
                            
                            scores = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, average='weighted'),
                                'recall': recall_score(y_test, y_pred, average='weighted'),
                                'f1': f1_score(y_test, y_pred, average='weighted')
                            }
                            
                            # Store results
                            trial.set_user_attr('scores', scores)
                            trial.set_user_attr('training_time', training_time)
                            trial.set_user_attr('prediction_time', pred_time)
                            trial.set_user_attr('circuit_info', {
                                'n_qubits': map_qubits,
                                'n_features': X_train_map.shape[1],
                                'feature_map': feature_map,
                                'ansatz': ansatz,
                                'optimizer': optimizer_type,
                                'uses_qubit_dimension': uses_qubit_dimension
                            })
                            
                            # Store results
                            results[combo_name] = {
                                'status': 'success',
                                'metrics': trial.user_attrs['scores'],
                                'parameters': trial.params,
                                'training_time': trial.user_attrs['training_time'],
                                'prediction_time': trial.user_attrs['prediction_time'],
                                'circuit_info': trial.user_attrs['circuit_info']
                            }
                            
                            logger.info(
                                f"Success - Accuracy: {trial.user_attrs['scores']['accuracy']:.4f}, "
                                f"Time: {trial.user_attrs['training_time']:.2f}s, "
                                f"Qubits: {map_qubits}, Features: {X_train_map.shape[1]}, "
                                f"Optimizer: {optimizer_type}"
                            )
                            
                        except Exception as e:
                            logger.error(f"Error testing {combo_name}: {str(e)}", exc_info=True)
                            results[combo_name] = {
                                'status': 'failed',
                                'error': f"{type(e).__name__}: {str(e)}",
                                'circuit_info': {
                                    'n_qubits': map_qubits,
                                    'n_features': X_train_map.shape[1] if 'X_train_map' in locals() else None,
                                    'feature_map': feature_map,
                                    'ansatz': ansatz,
                                    'optimizer': optimizer_type,
                                    'uses_qubit_dimension': uses_qubit_dimension
                                }
                            }
            
            # Log summary
            logger.info("\nTest Summary:")
            for combo, result in results.items():
                status = result['status']
                if status == 'success':
                    logger.info(
                        f"{combo}: Accuracy={result['metrics']['accuracy']:.4f}, "
                        f"Time={result['training_time']:.2f}s, "
                        f"Qubits={result['circuit_info']['n_qubits']}, "
                        f"Features={result['circuit_info']['n_features']}, "
                        f"Optimizer={result['circuit_info']['optimizer']}"
                    )
                else:
                    logger.info(f"{combo}: Failed - {result['error']}")
            
        except Exception as e:
            logger.error("Failed to run comprehensive tests", exc_info=True)
            raise RuntimeError(f"Comprehensive testing failed: {str(e)}")
        
        return results

    def _validate_backend(self, n_qubits: int) -> None:
        """Validate that the backend can support the required number of qubits."""
        logger = get_logger(__name__)
        
        if n_qubits > self.backend.num_qubits:
            raise ValueError(
                f"Backend {self.backend.name} only supports {self.backend.num_qubits} qubits, "
                f"but {n_qubits} qubits are required."
            )
        
        logger.debug(f"Backend validation passed: {n_qubits} qubits required, "
                    f"{self.backend.num_qubits} available")


if __name__ == "__main__":
    print("Running comprehensive tests...")
    
    # Initialize backend and trainer
    backend = GenericBackendV2(num_qubits=4)  # Backend needs enough qubits for all tests
    trainer = VQCTrainer(
        backend=backend,
        study_name="comprehensive_test",
        storage_path="quantum_optimization.db",
        n_trials=5
    )
    
    # Run comprehensive tests with fixed number of qubits
    print("\nTesting with 2 qubits (4 features) on reduced dataset (150 samples)...")
    results = trainer.test_comprehensive(n_qubits=2)
