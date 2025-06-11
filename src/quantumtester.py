from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Type, Sequence, Tuple, Iterable, Callable, Protocol, Final, cast, TypeVar
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    RealAmplitudes, EfficientSU2,
    ExcitationPreserving, TwoLocal, PauliFeatureMap
)
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import BackendSamplerV2
from qiskit_machine_learning.algorithms import VQC, PegasosQSVC
from qiskit.providers import BackendV2
from qiskit_machine_learning.optimizers import COBYLA, SPSA, GradientDescent, ADAM
import optuna
from optuna import Trial
from optuna.trial import FrozenTrial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from pathlib import Path
import logging
from datetime import datetime
from dataset import generate_dataset, load_dataset_info
from logger import get_logger
from optuna.samplers import NSGAIIISampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler

# Type definitions using Protocol for better type safety
class FeatureMapProtocol(Protocol):
    def __init__(self, feature_dimension: int, **kwargs: Any) -> None: ...

class AnsatzProtocol(Protocol):
    def __init__(self, num_qubits: int, **kwargs: Any) -> None: ...

class OptimizerProtocol(Protocol):
    def __init__(self, maxiter: int = 100, **kwargs: Any) -> None: ...

# Concrete circuit types
AnsatzImpl = Union[EfficientSU2, ExcitationPreserving, RealAmplitudes, TwoLocal]

class AnsatzEnum(Enum):
    EFFICIENT_SU2 = auto()
    EXCITATION_PRESERVING = auto()
    REAL_AMPLITUDES = auto()
    TWO_LOCAL = auto()

@dataclass(frozen=True)
class AnsatzConfig:
    circuit_class: Type[AnsatzImpl]
    reps: int = 2
    entanglement: str = 'linear'
    rotation_blocks: Optional[List[str]] = None
    entanglement_blocks: Optional[str] = None

@dataclass(frozen=True)
class CircuitMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    prediction_time: float
    n_qubits: int
    n_features: int

    @classmethod
    def from_scores(cls, 
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   training_time: float,
                   prediction_time: float,
                   n_qubits: int,
                   n_features: int) -> 'CircuitMetrics':
        """Create metrics from raw scores, handling type conversions."""
        return cls(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            training_time=training_time,
            prediction_time=prediction_time,
            n_qubits=n_qubits,
            n_features=n_features
        )

@dataclass(frozen=True)
class TestResult:
    ansatz: AnsatzEnum
    optimizer: str
    metrics: Optional[CircuitMetrics] = None
    error: Optional[str] = None

    @property
    def status(self) -> str:
        return "success" if self.metrics is not None else "failed"

class QuantumTester:
    """Quantum circuit testing class with NSGA-III optimization."""

    ANSATZE: Final[Dict[AnsatzEnum, AnsatzConfig]] = {
        AnsatzEnum.EFFICIENT_SU2: AnsatzConfig(
            circuit_class=EfficientSU2
        ),
        AnsatzEnum.EXCITATION_PRESERVING: AnsatzConfig(
            circuit_class=ExcitationPreserving
        ),
        AnsatzEnum.REAL_AMPLITUDES: AnsatzConfig(
            circuit_class=RealAmplitudes
        ),
        AnsatzEnum.TWO_LOCAL: AnsatzConfig(
            circuit_class=TwoLocal,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz'
        )
    }

    def __init__(self, 
                 backend: BackendV2,
                 study_name: str = "quantum_test_study",
                 storage_path: str = "quantum_test.db",
                 n_trials: int = 100):
        """Initialize quantum tester with specific backend and Optuna settings."""
        self.backend = backend
        self.sampler = BackendSamplerV2(backend=self.backend)
        self.logger = get_logger(__name__)
        self.results: Dict[str, TestResult] = {}
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials

    def _create_optimizer(self, optimizer_type: str, trial: Trial) -> Any:
        """Create optimizer based on trial parameters."""
        # Add optimizer prefix to parameter names
        prefix = f"{optimizer_type.lower()}_"
        
        optimizer_params: Dict[str, Any] = {}
        
        if optimizer_type == 'COBYLA':
            optimizer_params.update({
                'maxiter': 100,
                'tol': trial.suggest_categorical(f'{prefix}tol', [1e-6, 1e-5, 1e-4])
            })
            return COBYLA(**optimizer_params)
            
        else:  # SPSA
            optimizer_params.update({
                'maxiter': 100,
                'learning_rate': trial.suggest_categorical(f'{prefix}learning_rate', [0.01, 0.05, 0.1]),
                'perturbation': trial.suggest_categorical(f'{prefix}perturbation', [0.01, 0.05, 0.1]),
                'resamplings': trial.suggest_int(f'{prefix}resamplings', 1, 3),
                'trust_region': True,
                'blocking': True,
                'second_order': trial.suggest_categorical(f'{prefix}second_order', [True, False])
            })
            return SPSA(**optimizer_params)

    def _create_feature_map(self, n_qubits: int, trial: Trial, model_type: str) -> QuantumCircuit:
        """Create optimized feature map based on model type."""
        # Add model type prefix to parameter names
        prefix = f"{model_type.lower()}_"
        
        # Common entanglement options for both models
        entanglement_options = ['linear', 'circular', 'full', 'sca']
        
        if model_type == 'QSVM':
            # QSVM-specific feature map options
            rotation_gates = {
                'RX': ['X'],
                'RY': ['Y'],
                'RZ': ['Z'],
                'MIXED': ['X', 'Y', 'Z']
            }
        else:  # VQC
            # VQC-specific feature map options
            rotation_gates = {
                'Z': ['Z'],
                'ZZ': ['Z', 'ZZ'],
                'XX': ['X', 'XX'],
                'YY': ['Y', 'YY'],
                'ZZZ': ['Z', 'ZZ', 'ZZZ'],
                'MIXED': ['X', 'Y', 'Z', 'XY', 'YZ', 'ZX']
            }
        
        # Layer structure optimization with unique parameter names
        n_layers = trial.suggest_categorical(f'{prefix}feature_map_layers', [1, 2, 3])
        gate_type = trial.suggest_categorical(f'{prefix}rotation_type', list(rotation_gates.keys()))
        entanglement = trial.suggest_categorical(f'{prefix}entanglement', entanglement_options)
        
        # Scaling parameter with categorical choices
        alpha_choices = [0.1, 0.5, 1.0, 1.5, 2.0]
        alpha = trial.suggest_categorical(f'{prefix}alpha', alpha_choices)
        
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=n_layers,
            paulis=rotation_gates[gate_type],
            alpha=alpha,
            entanglement=entanglement,
            insert_barriers=False
        )

    def _create_ansatz(self, 
                      ansatz_type: AnsatzEnum, 
                      n_qubits: int,
                      trial: Trial) -> QuantumCircuit:
        """Create ansatz with optimized structure."""
        # Add VQC prefix to parameter names since ansatz is only used for VQC
        prefix = "vqc_"
        
        # Optimize circuit depth
        n_layers = trial.suggest_int(f'{prefix}ansatz_layers', 1, 4)
        
        # Optimize rotation gates
        rotation_blocks = trial.suggest_categorical(
            f'{prefix}rotation_blocks',
            ['rx,ry,rz', 'rx+rz', 'ry+rz', 'rx+ry+rz']
        ).replace('+', ',').split(',')
        
        # Optimize entangling gates
        entangling_blocks = trial.suggest_categorical(
            f'{prefix}entangling_blocks',
            ['cz', 'cx', 'crx', 'cry', 'crz']
        )
        
        # Optimize entanglement pattern
        entanglement = trial.suggest_categorical(
            f'{prefix}ansatz_entanglement',
            ['full', 'linear', 'circular', 'sca']
        )
        
        return TwoLocal(
            num_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entangling_blocks,
            entanglement=entanglement,
            reps=n_layers,
            insert_barriers=False,
            skip_final_rotation_layer=False
        )

    def _create_qsvm(self, feature_map: QuantumCircuit, trial: Trial) -> PegasosQSVC:
        """Create QSVM with optimized parameters."""
        # Add QSVM prefix to parameter names
        prefix = "qsvm_"
        
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        qkernel = FidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map
        )
        
        # Categorical choices for QSVM parameters
        num_steps = trial.suggest_categorical(f'{prefix}num_steps', [50, 100, 150, 200])
        C = trial.suggest_categorical(f'{prefix}C', [0.1, 1.0, 10.0, 100.0, 1000.0])
        
        return PegasosQSVC(
            quantum_kernel=qkernel,
            C=C,
            num_steps=num_steps
        )

    def test_all_combinations(self, n_features: int = 4) -> Dict[str, TestResult]:
        """Test all combinations with NSGA-III optimization."""
        try:
            n_qubits = n_features
            
            # Validate backend
            if n_qubits > self.backend.num_qubits:
                raise ValueError(
                    f"Backend only supports {self.backend.num_qubits} qubits, "
                    f"but {n_qubits} are required."
                )
            
            # Before the study creation, define better reference points
            reference_points = np.array([
                [1.0, 0.0, 0.0],  # Maximize F1
                [0.0, 1.0, 0.0],  # Minimize complexity
                [0.0, 0.0, 1.0],  # Minimize time
                [0.5, 0.25, 0.25],  # Balanced towards F1
                [0.33, 0.33, 0.33],  # Equal balance
            ])
            
            # Create study with NSGA-III sampler once for all folds
            sampler = NSGAIIISampler(
                population_size=100,  # Increased from 50
                crossover_prob=0.9,
                mutation_prob=0.2,  # Increased from 0.1 for better exploration
                seed=42,
                reference_points=reference_points
            )
            
            try:
                study = optuna.load_study(  # Try to load existing study first
                    study_name=self.study_name,
                    storage=f"sqlite:///./{self.storage_path}"
                )
            except Exception:
                # If study doesn't exist, create a new one
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=f"sqlite:///./{self.storage_path}",
                    directions=["minimize", "minimize", "minimize"],
                    sampler=sampler,
                    load_if_exists=True
                )
            
            # Set metric names after study creation
            study.set_metric_names(["negative_f1_score", "circuit_complexity", "training_time"])
            
            # Track best results across folds
            fold_metrics = []
            n_splits = 5
            
            for fold_idx in range(n_splits):
                self.logger.info(f"\nProcessing fold {fold_idx + 1}/{n_splits}")
                
                X_train, X_test, y_train, y_test = generate_dataset(
                    micro=True,
                    n_components=n_features,
                    quantum_mode=False,
                    max_samples=100,
                    n_splits=n_splits,
                    fold_idx=fold_idx
                )
                
                def objective(trial: Trial) -> Tuple[float, float, float]:
                    """Multi-objective optimization function."""
                    try:
                        # Choose between VQC and QSVM first
                        model_type = trial.suggest_categorical('model_type', ['VQC', 'QSVM'])
                        
                        # Create feature map based on model type
                        feature_map = self._create_feature_map(n_qubits, trial, model_type)
                        
                        if model_type == 'VQC':
                            # Create VQC components
                            ansatz_name = trial.suggest_categorical(
                                'ansatz', [ansatz.name for ansatz in AnsatzEnum]
                            )
                            ansatz = AnsatzEnum[ansatz_name]
                            ansatz_circuit = self._create_ansatz(ansatz, n_qubits, trial)
                            
                            # Only suggest optimizer for VQC
                            optimizer_type = trial.suggest_categorical(
                                'optimizer', ['COBYLA', 'SPSA']
                            )
                            optimizer = self._create_optimizer(optimizer_type, trial)
                            
                            model = VQC(
                                feature_map=feature_map,
                                ansatz=ansatz_circuit,
                                optimizer=optimizer,
                                loss="cross_entropy"
                            )
                        else:  # QSVM
                            model = self._create_qsvm(feature_map, trial)
                        
                        # Train and evaluate model using preprocessed data directly
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        pred_start = time.time()
                        y_pred = model.predict(X_test)
                        pred_time = time.time() - pred_start
                        
                        # Calculate metrics
                        metrics = CircuitMetrics.from_scores(
                            y_true=y_test,
                            y_pred=y_pred,
                            training_time=training_time,
                            prediction_time=pred_time,
                            n_qubits=n_qubits,
                            n_features=n_features
                        )
                        
                        # Store metrics in trial
                        trial.set_user_attr('metrics', metrics.__dict__)
                        trial.set_user_attr('model_type', model_type)
                        if model_type == 'VQC':
                            trial.set_user_attr('ansatz', ansatz_name)
                            trial.set_user_attr('optimizer', trial.params['optimizer'])
                        
                        # Calculate circuit complexity
                        circuit_complexity = len(feature_map.parameters)
                        if model_type == 'VQC':
                            circuit_complexity += len(ansatz_circuit.parameters)
                        
                        return (
                            -metrics.f1,  # Return negative F1 since we're minimizing
                            circuit_complexity,
                            metrics.training_time
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Trial failed: {str(e)}", exc_info=True)
                        return (float('inf'), float('inf'), float('inf'))
                
                # Run optimization with fewer trials
                study.optimize(objective, n_trials=200)
                
                # Get Pareto front
                pareto_trials = [
                    trial for trial in study.best_trials 
                    if trial.values[0] != float('inf')
                ]
                
                if pareto_trials:
                    best_trial = min(
                        pareto_trials,
                        key=lambda t: (
                            0.6 * t.values[0] +     # More weight on negative F1 score
                            0.2 * (t.values[1] / max(trial.values[1] for trial in pareto_trials)) +  # Normalized complexity
                            0.2 * (t.values[2] / max(trial.values[2] for trial in pareto_trials))    # Normalized time
                        )
                    )
                    metrics = CircuitMetrics(**best_trial.user_attrs['metrics'])
                    fold_metrics.append((best_trial, metrics))
                    
                    self.logger.info(f"Fold {fold_idx + 1} best F1: {metrics.f1:.4f}")
            
            # Aggregate results across folds
            if fold_metrics:
                # Select best configuration based on average F1 score
                best_fold_idx = np.argmin([m.f1 for _, m in fold_metrics])  # Use argmin since we're working with negative F1
                best_trial, best_metrics = fold_metrics[best_fold_idx]
                
                model_type = best_trial.user_attrs['model_type']
                if model_type == 'VQC':
                    combo_name = f"PAULI_VQC_{best_trial.user_attrs['ansatz']}"
                    self.results[combo_name] = TestResult(
                        ansatz=AnsatzEnum[best_trial.user_attrs['ansatz']],
                        optimizer=best_trial.user_attrs['optimizer'],
                        metrics=best_metrics
                    )
                else:  # QSVM
                    combo_name = "PAULI_QSVM"
                    self.results[combo_name] = TestResult(
                        ansatz=AnsatzEnum.EFFICIENT_SU2,  # Placeholder for QSVM
                        optimizer="QSVM",
                        metrics=best_metrics
                    )
                
                # Log final results
                self.logger.info("\nBest Configuration Across Folds:")
                self.logger.info(f"Average F1 Score: {np.mean([m.f1 for _, m in fold_metrics]):.4f}")
                self.logger.info(f"Std Dev F1 Score: {np.std([m.f1 for _, m in fold_metrics]):.4f}")
                self.logger.info(f"Best Fold: {best_fold_idx + 1}")
                self.logger.info(f"Best F1 Score: {best_metrics.f1:.4f}")
                
                if model_type == 'VQC':
                    self.logger.info(f"Model Type: VQC")
                    self.logger.info(f"Ansatz: {best_trial.user_attrs['ansatz']}")
                    self.logger.info(f"Rotation Gates: {best_trial.params['rotation_blocks']}")
                    self.logger.info(f"Entangling Gates: {best_trial.params['entangling_blocks']}")
                    self.logger.info(f"Ansatz Layers: {best_trial.params['ansatz_layers']}")
                else:
                    self.logger.info(f"Model Type: QSVM")
                    
                self.logger.info(f"Feature Map:")
                self.logger.info(f"  Rotation Type: {best_trial.params['rotation_type']}")
                self.logger.info(f"  Layers: {best_trial.params['feature_map_layers']}")
                self.logger.info(f"  Entanglement: {best_trial.params['entanglement']}")
                self.logger.info(f"  Alpha: {best_trial.params['alpha']:.2f}")
            
            else:
                self.logger.error("All trials failed across all folds")
                    
        except Exception as e:
            self.logger.error("Failed to run optimization", exc_info=True)
            raise RuntimeError(f"Testing failed: {str(e)}")
        
        return self.results

if __name__ == "__main__":
    # Initialize backend and tester
    backend = GenericBackendV2(num_qubits=8)
    tester = QuantumTester(
        backend=backend,
        study_name="quantum_pauli_optimization",
        storage_path="quantum_pauli_optimization.db",
        n_trials=200  # Reduced from 2000 to 200
    )
    
    # Run optimization with 8 features
    print("\nOptimizing quantum circuits with 8 features...")
    results = tester.test_all_combinations(n_features=8)