from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from dataset import generate_dataset
from logger import get_logger
import numpy as np
import time
from sklearn.metrics import f1_score
import optuna
from optuna import Trial
from optuna.samplers import NSGAIIISampler
from typing import Tuple

class QSVMTester:
    """Enhanced QSVM tester with circuit and hyperparameter optimization."""
    
    def __init__(self, study_name: str = "qsvm_optimization"):
        self.logger = get_logger(__name__)
        self.study_name = study_name
        
    def _create_feature_map(self, n_qubits: int, trial: Trial) -> PauliFeatureMap:
        """Create optimized feature map."""
        # Gate types for data encoding
        rotation_gates = {
            'RX': ['X'],
            'RY': ['Y'],
            'RZ': ['Z'],
            'MIXED': ['X', 'Y', 'Z']
        }
        
        # Layer structure optimization
        n_layers = trial.suggest_int('feature_map_layers', 1, 3)
        gate_type = trial.suggest_categorical('rotation_type', list(rotation_gates.keys()))
        
        # Entanglement pattern
        entanglement = trial.suggest_categorical(
            'entanglement', 
            ['full', 'linear', 'circular', 'sca', 'none']
        )
        
        # Scaling parameter for data embedding
        alpha = trial.suggest_float('alpha', 0.1, 2.0, log=True)
        
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=n_layers,
            paulis=rotation_gates[gate_type],
            alpha=alpha,
            entanglement=entanglement,
            insert_barriers=False
        )
    
    def _create_qsvm(self, feature_map: PauliFeatureMap, trial: Trial) -> PegasosQSVC:
        """Create QSVM with optimized parameters."""
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        qkernel = FidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map
        )
        
        # Optimize QSVM hyperparameters
        num_steps = trial.suggest_categorical('num_steps', [50, 100, 150, 200])
        C = trial.suggest_categorical('C', [0.1, 1.0, 10.0, 100.0, 1000.0])
        
        return PegasosQSVC(
            quantum_kernel=qkernel,
            C=C,
            num_steps=num_steps
        )
    
    def optimize_qsvm(self, n_features: int = 8, n_samples: int = 100, n_trials: int = 50) -> optuna.Trial:
        """Optimize QSVM with multi-objective optimization."""
        try:
            # Create study with NSGA-III sampler
            sampler = NSGAIIISampler(
                population_size=20,
                seed=42,
                reference_points=np.array([
                    [1.0, 0.0, 0.0],  # Maximize F1
                    [0.0, 1.0, 0.0],  # Minimize complexity
                    [0.0, 0.0, 1.0],  # Minimize time
                ])
            )
            
            study = optuna.create_study(
                study_name=self.study_name,
                storage=None,  # In-memory storage
                directions=["minimize", "minimize", "minimize"],
                sampler=sampler
            )
            
            # Set metric names
            study.set_metric_names(["negative_f1_score", "circuit_complexity", "training_time"])
            
            def objective(trial: Trial) -> Tuple[float, float, float]:
                """Multi-objective optimization function."""
                try:
                    # Load dataset
                    X_train, X_test, y_train, y_test = generate_dataset(
                        micro=True,
                        n_components=8,
                        quantum_mode=False,
                        max_samples=n_samples
                    )
                    
                    
                    # Create optimized feature map and QSVM
                    feature_map = self._create_feature_map(n_features, trial)
                    model = self._create_qsvm(feature_map, trial)
                    
                    # Train and evaluate
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Calculate circuit complexity
                    circuit_complexity = len(feature_map.parameters)
                    
                    # Ensure float returns
                    return float(-f1), float(circuit_complexity), float(training_time)
                    
                except Exception as e:
                    self.logger.error(f"Trial failed: {str(e)}", exc_info=True)
                    return float('inf'), float('inf'), float('inf')
            
            # Run optimization
            self.logger.info(f"\nStarting QSVM optimization with {n_trials} trials...")
            
            for trial_num in range(n_trials):
                trial = study.ask()
                try:
                    values = objective(trial)
                    study.tell(trial, values)
                    
                    # Log progress
                    self.logger.info(
                        f"\nTrial {trial_num + 1}/{n_trials}:"
                        f"\n  F1 Score: {-values[0]:.4f}"
                        f"\n  Circuit Complexity: {values[1]}"
                        f"\n  Training Time: {values[2]:.2f}s"
                        f"\n  Feature Map:"
                        f"\n    Type: {trial.params['rotation_type']}"
                        f"\n    Layers: {trial.params['feature_map_layers']}"
                        f"\n    Entanglement: {trial.params['entanglement']}"
                        f"\n    Alpha: {trial.params['alpha']:.3f}"
                        f"\n  QSVM Parameters:"
                        f"\n    C: {trial.params['C']:.3f}"
                        f"\n    Steps: {trial.params['num_steps']}"
                    )
                except Exception as e:
                    self.logger.warning(f"Trial {trial_num + 1} failed: {str(e)}")
                    study.tell(trial, (float('inf'), float('inf'), float('inf')))
            
            # Get best trial
            best_trial = min(
                study.best_trials,
                key=lambda t: 0.6 * (-t.values[0]) + 0.2 * t.values[1] + 0.2 * t.values[2]
            )
            
            # Log results
            self.logger.info("\nBest QSVM Configuration:")
            self.logger.info(f"F1 Score: {-best_trial.values[0]:.4f}")
            self.logger.info(f"Circuit Complexity: {best_trial.values[1]}")
            self.logger.info(f"Training Time: {best_trial.values[2]:.2f}s")
            self.logger.info("\nFeature Map Configuration:")
            self.logger.info(f"Rotation Type: {best_trial.params['rotation_type']}")
            self.logger.info(f"Layers: {best_trial.params['feature_map_layers']}")
            self.logger.info(f"Entanglement: {best_trial.params['entanglement']}")
            self.logger.info(f"Alpha: {best_trial.params['alpha']:.2f}")
            self.logger.info("\nQSVM Parameters:")
            self.logger.info(f"C: {best_trial.params['C']:.2f}")
            self.logger.info(f"Number of Steps: {best_trial.params['num_steps']}")
            
            return best_trial
            
        except Exception as e:
            self.logger.error("Optimization failed", exc_info=True)
            raise

if __name__ == "__main__":
    tester = QSVMTester()
    print("\nOptimizing QSVM configuration...")
    tester.optimize_qsvm(n_features=8, n_samples=100, n_trials=50) 