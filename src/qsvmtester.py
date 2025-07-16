from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from dataset import generate_dataset
from logger import get_logger
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score
import os
import argparse
from typing import Tuple, Optional

def parse_qsvm_args() -> argparse.Namespace:
    """Parse command-line arguments for QSVM tester."""
    parser = argparse.ArgumentParser(description='QSVM Training and Evaluation')
    parser.add_argument('--n-features', type=int, default=8, help='Number of features for dimensionality reduction.')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples to use from the dataset.')
    parser.add_argument('--dataset-type', type=str, default="EdgeIIoT", help='Type of dataset to use.')
    return parser.parse_args()

class QSVMTester:
    """QSVM tester for training a model with a fixed configuration."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def _create_feature_map(
        self, n_qubits: int, paulis: list, reps: int, entanglement: str
    ) -> PauliFeatureMap:
        """Create a feature map with the specified parameters."""
        self.logger.info(
            f"Creating PauliFeatureMap with paulis={paulis}, reps={reps}, entanglement='{entanglement}'"
        )
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            paulis=paulis,
            entanglement=entanglement,
            insert_barriers=False,
        )
    
    def _create_qsvm(self, feature_map: PauliFeatureMap, num_steps: int, device: str) -> PegasosQSVC:
        """Create QSVM with the specified parameters."""
        self.logger.info(f"Using {device} backend for QSVM.")

        backend_options = {
            "method": "statevector",
            "device": device.upper()
        }
        sampler = AerSamplerV2(options={"backend_options": backend_options})
        fidelity = ComputeUncompute(sampler=sampler)
        
        transpiled_feature_map = transpile(feature_map, backend=AerSimulator())
        qkernel = FidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=transpiled_feature_map
        )
        
        return PegasosQSVC(
            quantum_kernel=qkernel,
            C=1000.0,
            num_steps=num_steps
        )
    
    def train_and_evaluate(
        self, 
        n_features: int, 
        n_samples: int, 
        dataset_type: str, 
        feature_map_params: dict,
        num_steps: int,
        device: str,
        experiment_id: str,
    ) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Train and evaluate the QSVM model and return metrics and predictions."""
        
        self.logger.info(f"--- Starting Experiment: {experiment_id} ---")

        try:
            X_train, X_test, y_train, y_test = generate_dataset(
                micro=False,  # Use full dataset slices
                n_components=n_features,
                max_samples=n_samples,
                dataset_type=dataset_type,
            )

            feature_map = self._create_feature_map(n_features, **feature_map_params)
            model = self._create_qsvm(feature_map, num_steps=num_steps, device=device)
            
            self.logger.info("Starting QSVM training...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.logger.info("Evaluating model...")
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)
            
            self.logger.info("\n--- QSVM Evaluation Results ---")
            self.logger.info(f"  Accuracy: {acc:.4f}")
            self.logger.info(f"  F1 Score: {f1:.4f}")
            self.logger.info(f"  Training Time: {training_time:.2f}s")
            self.logger.info(f"  Dataset Size: {n_samples} samples")
            self.logger.info(f"--- Finished Experiment: {experiment_id} ---")
            
            return float(acc), float(f1), float(training_time), y_test, y_pred
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed", exc_info=True)
            return 0.0, 0.0, -1.0, None, None

if __name__ == "__main__":
    args = parse_qsvm_args()
    tester = QSVMTester()
    
    default_feature_map_params = {
        "paulis": ["X", "Y", "Z"],
        "reps": 1,
        "entanglement": "full",
    }
    
    tester.train_and_evaluate(
        n_features=args.n_features, 
        n_samples=args.n_samples, 
        dataset_type=args.dataset_type,
        feature_map_params=default_feature_map_params,
        num_steps=50,
        device="CPU",
        experiment_id="default_main"
    ) 