from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import BackendSamplerV2
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.optimizers import SPSA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
from dataset import generate_dataset
from logger import get_logger
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer import QasmSimulator
import os
import cupy as cp  # For GPU array operations
from qiskit.providers.aer.backends.exceptions import QiskitError

# Set environment variables for performance
os.environ['QISKIT_PARALLEL'] = 'TRUE'
os.environ['QISKIT_NUM_PROCS'] = '8'  # Adjust based on your CPU
os.environ['OMP_NUM_THREADS'] = '8'    # OpenMP threads

def check_gpu_availability():
    """Check if GPU is available and properly configured"""
    try:
        # Check if cupy can access GPU
        cp.cuda.runtime.getDeviceCount()
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False

def evaluate_best_circuit():
    """Evaluate the best quantum circuit configuration on the full dataset."""
    logger = get_logger(__name__)
    
    # Check GPU availability
    if not check_gpu_availability():
        logger.warning("GPU not available. Falling back to CPU...")
        device = 'CPU'
        parallel_experiments = True
        max_parallel_threads = 8
    else:
        logger.info("GPU available. Using GPU acceleration...")
        device = 'GPU'
        parallel_experiments = False
        max_parallel_threads = 1
    
    # Initialize backend with dynamic device selection
    try:
        backend = AerSimulator(
            method='statevector',
            device=device,
            max_parallel_experiments=1 if device == 'GPU' else 8,
            cuStateVec_enable=device == 'GPU'
        )
    except QiskitError as e:
        logger.error(f"Error initializing backend: {e}")
        logger.warning("Falling back to CPU backend...")
        backend = AerSimulator(
            method='statevector',
            device='CPU',
            max_parallel_threads=8
        )
    
    # Create quantum instance with parallel settings
    quantum_instance = QuantumInstance(
        backend=backend,
        shots=2048,
        seed_simulator=42,
        seed_transpiler=42,
        optimization_level=3,
        parallel_experiments=False,  # GPU works better with sequential execution
        max_parallel_threads=1,      # Let GPU handle parallelization
        memory=True,
        skip_qobj_validation=True
    )
    
    # Load dataset with reduced size
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test = generate_dataset(
        micro=False,
        n_components=3,
        quantum_mode=True,
        max_samples=10000
    )
    
    # Create circuit components with best parameters
    feature_map = RawFeatureVector(feature_dimension=8)
    ansatz = RealAmplitudes(num_qubits=3, reps=1)
    
    # Create optimizer with optimized parameters
    optimizer_params = {
        "maxiter": 50,  # Reduced from 200
        "learning_rate": 0.08461183051421624,
        "perturbation": 0.04516196044019855,
        "trust_region": True,
        "resamplings": 1
    }
    
    optimizer = SPSA(**optimizer_params)
    
    # Implement batch processing for training
    batch_size = 128  # Adjust based on your memory constraints
    n_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
    
    class BatchCallback:
        def __init__(self):
            self.loss_history = []
            self.iteration_history = []
            self.time_history = []
            self.iteration = 0
            self.start_time = time.time()
            
        def __call__(self, weights, objective_value, batch_index):
            self.iteration += 1
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            self.loss_history.append(objective_value)
            self.iteration_history.append(self.iteration)
            self.time_history.append(elapsed)
            
            if self.iteration % 10 == 0:
                remaining = (optimizer_params["maxiter"] - self.iteration) * (elapsed / self.iteration)
                eta = datetime.now() + timedelta(seconds=remaining)
                
                logger.info(
                    f"Iteration {self.iteration}/{optimizer_params['maxiter']} "
                    f"(Loss: {objective_value:.6f}, Batch: {batch_index}/{n_batches}) | "
                    f"ETA: {eta.strftime('%H:%M:%S')}"
                )
    
    callback = BatchCallback()
    
    # Create VQC model with batch processing capability
    class BatchVQC(VQC):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._circuit_cache = {}
            self._compiled_circuits = None
            self._device = device  # Store device type
            
        def _gpu_memory_cleanup(self):
            """Clean up GPU memory"""
            if self._device == 'GPU':
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        
        def _prepare_circuits(self):
            """Pre-compile common circuits for faster execution"""
            if self._compiled_circuits is None:
                # Create base circuit templates
                base_circuit = self.construct_circuit(cp.zeros(8))  # Use cupy instead of numpy
                self._compiled_circuits = {
                    'base': self.quantum_instance.transpile(base_circuit)
                }
        
        def _parallel_predict(self, X_batch):
            """Execute predictions in parallel"""
            # Convert input to GPU array
            X_batch_gpu = cp.array(X_batch)
            circuits = []
            
            for x in X_batch_gpu:
                if str(x.get()) in self._circuit_cache:  # Get CPU array for hashing
                    circuits.append(self._circuit_cache[str(x.get())])
                else:
                    circuit = self.construct_circuit(x)
                    self._circuit_cache[str(x.get())] = circuit
                    circuits.append(circuit)
            
            # Execute circuits sequentially on GPU
            results = self.quantum_instance.execute(circuits)
            return [self._interpret_results(results, i) for i in range(len(circuits))]
        
        def fit(self, X, y):
            """Override fit method to implement batch processing"""
            n_samples = len(X)
            indices = cp.arange(n_samples)  # Use cupy array
            
            # Pre-compile circuits
            self._prepare_circuits()
            
            # Pre-allocate memory for results
            batch_results = cp.zeros(n_batches)
            
            for epoch in range(optimizer_params["maxiter"]):
                cp.random.shuffle(indices)
                batch_losses = []
                
                # Process batches sequentially (GPU handles parallelization)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size].get()  # Convert to CPU for indexing
                    X_batch = cp.array(X[batch_indices])  # Move batch to GPU
                    y_batch = cp.array(y[batch_indices])
                    
                    # Process batch
                    batch_idx, batch_loss = self._process_batch(X_batch, y_batch, i // batch_size)
                    batch_results[batch_idx] = batch_loss
                    
                    current_batch = batch_idx + 1
                    epoch_loss = float(cp.mean(batch_results[:current_batch]) * n_samples)
                    callback(self.weights, epoch_loss, current_batch)
            
            return self
        
        def _process_batch(self, X_batch, y_batch, batch_idx):
            """Process a single batch in parallel"""
            try:
                if self._device == 'GPU':
                    X_batch_cpu = cp.asnumpy(X_batch)
                    y_batch_cpu = cp.asnumpy(y_batch)
                else:
                    X_batch_cpu = X_batch
                    y_batch_cpu = y_batch
                
                super().fit(X_batch_cpu, y_batch_cpu)
                batch_loss = self.fit_result.fun if self.fit_result is not None else 0
                return batch_idx, batch_loss
            finally:
                if self._device == 'GPU':
                    self._gpu_memory_cleanup()
        
        def predict(self, X):
            """Override predict method with parallel processing"""
            predictions = []
            
            # Process predictions in batches
            for i in range(0, len(X), batch_size):
                X_batch = cp.array(X[i:i + batch_size])  # Move batch to GPU
                batch_pred = self._parallel_predict(X_batch)
                predictions.extend(cp.asnumpy(batch_pred))  # Convert back to CPU
            
            return np.array(predictions)
    
    # Create and train model
    logger.info("Training quantum model with batch processing...")
    model = BatchVQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        loss="cross_entropy",
        quantum_instance=quantum_instance  # Add quantum instance
    )
    
    # Training with batch processing
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions with batch processing
    logger.info("Making predictions...")
    pred_start = time.time()
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    prediction_time = time.time() - pred_start
    
    # Record dataset info
    dataset_info = {
        "total_samples": len(X_train) + len(X_test),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "original_features": X_train.shape[1],
        "quantum_features": 8,  # After PCA
        "class_distribution_train": {str(label): int(count) 
            for label, count in zip(*np.unique(y_train, return_counts=True))},
        "class_distribution_test": {str(label): int(count) 
            for label, count in zip(*np.unique(y_test, return_counts=True))}
    }
    
    logger.info("Using preprocessed quantum data...")
    
    # Move this block up - right after training_time calculation
    results = {
        "timestamp": timestamp,
        "dataset_info": dataset_info,
        "model_config": {
            "feature_map": "RawFeatureVector",
            "ansatz": "RealAmplitudes",
            "n_qubits": 3,
            "reps": 1,
            "optimizer": "SPSA",
            "optimizer_params": optimizer_params
        }
    }
    
    # Then do the plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss vs iterations
    ax1.plot(callback.iteration_history, callback.loss_history)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Iterations')
    ax1.grid(True)
    
    # Plot loss vs time
    ax2.plot(callback.time_history, callback.loss_history)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss vs Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_file = results_dir / "training_history.png"
    plt.savefig(plot_file)
    plt.close()
    
    # Then add training history
    results['training_history'] = {
        'loss': [float(x) for x in callback.loss_history],
        'iterations': callback.iteration_history,
        'times': callback.time_history
    }
    
    # Calculate metrics
    metrics = {
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1": float(f1_score(y_test, y_pred, average='weighted')),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        },
        "train": {
            "accuracy": float(accuracy_score(y_train, y_pred_train)),
            "precision": float(precision_score(y_train, y_pred_train, average='weighted')),
            "recall": float(recall_score(y_train, y_pred_train, average='weighted')),
            "f1": float(f1_score(y_train, y_pred_train, average='weighted')),
            "confusion_matrix": confusion_matrix(y_train, y_pred_train).tolist()
        }
    }
    
    try:
        metrics["test"]["auc_roc"] = float(roc_auc_score(y_test, y_pred))
        metrics["train"]["auc_roc"] = float(roc_auc_score(y_train, y_pred_train))
    except:
        logger.warning("Could not calculate AUC-ROC score")
    
    # Timing metrics
    timing = {
        "training_time": training_time,
        "prediction_time": prediction_time,
        "prediction_time_per_sample": prediction_time / len(X_test)
    }
    
    # Save all results
    results = {
        "timestamp": timestamp,
        "dataset_info": dataset_info,
        "model_config": {
            "feature_map": "RawFeatureVector",
            "ansatz": "RealAmplitudes",
            "n_qubits": 3,
            "reps": 1,
            "optimizer": "SPSA",
            "optimizer_params": optimizer_params
        },
        "metrics": metrics,
        "timing": timing
    }
    
    # Save to JSON
    results_file = results_dir / "quantum_evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # Log results
    logger.info("\n=== Best Quantum Circuit Performance ===")
    logger.info(f"\nDataset Info:")
    logger.info(f"Total Samples: {dataset_info['total_samples']}")
    logger.info(f"Training Samples: {dataset_info['training_samples']}")
    logger.info(f"Testing Samples: {dataset_info['testing_samples']}")
    
    logger.info("\nTest Set Metrics:")
    logger.info(f"Accuracy: {metrics['test']['accuracy']:.4f}")
    logger.info(f"Precision (weighted): {metrics['test']['precision']:.4f}")
    logger.info(f"Recall (weighted): {metrics['test']['recall']:.4f}")
    logger.info(f"F1 Score (weighted): {metrics['test']['f1']:.4f}")
    
    logger.info("\nTiming:")
    logger.info(f"Training Time: {timing['training_time']:.2f} seconds")
    logger.info(f"Prediction Time: {timing['prediction_time']:.2f} seconds")
    logger.info(f"Average Prediction Time per Sample: {timing['prediction_time_per_sample']*1000:.2f} ms")
    
    logger.info(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    evaluate_best_circuit()
