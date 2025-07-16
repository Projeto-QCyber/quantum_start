import argparse
import time
import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from dataset import generate_dataset
from logger import get_logger

def run_benchmark(device: str):
    """
    Runs a single QSVM training pass on a specified device and measures the time.
    """
    logger = get_logger(__name__)
    logger.info(f"Starting benchmark for device: {device}")

    # 1. Generate a dataset
    logger.info("Generating dataset...")
    X_train, _, y_train, _ = generate_dataset(micro=False, max_samples=2000, n_components=8)
    logger.info(f"Dataset generated. X_train shape: {X_train.shape}")

    # 2. Create a feature map
    logger.info("Creating feature map...")
    n_features = X_train.shape[1]
    feature_map = PauliFeatureMap(feature_dimension=n_features, reps=2, entanglement='linear')
    transpiled_feature_map = transpile(feature_map, backend=AerSimulator())


    # 3. Create the QSVM with the specified device
    logger.info(f"Creating QSVM with device='{device}'...")
    backend_options = {
        "method": "statevector",
        "device": device
    }
    sampler = AerSamplerV2(options={"backend_options": backend_options})
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=transpiled_feature_map
    )
    qsvm = PegasosQSVC(quantum_kernel=qkernel, C=1.0, num_steps=100)

    # 4. Run the training and time it
    logger.info("Starting training...")
    start_time = time.time()
    try:
        qsvm.fit(X_train, y_train)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- BENCHMARK RESULT ---")
        logger.info(f"Device: {device}")
        logger.info(f"Training time: {duration:.4f} seconds")
        logger.info(f"------------------------")
    except Exception as e:
        logger.error(f"An error occurred during training on {device}: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single QSVM training benchmark.")
    parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], required=True, help='The device to run the benchmark on.')
    args = parser.parse_args()

    # It's better to run from inside the src directory
    # to ensure relative paths to datasets work correctly.
    run_benchmark(args.device) 