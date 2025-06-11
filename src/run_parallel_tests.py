import multiprocessing as mp
from qiskit.providers.fake_provider import GenericBackendV2
from quantumtester import QuantumTester
import time
from pathlib import Path

def run_single_test(process_id: int):
    """Run a single instance of the quantum tester."""
    # Create unique study name and storage path for this process
    study_name = f"quantum_raw_optimization6"
    storage_path = f"quantum_raw_optimization6.db"
    
    # Initialize backend and tester
    backend = GenericBackendV2(num_qubits=8)
    tester = QuantumTester(
        backend=backend,
        study_name=study_name,
        storage_path=storage_path,
        n_trials=250  # Each instance will run 25 trials
    )
    
    # Run optimization
    print(f"Process {process_id}: Starting optimization...")
    results = tester.test_all_combinations(n_features=8)
    print(f"Process {process_id}: Completed")
    return results

def main():
    # Number of parallel processes to run
    n_processes = 50
    
    # Create and start processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_single_test, range(n_processes))
    
    print("All processes completed!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 