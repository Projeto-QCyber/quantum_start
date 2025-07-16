import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import base64
from io import BytesIO
from multiprocessing import Pool, cpu_count
from qsvmtester import QSVMTester, parse_qsvm_args
import time

def generate_confusion_matrix_image(y_test, y_pred, title):
    """Generate a confusion matrix plot and return as a base64 encoded string."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def run_single_experiment(config):
    """Function to run a single experiment configuration."""
    args = parse_qsvm_args()
    tester = QSVMTester()
    
    experiment_id = (
        f"{config['device']}_"
        f"paulis={config['feature_map']['paulis']}_"
        f"reps={config['feature_map']['reps']}_"
        f"ent={config['feature_map']['entanglement']}_"
        f"steps={config['num_steps']}"
    )
    
    params = {
        "n_features": args.n_features,
        "n_samples": 1000,
        "dataset_type": args.dataset_type,
        "feature_map_params": config["feature_map"],
        "num_steps": config["num_steps"],
        "device": config["device"],
        "experiment_id": experiment_id,
    }
    
    acc, f1, training_time, y_test, y_pred = tester.train_and_evaluate(**params)
    
    cm_title = f"Confusion Matrix for {experiment_id}"
    
    cm_image_b64 = None
    if y_test is not None and y_pred is not None:
        cm_image_b64 = generate_confusion_matrix_image(y_test, y_pred, cm_title)

    return {
        "Accuracy": f"{acc:.4f}",
        "F1 Score": f"{f1:.4f}",
        "Training Time (s)": f"{training_time:.2f}",
        "Device": config['device'],
        "Paulis": str(config['feature_map']['paulis']),
        "Reps": config['feature_map']['reps'],
        "Entanglement": config['feature_map']['entanglement'],
        "Num Steps": config['num_steps'],
        "Confusion Matrix": cm_image_b64
    }

def run_all_experiments_in_parallel():
    """Run all QSVM experiments in parallel and generate an HTML report."""
    
    # Define configurations for CPU
    cpu_configs = [
        {"feature_map": {"paulis": ["Z", "ZZ"], "reps": 3, "entanglement": "linear"}, "num_steps": 50, "device": "CPU"},
        {"feature_map": {"paulis": ["X", "Y", "Z"], "reps": 3, "entanglement": "linear"}, "num_steps": 50, "device": "CPU"},
    ]
    
    # Define configurations for GPU
    gpu_configs = [
        {"feature_map": {"paulis": ["Z", "ZZ"], "reps": 3, "entanglement": "linear"}, "num_steps": 50, "device": "GPU"},
        {"feature_map": {"paulis": ["X", "Y", "Z"], "reps": 3, "entanglement": "linear"}, "num_steps": 50, "device": "GPU"},
    ]

    all_configs = cpu_configs + gpu_configs
    
    # Use multiprocessing to run experiments in parallel
    num_processes = min(cpu_count(), len(all_configs))
    
    print(f"Starting {len(all_configs)} experiments in parallel using {num_processes} processes...")
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_experiment, all_configs)
        
    total_time = time.time() - start_time
    print(f"All experiments completed in {total_time:.2f} seconds.")

    # Create and display a summary table
    results_df = pd.DataFrame(results)
    
    # Generate HTML report
    html_content = "<html><head><title>QSVM Experiment Report</title>"
    html_content += "<style>"
    html_content += "body { font-family: Arial, sans-serif; margin: 20px; } "
    html_content += "h1, h2 { color: #333; } "
    html_content += "table { border-collapse: collapse; width: 100%; } "
    html_content += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } "
    html_content += "th { background-color: #f2f2f2; } "
    html_content += "tr:nth-child(even) { background-color: #f9f9f9; } "
    html_content += ".cm-image { max-width: 600px; height: auto; display: block; margin-top: 10px; } "
    html_content += "</style></head><body>"
    html_content += "<h1>QSVM Experiment Summary</h1>"
    
    # Results table
    results_table_df = results_df.drop(columns=['Confusion Matrix'])
    html_content += results_table_df.to_html(index=False, escape=False)
    
    # Confusion matrices
    html_content += "<h2>Confusion Matrices</h2>"
    for _, row in results_df.iterrows():
        if row['Confusion Matrix']:
            device = row['Device']
            paulis = row['Paulis']
            reps = row['Reps']
            entanglement = row['Entanglement']
            steps = row['Num Steps']
            
            html_content += f"<h3>Config: Device={device}, Paulis={paulis}, Reps={reps}, Entanglement='{entanglement}', Steps={steps}</h3>"
            html_content += f"<img src='data:image/png;base64,{row['Confusion Matrix']}' class='cm-image' alt='Confusion Matrix'>"
            html_content += "<hr>"

    html_content += "</body></html>"
    
    report_path = "qsvm_experiment_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
        
    print(f"\nHTML report saved to: {report_path}")

if __name__ == "__main__":
    run_all_experiments_in_parallel() 