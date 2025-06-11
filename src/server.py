import streamlit as st
import json
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
from quantumtrainers import FeatureMapEnum, AnsatzEnum

st.set_page_config(page_title="ML Models Dashboard", layout="wide")
st.markdown("""
  <style>
    .st-emotion-cache-kgpedg {
      padding: 0rem 0rem 0rem 0rem;
    }
    .st-emotion-cache-kgpedg.e1dbuyne10 {
      height: 0px;
    }
    .element-container hr {
      height: 0px !important;
      border: 2px;
      border-top: 1px solid rgba(49, 51, 63, 0.2);
      margin: 0;
    }
</style>
  
""", unsafe_allow_html=True)

st.title("ML Models Dashboard", anchor=False)

def get_metric_value(metrics: Dict[str, Any], metric_name: str) -> float:
    """Extract metric value with proper fallback and type conversion."""
    if metric_name == 'accuracy':
        return float(metrics.get('accuracy', 0))
    
    weighted_key = f"{metric_name}_weighted"
    micro_key = f"{metric_name}_micro"
    
    # Try weighted first, then micro, then fallback to 0
    value = metrics.get(weighted_key, metrics.get(micro_key, 0))
    return float(value)

def format_value(value: float, is_best: bool) -> str:
    """Format a value with optional bold markdown."""
    formatted = f"{value:.3f}"
    return f"**{formatted}**" if is_best else formatted

def load_models() -> List[Dict[str, Any]]:
    """Load model data from JSON files."""
    models = []
    results_path = Path("results/data")
    if results_path.exists():
        for json_file in results_path.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    model_data = json.load(f)
                    model_data['file_path'] = str(json_file)
                    models.append(model_data)
            except Exception as e:
                st.error(f"Error loading {json_file}: {str(e)}")
    return models

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.resolve()

# Update model options to include quantum-specific configurations
def get_model_options():
    return {
        "available": {
            "Classical Models": {
                "Naive Bayes (NB)": "--nb",
                "Decision Tree (DT)": "--dt"
            },
            "Quantum Models": {
                "Variational Quantum Classifier (VQC)": "--vqc"
            }
        },
        "planned": {
            "Classical Models": [
                "Random Forest (RF)",
                "XGBoost (XGB)"
            ],
            "Quantum Models": [
                "Quantum Neural Network (QNN)",
                "Quantum Random Forest (QRF)"
            ]
        }
    }

# Add quantum-specific configuration options
def get_quantum_config_options():
    return {
        "feature_maps": [fm.name for fm in FeatureMapEnum],
        "ansatz_types": [a.name for a in AnsatzEnum],
        "optimizers": ["COBYLA", "ADAM", "SPSA", "GradientDescent"]
    }

def get_preprocessing_options():
    return {
        "scalers": {
            "available": {
                "StandardScaler (SS)": "standard",
                "MaxAbsScaler (MAS)": "maxabs"
            },
            "planned": [
                "RobustScaler (RS)",
                "MinMaxScaler (MMS)"
            ]
        },
        "dim_reduction": {
            "available": {
                "Classical PCA": "pca",
                "None": "none"
            },
            "planned": [
                "Quantum PCA (qPCA)",
                "Kernel PCA (kPCA)"
            ]
        }
    }

def get_configuration_options():
    return {
        "preprocessing": {
            "scaler": {
                "name": "Scaler Selection",
                "options": {
                    "available": {
                        "StandardScaler (SS)": "standard",
                        "MaxAbsScaler (MAS)": "maxabs"
                    },
                    "planned": ["RobustScaler (RS)", "MinMaxScaler (MMS)"]
                },
                "help": "Data scaling method",
                "optuna_space": ["standard", "maxabs"]
            },
            "dim_reduction": {
                "name": "Dimension Reduction",
                "options": {
                    "available": {
                        "Classical PCA": "pca",
                        "None": "none"
                    },
                    "planned": ["Quantum PCA (qPCA)", "Kernel PCA (kPCA)"]
                },
                "help": "Dimensionality reduction method",
                "optuna_space": ["pca", "none"]
            }
        },
        "quantum": {
            "feature_map": {
                "name": "Feature Map",
                "options": {
                    "available": {
                        str(fm.name): fm.name.lower() for fm in FeatureMapEnum
                    },
                    "planned": ["Custom Feature Map"]
                },
                "help": "Quantum feature mapping strategy",
                "optuna_space": [fm.name.lower() for fm in FeatureMapEnum]
            },
            "ansatz": {
                "name": "Ansatz",
                "options": {
                    "available": {
                        str(a.name): a.name.lower() for a in AnsatzEnum
                    },
                    "planned": ["Custom Ansatz"]
                },
                "help": "Quantum circuit architecture",
                "optuna_space": [a.name.lower() for a in AnsatzEnum]
            },
            "optimizer": {
                "name": "Optimizer",
                "options": {
                    "available": {
                        "COBYLA": "cobyla",
                        "ADAM": "adam",
                        "SPSA": "spsa",
                        "GradientDescent": "gradient_descent"
                    },
                    "planned": ["Custom Optimizer"]
                },
                "help": "Quantum circuit optimizer",
                "optuna_space": ["cobyla", "adam", "spsa", "gradient_descent"]
            }
        }
    }

def calculate_qubit_requirements(n_features: int, dim_reduction: str, n_components: Optional[int] = None) -> Dict[str, int]:
    """Calculate qubit requirements for quantum circuits.
    
    In quantum circuits:
    - Number of qubits directly determines input dimension
    - For n qubits, we can represent 2^n amplitudes
    - PCA dimension must match number of qubits exactly
    """
    result = {
        "original_features": n_features,
        "required_qubits": int(np.ceil(np.log2(n_features))),  # Minimum qubits needed
        "reduced_features": n_features
    }
    
    if dim_reduction.lower() == "pca":
        if n_components:
            # If PCA components specified, use that directly as number of qubits
            result["required_qubits"] = n_components
            result["reduced_features"] = n_components
        else:
            # Auto calculate based on log2 of features
            n_qubits = int(np.ceil(np.log2(n_features)))
            result["required_qubits"] = n_qubits
            result["reduced_features"] = n_qubits
    
    return result

def render_configuration_section(section_name: str, config: dict, auto_mode: bool):
    """Dynamically render configuration section with Optuna integration"""
    optuna_choices = {}
    manual_choices = {}
    
    for param, details in config.items():
        st.write(f"**{details['name']}**")
        
        # Add toggle for Optuna optimization per parameter
        use_optuna = st.checkbox("🎯 Use Optuna Optimization", key=f"optuna_{param}")
        optuna_choices[param] = use_optuna
        
        if use_optuna:
            st.info(f"Optuna will search through: {', '.join(details['optuna_space'])}")
        else:
            # Show available options
            options = list(details['options']['available'].keys())
            value = st.selectbox(
                "Select value",
                options=options,
                help=details['help']
            )
            manual_choices[param] = value
        
        # Show planned features as plain text
        if details['options']['planned']:
            st.caption("🔜 Planned features: " + ", ".join(details['options']['planned']))
        
        st.divider()

    return optuna_choices, manual_choices

def render_quantum_config(selected_model: str, config_options: dict):
    """Render quantum-specific configuration UI."""
    quantum_optuna, quantum_manual = render_configuration_section(
        "Quantum",
        config_options["quantum"],
        auto_mode=True
    )
    
    # Get feature map from manual selection or first option if using Optuna
    feature_map = quantum_manual.get('feature_map',
        list(config_options['quantum']['feature_map']['options']['available'].values())[0])
    
    # For VQC, get additional parameters
    ansatz = None
    optimizer = None
    if "VQC" in selected_model:
        ansatz = quantum_manual.get('ansatz',
            list(config_options['quantum']['ansatz']['options']['available'].values())[0])
        if not quantum_optuna.get('optimizer', False):
            optimizer = quantum_manual.get('optimizer',
                list(config_options['quantum']['optimizer']['options']['available'].values())[0])
    
    return {
        "optuna_config": quantum_optuna,
        "feature_map": feature_map,
        "ansatz": ansatz,
        "optimizer": optimizer
    }

def load_dataset_info():
    """Load basic dataset information including actual dataset sizes."""
    try:
        # Load micro dataset info
        df_micro = pd.read_csv("../datasets/UNSW_NB15_training-set.micro.csv")
        micro_samples = len(df_micro)
        
        # Calculate actual feature count (excluding label and attack_cat)
        exclude_cols = ['label', 'attack_cat']
        n_features = len([col for col in df_micro.columns if col not in exclude_cols])
        
        # Load full dataset info - get actual row counts
        df_train = pd.read_csv("../datasets/UNSW_NB15_training-set.csv")
        df_test = pd.read_csv("../datasets/UNSW_NB15_testing-set.csv")
        train_samples = len(df_train)
        test_samples = len(df_test)
        total_samples = train_samples + test_samples
        
        return {
            "n_features": n_features,
            "micro_samples": micro_samples,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "total_samples": total_samples
        }
    except Exception as e:
        st.warning(f"Could not load dataset info: {str(e)}")
        return {
            "n_features": 42,  # More realistic fallback for UNSW-NB15
            "micro_samples": 100,
            "train_samples": 175341,  # Actual size from training set
            "test_samples": 82332,    # Actual size from test set
            "total_samples": 257673   # Total samples available
        }

with st.sidebar:
    st.title("Models", anchor=False)
    
    with st.expander("🚀 Start New Run", expanded=True):
        # Model Selection Section
        st.subheader("Model Selection")
        model_options = get_model_options()
        
        # Available models
        selected_model = st.selectbox(
            "Select Model",
            options=[opt for category in model_options["available"].values() 
                    for opt in category.keys()],
            help="Choose a model to train"
        )
        
        # Show planned models as plain text
        for category, models in model_options["planned"].items():
            if models:
                st.caption(f"🔜 Coming soon in {category}: " + ", ".join(models))
        
        config_options = get_configuration_options()
        
        # Preprocessing Section
        st.subheader("Preprocessing Pipeline")
        preproc_optuna, preproc_manual = render_configuration_section(
            "Preprocessing", 
            config_options["preprocessing"],
            auto_mode=True
        )
        
        # Show data dimensionality impact
        if any(preproc_optuna.values()):
            optimized_params = [
                config_options['preprocessing'][k]['name']
                for k, v in preproc_optuna.items() if v
            ]
            st.info("🔍 Optuna will optimize: " + ", ".join(optimized_params))
        
        # Get dataset info for feature calculations
        dataset_info = load_dataset_info()
        n_features = dataset_info["n_features"]
        
        # PCA and Qubit Configuration
        if 'pca' in preproc_manual.get('dim_reduction', '').lower():
            st.write("**🔄 Quantum Dimensions Configuration**")
            
            # Control toggle
            qubit_controls = st.radio(
                "Control Mode",
                options=["Qubit-Controlled", "PCA-Controlled"],
                help="Choose which parameter controls the other:\n- Qubit-Controlled: Number of qubits determines PCA dimensions (2^n_qubits)\n- PCA-Controlled: PCA dimensions determine required qubits (log2(n_pca))",
                horizontal=True
            )
            
            # Calculate bounds
            min_qubits = max(1, int(np.ceil(np.log2(2))))  # Minimum 1 qubit (2 dimensions)
            max_qubits = max(min_qubits, int(np.ceil(np.log2(n_features))))  # log2 of original features
            min_pca = 2  # Minimum 2 dimensions (1 qubit)
            max_pca = 2 ** max_qubits  # Maximum PCA dimensions
            
            # Two columns for linked inputs
            col1, col2 = st.columns(2)
            
            if qubit_controls == "Qubit-Controlled":
                with col1:
                    n_qubits = st.number_input(
                        "Number of Qubits",
                        min_value=min_qubits,
                        max_value=max_qubits,
                        value=min(4, max_qubits),
                        help="Number of qubits in the quantum circuit"
                    )
                    st.caption(f"Circuit will use {n_qubits} qubits")
                
                with col2:
                    n_pca = 2 ** n_qubits
                    st.number_input(
                        "PCA Components",
                        min_value=min_pca,
                        max_value=max_pca,
                        value=n_pca,
                        disabled=True,
                        help="PCA dimensions automatically set to 2^(number of qubits)"
                    )
                    st.caption(f"Data reduced to {n_pca} dimensions")
            else:  # PCA-Controlled
                with col2:
                    n_pca = st.number_input(
                        "PCA Components",
                        min_value=min_pca,
                        max_value=max_pca,
                        value=min(16, max_pca),
                        help="Number of PCA dimensions (will be rounded up to next power of 2)"
                    )
                    # Round up to next power of 2
                    n_pca = 2 ** int(np.ceil(np.log2(n_pca)))
                    st.caption(f"Rounded to {n_pca} dimensions (next power of 2)")
                
                with col1:
                    n_qubits = int(np.log2(n_pca))
                    st.number_input(
                        "Number of Qubits",
                        min_value=min_qubits,
                        max_value=max_qubits,
                        value=n_qubits,
                        disabled=True,
                        help="Number of qubits automatically set to log2(PCA dimensions)"
                    )
                    st.caption(f"Circuit will use {n_qubits} qubits")
            
            # Display quantum circuit information
            st.info(f"""
                🔄 **Quantum Circuit Information:**
                - Number of qubits: {n_qubits}
                - Input features: {n_pca} (2^{n_qubits})
                - Quantum state space: {2**n_qubits} dimensions
                - Original features: {n_features}
                - Compression ratio: {n_features/n_pca:.2f}x
            """)
            
            # Show warning if compression is too aggressive
            if n_pca < n_features/4:
                st.warning(f"""
                    ⚠️ High compression ratio ({n_features/n_pca:.1f}x)!
                    Consider using more qubits to preserve more information.
                    Next power of 2 would use {n_qubits + 1} qubits for {2**(n_qubits + 1)} dimensions.
                """)
            
            st.caption(f"""
                Currently in {qubit_controls} mode:
                {'Number of qubits determines PCA dimensions (2^n_qubits)' if qubit_controls == 'Qubit-Controlled' else 'PCA dimensions determine required qubits (log2(n_pca))'}
            """)
            
            # Store for later use
            n_components = n_qubits  # We pass number of qubits to the backend
        
        # Quantum Configuration (if applicable)
        if "Quantum" in selected_model:
            st.subheader("Quantum Configuration")
            quantum_config = render_quantum_config(selected_model, config_options)
            
            # Calculate and display qubit requirements
            qubit_info = calculate_qubit_requirements(
                n_features=n_features,
                dim_reduction=preproc_manual.get('dim_reduction', 'none'),
                n_components=n_components
            )
            
            st.info(f"""
                📊 **Circuit Requirements:**
                - Input features: {qubit_info['original_features']}
                - Required qubits: {qubit_info['required_qubits']}
                {f"- After PCA: {qubit_info['reduced_features']}" if 'pca' in preproc_manual.get('dim_reduction', '').lower() else ''}
            """)

        # Dataset Configuration
        st.subheader("Dataset Configuration")
        
        # Show detailed dataset statistics
        st.info(f"""
            📊 **Available Data:**
            - Training set: {dataset_info['train_samples']:,} samples
            - Testing set: {dataset_info['test_samples']:,} samples
            - Total available: {dataset_info['total_samples']:,} samples
            - Features per sample: {dataset_info['n_features']}
            
            Micro dataset has {dataset_info['micro_samples']:,} samples for quick testing.
        """)
        
        use_micro = st.checkbox(
            "Use Micro Dataset",
            value=True,
            help=f"Use micro dataset ({dataset_info['micro_samples']:,} samples) for quick testing"
        )
        
        # Only show sample selection when micro dataset is not selected
        if not use_micro:
            st.write("**Sample Size Configuration**")
            
            # Calculate reasonable step size based on dataset size
            if dataset_info['total_samples'] > 100000:
                step_size = 5000
            elif dataset_info['total_samples'] > 10000:
                step_size = 1000
            else:
                step_size = 100
            
            n_samples = st.slider(
                "Number of Samples",
                min_value=dataset_info['micro_samples'],
                max_value=dataset_info['total_samples'],
                value=min(10000, dataset_info['total_samples']),
                step=step_size,
                help=f"Number of samples to use (max: {dataset_info['total_samples']:,})"
            )
            
            # Show detailed sample usage information
            st.info(f"""
                🔄 **Selected Sample Size:**
                - Using {n_samples:,} samples
                - {(n_samples/dataset_info['total_samples']*100):.1f}% of total dataset
                - Recommended: Use 10k-50k samples for development, full dataset for final training
            """)

        # Training Configuration
        st.subheader("Training Configuration")
        n_trials = st.slider(
            "Number of Trials",
            min_value=1,
            max_value=2500,
            value=10,
            help="Number of optimization trials"
        )

        # Start Training Button
        start_training = st.button("Start Training", type="primary")
            
        if start_training:
            try:
                project_root = get_project_root()
                venv_python = project_root / ".venv" / "Scripts" / "python"
                
                # Change working directory to project root
                os.chdir(project_root)
                
                # Use relative paths since we changed directory
                cmd = [str(venv_python), "main.py"]
                
                # Add model selection argument
                if selected_model != "All Models":
                    cmd.append(model_options[selected_model])
                
                # Add quantum configuration if applicable
                if "Quantum" in selected_model:
                    quantum_cfg = {
                        "feature_map": quantum_config["feature_map"],
                        "n_qubits": quantum_config["n_qubits"],
                        "shots": quantum_config["shots"],
                        "optimization_level": quantum_config["optimization_level"],
                        "feature_map_reps": 2  # Default value from paper
                    }
                    
                    # Add VQC-specific configuration
                    if "VQC" in selected_model:
                        quantum_cfg.update({
                            "ansatz": quantum_config["ansatz"],
                            "optimizer": quantum_config["optimizer"]
                        })
                    
                    cmd.extend(["--quantum-config", json.dumps(quantum_cfg)])
                
                # Add number of trials
                cmd.extend(["--trials", str(n_trials)])
                
                # Add dataset flags
                if use_micro:
                    cmd.append("--micro")
                else:
                    cmd.extend(["--samples", str(n_samples)])
                                
                # Create progress placeholder
                progress_container = st.empty()
                with progress_container.container():
                    st.subheader("Training Progress")
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    metrics_area = st.empty()
                    
                    # Remove old status file if exists
                    status_file = Path("results/status.json")
                    if status_file.exists():
                        status_file.unlink()
                    
                    # Launch training process
                    process = subprocess.Popen(cmd)
                    
                    last_status = {}
                    while process.poll() is None:
                        if status_file.exists():
                            try:
                                with open(status_file) as f:
                                    status = json.load(f)
                                    
                                    if status != last_status:
                                        # Update progress display
                                        progress_bar.progress(status["progress"])
                                        status_text.info(f"{status['model']}: {status['message']}")
                                        
                                        if status["stage"] == "optimizing":
                                            metrics_area.markdown(f"""
                                                **Trial Metrics:**
                                                - Current Score: {status['current_score']:.4f}
                                                - Best Score: {status['best_score']:.4f}
                                                - Time: {status['trial_time']:.2f}s
                                            """)
                                        elif status["stage"] == "completed":
                                            metrics_area.success(f"""
                                                ✨ **Training Complete!**
                                                - F1 Score: {status['metrics']['f1_weighted']:.4f}
                                                - Total Time: {status['training_time']:.1f}s
                                            """)
                                        elif status["stage"] == "error":
                                            metrics_area.error(f"Error: {status['message']}")
                                        
                                        last_status = status
                            except json.JSONDecodeError:
                                # File might be in the middle of being written
                                pass
                                
                        time.sleep(0.1)
                    
                    if process.returncode != 0:
                        status_text.error("Training process failed!")
                    else:
                        status_text.success("Training completed successfully!")
                    
                    # Final cleanup
                    if status_file.exists():
                        status_file.unlink()
                    
                    time.sleep(1)  # Give user time to see final status
                    progress_container.empty()
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error starting run: {str(e)}")
    
    st.divider()
    models = load_models()
    selected_models = []
    
    # Sort models by timestamp for consistent display
    models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    for model in models:
        col1, col2 = st.columns([4, 1])
        with col1:
            model_name = model.get('model_name', 'Unknown')
            timestamp = model.get('timestamp', 'No timestamp')
            if st.checkbox(f"{model_name}\n*{timestamp}*", key=model['file_path']):
                selected_models.append(model)
        with col2:
            if st.button("❌", key=f"delete_{model['file_path']}", help="Delete model"):
                try:
                    os.remove(model['file_path'])
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting model: {str(e)}")

if len(selected_models) > 0:
    # Prepare data with proper metric extraction
    comparison_data: List[Dict[str, Any]] = []
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    display_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for model in selected_models:
        metrics = model.get('metrics', {})
        model_name = model.get('model_name', 'Unknown')
        timestamp = model.get('timestamp', 'No timestamp')
        full_name = f"{model_name} ({timestamp})"
        
        row: Dict[str, Any] = {'Model': full_name}
        for metric, display_name in zip(metric_names, display_names):
            row[display_name] = get_metric_value(metrics, metric)
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Create a table with markdown formatting
    st.subheader("Metrics Comparison", anchor=False)
    
    # Headers
    cols = st.columns([0.5, 2] + [1] * len(display_names))
    cols[0].write("**#**")
    cols[1].write("**Model**")
    for i, col in enumerate(display_names):
        cols[i + 2].write(f"**{col}**")
    
    # Data rows with bold best values
    for idx, row in df.iterrows(): #type: ignore
        full_name = row['Model']
        cols = st.columns([0.5, 2] + [1] * len(display_names))
        
        # Count bests and format values
        model_bests = 0
        for i, col in enumerate(display_names):
            value = float(row[col])  # Ensure float type
            max_val = df[col].max()
            is_best = np.isclose(value, max_val, rtol=1e-05, atol=1e-08)
            if is_best:
                model_bests += 1
            
            # Format and display value
            formatted = format_value(value, is_best)
            cols[i + 2].write(formatted)
        
        # Display winning count and model name
        cols[0].markdown(f"**{model_bests}**")
        cols[1].write(full_name)
    
    # Create bar chart
    st.divider()
    
    # Prepare data for plotting
    fig = go.Figure()
    
    # Add bars for each model
    for _, row in df.iterrows(): #type: ignore
        model_name = row['Model']
        values = [row[metric] for metric in display_names]
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=display_names,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
        ))
    
    # Update layout
    fig.update_layout(
        title="Metrics Comparison Chart",
        barmode='group',
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            range=[0, max(df[display_names].values.flatten()) * 1.1]  # Add 10% padding
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model details in a more compact form
    st.divider()
    st.subheader("Model Details", anchor=False)
    for model in selected_models:
        model_name = model.get('model_name', 'Unknown')
        timestamp = model.get('timestamp', 'No timestamp')
        with st.expander(f"📊 {model_name} ({timestamp})"):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.write("**Parameters:**")
                st.json(model.get('parameters', {}))
            
            with col2:
                st.write("**Training Info:**")
                st.write(f"- Dataset: {model.get('dataset_info', 'N/A')}")
                st.write(f"- Training Time: {model.get('training_time', 'N/A'):.2f}s")
                st.write(f"- Number of Trials: {model.get('n_trials', 'N/A')}")
                
                # Display quantum-specific info if available
                quantum_config = model.get('quantum_config', {})
                if quantum_config:
                    st.write("**Quantum Configuration:**")
                    st.write(f"- Feature Map: {quantum_config.get('feature_map', 'N/A')}")
                    st.write(f"- Feature Map Reps: {quantum_config.get('feature_map_reps', 'N/A')}")
                    st.write(f"- Qubits: {quantum_config.get('n_qubits', 'N/A')}")
                    st.write(f"- Shots: {quantum_config.get('shots', 'N/A')}")
                    st.write(f"- Optimization Level: {quantum_config.get('optimization_level', 'N/A')}")
                    
                    if 'ansatz' in quantum_config:
                        st.write(f"- Ansatz: {quantum_config.get('ansatz', 'N/A')}")
                        st.write(f"- Optimizer: {quantum_config.get('optimizer', 'N/A')}")
            
            with col3:
                if "Quantum" in model_name:
                    st.write("**Best Trial Configuration:**")
                    best_trial = model.get('best_trial', {})
                    if best_trial:
                        st.write(f"- Score: {best_trial.get('value', 'N/A'):.4f}")
                        st.write("- Parameters:")
                        st.json(best_trial.get('params', {}))
                else:
                    st.write("**Model Metrics:**")
                    metrics = model.get('metrics', {})
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            st.write(f"- {metric}: {value:.4f}")
else:
    st.info("Select models from the sidebar to compare them.")
