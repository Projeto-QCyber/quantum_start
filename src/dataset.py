from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from preprocessing import IDSPreprocessor
from logger import get_logger
import json
from sklearn.model_selection import StratifiedKFold

def save_dataset_info(info: Dict[str, Any], path: str = "../datasets/dataset_info.json") -> None:
    """Save dataset information to a JSON file."""
    with open(path, 'w') as f:
        json.dump(info, f, indent=4)

def load_dataset_info(path: str = "../datasets/dataset_info.json") -> Dict[str, Any]:
    """Load dataset information from JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def generate_dataset(
    micro: bool = True,
    preprocessor: Optional[IDSPreprocessor] = None,
    n_components: Optional[int] = None,
    quantum_mode: bool = False,
    max_samples: Optional[int] = None,
    n_splits: int = 5,  # Number of folds
    fold_idx: int = 0,  # Which fold to use
    dataset_type: str = "EdgeIIoT"  # "UNSW" or "EdgeIIoT"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare UNSW-NB15 dataset with k-fold splitting
    
    Args:
        micro: If True, use micro dataset, otherwise use full dataset
        preprocessor: Optional preprocessor instance to use
        n_components: Number of components for PCA (if used)
        quantum_mode: If True, n_components specifies number of qubits instead of features
        max_samples: Optional maximum number of total samples to use (will be split into train/test)
        n_splits: Number of folds for cross-validation
        fold_idx: Which fold to use (0 to n_splits-1)
    """
    logger = get_logger(__name__)

    # Define dataset-specific properties
    dataset_configs = {
        "unsw": {
            "train_micro_path": "../datasets/UNSW_NB15_training-set.micro.csv",
            "test_micro_path": "../datasets/UNSW_NB15_testing-set.micro.csv",
            "train_full_path": "../datasets/UNSW_NB15_training-set.csv",
            "test_full_path": "../datasets/UNSW_NB15_testing-set.csv",
            "label_col": "label",
            "attack_cat_col": "attack_cat",
            "categorical_features": ['proto', 'service', 'state']
        },
        "edgeiiot": {
            "train_micro_path": "../datasets/ML-EdgeIIoT-dataset-training.csv",
            "test_micro_path": "../datasets/ML-EdgeIIoT-dataset-testing.csv",
            "train_full_path": "../datasets/ML-EdgeIIoT-dataset-training.csv",
            "test_full_path": "../datasets/ML-EdgeIIoT-dataset-testing.csv",
            "label_col": "Attack_label",
            "attack_cat_col": "Attack_type",
            "categorical_features": []
        }
    }
    
    # Get config for selected dataset
    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower not in dataset_configs:
        raise ValueError(f"Invalid dataset_type '{dataset_type}'. Must be one of {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_type_lower]
    
    # Determine file paths
    if micro:
        train_path = Path(config["train_micro_path"])
        test_path = Path(config["test_micro_path"])
    else:
        if config["train_full_path"] is None:
            logger.warning(f"Full dataset path not specified for '{dataset_type}'. Falling back to micro dataset.")
            train_path = Path(config["train_micro_path"])
            test_path = Path(config["test_micro_path"])
        else:
            train_path = Path(config["train_full_path"])
            test_path = Path(config["test_full_path"])

    # Check if files exist
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_path} or {test_path}. "
            f"Please ensure the {'micro' if micro else 'full'} dataset files for '{dataset_type}' are present."
        )
    
    # Load datasets
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        raise RuntimeError(f"Error loading CSV files: {e}")

    # Harmonize column names to internal standard ('label', 'attack_cat')
    rename_map = {}
    if config["label_col"] in train_df.columns and config["label_col"] != "label":
        rename_map[config["label_col"]] = "label"
    if config["attack_cat_col"] in train_df.columns and config["attack_cat_col"] != "attack_cat":
        rename_map[config["attack_cat_col"]] = "attack_cat"
        
    if rename_map:
        train_df = train_df.rename(columns=rename_map)
        test_df = test_df.rename(columns=rename_map)

    # Validate that the standardized 'label' column exists
    if "label" not in train_df.columns:
        raise ValueError(f"Standardized 'label' column not found in the dataset. Original expected: '{config['label_col']}'")

    # Combine datasets for processing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Reduce dataset size while maintaining class distribution
    if max_samples and len(combined_df) > max_samples:
        logger.info(f"Reducing dataset from {len(combined_df)} to {max_samples} samples")
        # Ensure 'label' column exists before grouping
        if 'label' in combined_df.columns:
            combined_df = combined_df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(max_samples * len(x) / len(combined_df))), random_state=42)
            )
            if len(combined_df) > max_samples:
                combined_df = combined_df.sample(n=max_samples, random_state=42)
        else:
            logger.warning("No 'label' column found for stratified sampling, using random sampling instead.")
            combined_df = combined_df.sample(n=max_samples, random_state=42)

    logger.info(f"Dataset loaded: {len(combined_df)} total samples")
    
    # Separate features and target
    drop_cols = ['label']
    if 'attack_cat' in combined_df.columns:
        drop_cols.append('attack_cat')
    
    X = combined_df.drop(columns=drop_cols)
    y = combined_df['label']
    
    # Handle categorical columns dynamically
    categorical_columns = [col for col in config["categorical_features"] if col in X.columns]
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get the specified fold
    train_indices = np.array([], dtype=np.intp)
    test_indices = np.array([], dtype=np.intp)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_indices, test_indices = train_idx, test_idx
            break
    
    # Split the data using the fold indices
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    # Create default preprocessor if none provided
    if preprocessor is None:
        # Define log transform features common to both datasets if applicable
        log_transform_features = [
            'dur', 'sbytes', 'dbytes', 'spkts', 'dpkts',
            'sload', 'dload', 'sinpkt', 'dinpkt'
        ]
        # Filter features that are actually in the dataset
        log_transform_features = [f for f in log_transform_features if f in X_train.columns]

        if quantum_mode and n_components:
            logger.info(f"Creating quantum-ready preprocessor with {n_components} qubits")
            preprocessor = IDSPreprocessor(
                scaler_type='standard',
                log_transform_features=log_transform_features,
                feature_selection='variance',
                dim_reduction_type='pca',
                n_components=n_components,
                quantum_mode=True
            )
        else:
            preprocessor = IDSPreprocessor(
                scaler_type='standard',
                log_transform_features=log_transform_features,
                feature_selection='variance',
                dim_reduction_type='pca' if n_components else 'none',
                n_components=n_components
            )
    
    # Fit preprocessor on training data only and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert target to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # Save dataset information
    dataset_info = {
        "n_features": X_train_processed.shape[1],
        "n_samples": len(X_train_processed),
        "n_classes": len(np.unique(y_train)),
        "original_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "preprocessing": {
            "scaler": preprocessor.scaler_type,
            "dim_reduction": preprocessor.dim_reduction_type,
            "n_components": preprocessor.n_components if preprocessor.pca else None,
            "quantum_mode": getattr(preprocessor, 'quantum_mode', False),
            "n_qubits": getattr(preprocessor, 'n_qubits', None) if preprocessor.pca else None
        }
    }
    save_dataset_info(dataset_info)
    
    logger.info(f"Final dataset shapes - X_train: {X_train_processed.shape}, X_test: {X_test_processed.shape}")
    if quantum_mode:
        logger.info(f"Quantum-ready features: {dataset_info['n_features']} (2^{dataset_info['preprocessing']['n_qubits']})")
    logger.info(f"Class distribution - Train: {np.unique(y_train, return_counts=True)}")
    logger.info(f"Class distribution - Test: {np.unique(y_test, return_counts=True)}")
    
    return X_train_processed, X_test_processed, y_train, y_test 