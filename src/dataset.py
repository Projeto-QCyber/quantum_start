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
    fold_idx: int = 0  # Which fold to use
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
    
    # Define paths relative to current directory
    if micro:
        train_path = Path("../datasets/UNSW_NB15_training-set.micro.csv")
        test_path = Path("../datasets/UNSW_NB15_testing-set.micro.csv")
    else:
        train_path = Path("../datasets/UNSW_NB15_training-set.csv")
        test_path = Path("../datasets/UNSW_NB15_testing-set.csv")
    
    # Check if files exist
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_path} or {test_path}. "
            f"Please ensure the UNSW-NB15 {'micro' if micro else 'full'} dataset files are present."
        )
    
    # Load datasets with error handling
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")
    
    # Validate required columns exist
    required_columns = ['attack_cat', 'label']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    
    # Combine datasets for processing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Reduce dataset size while maintaining class distribution if max_samples specified
    if max_samples and len(combined_df) > max_samples:
        logger.info(f"Reducing dataset from {len(combined_df)} to {max_samples} samples")
        combined_df = combined_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(max_samples * len(x) / len(combined_df))), random_state=42)
        )
        if len(combined_df) > max_samples:  # Handle rounding up
            combined_df = combined_df.sample(n=max_samples, random_state=42)
    
    logger.info(f"Dataset loaded: {len(combined_df)} total samples")
    
    # Separate features and target
    X = combined_df.drop(['attack_cat', 'label'], axis=1)
    y = combined_df['label']
    
    # Handle categorical columns
    categorical_columns = ['proto', 'service', 'state']
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get the specified fold
    train_indices = []
    test_indices = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_indices = train_idx
            test_indices = test_idx
            break
    
    # Split the data using the fold indices
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    # Create default preprocessor if none provided
    if preprocessor is None:
        if quantum_mode and n_components:
            logger.info(f"Creating quantum-ready preprocessor with {n_components} qubits")
            preprocessor = IDSPreprocessor(
                scaler_type='standard',
                log_transform_features=[
                    'dur', 'sbytes', 'dbytes', 'spkts', 'dpkts',
                    'sload', 'dload', 'sinpkt', 'dinpkt'
                ],
                feature_selection='variance',
                dim_reduction_type='pca',
                n_components=n_components,
                quantum_mode=True
            )
        else:
            preprocessor = IDSPreprocessor(
                scaler_type='standard',
                log_transform_features=[
                    'dur', 'sbytes', 'dbytes', 'spkts', 'dpkts',
                    'sload', 'dload', 'sinpkt', 'dinpkt'
                ],
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