from typing import List, Dict, Any, Optional, Union, Mapping
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from logger import get_logger
import os
import sys

class IDSPreprocessor:
    """Preprocessor for Intrusion Detection System data with quantum-specific handling"""
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        log_transform_features: Optional[List[str]] = None,
        dim_reduction_type: str = 'none',
        n_components: Optional[int] = None,
        force_power_of_two: bool = False,
        quantum_mode: bool = False,
        feature_selection: str = 'none',
        variance_threshold: float = 0.01
    ):
        """Initialize preprocessor with configuration.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'maxabs', or 'robust')
            log_transform_features: List of features to apply log transformation
            dim_reduction_type: Type of dimensionality reduction ('none' or 'pca')
            n_components: Number of components for PCA (if used)
            force_power_of_two: If True, ensures output dimension is power of 2
            quantum_mode: If True, ensures dimensions are compatible with quantum circuits
            feature_selection: Type of feature selection ('none' or 'variance')
            variance_threshold: Threshold for variance-based feature selection
        """
        self.logger = get_logger(__name__)
        self.scaler_type = scaler_type.lower()
        self.log_transform_features = log_transform_features or []
        self.dim_reduction_type = dim_reduction_type.lower()
        self.n_components = n_components
        self.force_power_of_two = force_power_of_two or quantum_mode
        self.quantum_mode = quantum_mode
        self.feature_selection = feature_selection.lower()
        self.variance_threshold = variance_threshold
        
        # Initialize components
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Initialize feature selector
        self.selector = None
        if self.feature_selection == 'variance':
            self.selector = VarianceThreshold(threshold=self.variance_threshold)
        elif self.feature_selection != 'none':
            raise ValueError(f"Unknown feature selection type: {feature_selection}")
        
        self.pca = None
        if self.dim_reduction_type == 'pca':
            if not n_components:
                raise ValueError("n_components must be specified when using PCA")
            
            # For quantum circuits, n_components is the number of qubits
            if self.quantum_mode:
                self.n_qubits = n_components
                self.n_components = 2 ** n_components  # Convert to feature dimension
                self.logger.info(f"Quantum mode: using {self.n_qubits} qubits ({self.n_components} features)")
            else:
                # Adjust n_components to next power of 2 if needed
                if self.force_power_of_two:
                    self.n_components = self._next_power_of_two(n_components)
                    if self.n_components != n_components:
                        self.logger.warning(
                            f"Adjusted n_components from {n_components} to {self.n_components} "
                            "to ensure power of 2"
                        )
                else:
                    self.n_components = n_components
            
            self.pca = PCA(n_components=self.n_components)
    
    def _next_power_of_two(self, n: int) -> int:
        """Return the next power of 2 greater than or equal to n."""
        return 2 ** int(np.ceil(np.log2(n)))
    
    def _validate_dimensions(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Validate input dimensions and warn about potential issues."""
        n_features = X.shape[1]
        
        if self.dim_reduction_type == 'pca' and self.n_components is not None:
            if self.quantum_mode:
                # For quantum mode, we need exactly 2^n_qubits features
                target_features = 2 ** self.n_qubits
                if self.n_components != target_features:
                    self.logger.warning(
                        f"Adjusting PCA components from {self.n_components} to {target_features} "
                        f"to match quantum requirements ({self.n_qubits} qubits)"
                    )
                    self.n_components = target_features
                    self.pca = PCA(n_components=self.n_components)
            elif self.n_components > n_features:
                # For classical mode, handle too many components
                self.logger.warning(
                    f"n_components ({self.n_components}) is greater than number of features "
                    f"({n_features}). Will use {n_features} components instead."
                )
                self.n_components = n_features
                if self.force_power_of_two:
                    self.n_components = self._next_power_of_two(n_features)
                self.pca = PCA(n_components=self.n_components)
    
    def get_scaler_params(self) -> Dict[str, Any]:
        """Get scaler parameters."""
        params = self.scaler.get_params()
        if hasattr(self.scaler, 'scale_'):
            scale = self.scaler.scale_.tolist() if self.scaler.scale_ is not None else None
            params['scale_'] = scale
            
            # Only StandardScaler has mean_
            if isinstance(self.scaler, StandardScaler) and hasattr(self.scaler, 'mean_'):
                mean = self.scaler.mean_.tolist() if self.scaler.mean_ is not None else None
                params['mean_'] = mean
                
        return params
    
    def get_dim_reduction_params(self) -> Dict[str, str]:
        """Get dimensionality reduction parameters."""
        params: Dict[str, str] = {
            "type": str(self.dim_reduction_type),
            "feature_selection": str(self.feature_selection)
        }
        
        if self.selector is not None and hasattr(self.selector, 'get_support'):
            params['selected_features'] = str(int(np.sum(self.selector.get_support())))
            params['variance_threshold'] = str(self.variance_threshold)
        
        if self.pca is None:
            return params
            
        params.update({
            "n_components": str(self.n_components),
            "force_power_of_two": str(self.force_power_of_two),
            "quantum_mode": str(self.quantum_mode)
        })
        
        if self.quantum_mode:
            params["n_qubits"] = str(self.n_qubits)
        
        if hasattr(self.pca, 'explained_variance_ratio_'):
            params.update({
                'explained_variance_ratio': str(self.pca.explained_variance_ratio_.tolist()),
                'n_components_': str(self.pca.n_components_),
                'total_variance_explained': str(float(np.sum(self.pca.explained_variance_ratio_)))
            })
        return params
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform the data."""
        self._validate_dimensions(X)
        
        # ------------------------------------------------------------------
        # Pre-clean: drop non-numeric columns (e.g. strings such as IPs) that
        # would otherwise break the downstream scalers. The user is warned
        # once, but the operation is silent when the DataFrame is already
        # purely numeric. This makes the preprocessor plug-and-play with the
        # Edge-IIoT dataset whose raw CSV contains several textual columns.
        # ------------------------------------------------------------------
        non_num_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_num_cols:
            self.logger.warning(
                "Dropping %d non-numeric feature(s): %s", len(non_num_cols), non_num_cols
            )
            X = X.drop(columns=non_num_cols)
        
        # Apply log transformation if specified
        for feature in self.log_transform_features:
            if feature in X.columns:
                # Add small constant to handle zeros
                X[feature] = np.log1p(X[feature])
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply feature selection if specified
        if self.selector is not None:
            X_selected = self.selector.fit_transform(X_scaled)
            self.logger.info(
                f"Selected {X_selected.shape[1]} features out of {X_scaled.shape[1]} "
                f"using variance threshold {self.variance_threshold}"
            )
            X_scaled = X_selected
        
        # Apply dimensionality reduction if specified
        if self.pca is not None:
            X_reduced = self.pca.fit_transform(X_scaled)
            self.logger.info(
                f"PCA reduced features from {X_scaled.shape[1]} to {X_reduced.shape[1]} dimensions. "
                f"Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.2%}"
            )
            if self.quantum_mode:
                self.logger.info(f"Quantum-ready output: {self.n_qubits} qubits, {X_reduced.shape[1]} features")
            return np.asarray(X_reduced)
        return np.asarray(X_scaled)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data using fitted parameters."""
        # ------------------------------------------------------------------
        # Ensure we perform the same non-numeric column drop as in fit_transform
        # and align the feature set to those seen during fit using scaler.feature_names_in_.
        # Extra columns are discarded and missing ones are filled with zeros (or left-zero after scaling).
        # ------------------------------------------------------------------

        # Drop non-numeric columns consistently
        X_transformed = X.copy()
        non_num_cols = X_transformed.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_num_cols:
            # Do not warn repeatedly during batch processing – only debug level
            self.logger.debug(
                "Dropping %d non-numeric feature(s) at transform time: %s",
                len(non_num_cols), non_num_cols,
            )
            X_transformed = X_transformed.drop(columns=non_num_cols, errors="ignore")

        # Align feature columns to those used during fit
        if hasattr(self.scaler, "feature_names_in_"):
            fit_cols = list(self.scaler.feature_names_in_)
            X_transformed = X_transformed.reindex(columns=fit_cols, fill_value=0)
        
        # Apply log transformation if specified
        for feature in self.log_transform_features:
            if feature in X_transformed.columns:
                X_transformed[feature] = np.log1p(X_transformed[feature])
        
        # Scale the features
        X_scaled = self.scaler.transform(X_transformed)
        
        # Apply feature selection if specified
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        
        # Apply dimensionality reduction if specified
        if self.pca is not None:
            return np.asarray(self.pca.transform(X_scaled))
        return np.asarray(X_scaled)

# -----------------------------------------------------------------------------
# Utility helpers specific to the Edge-IIoT dataset
# -----------------------------------------------------------------------------

def load_EdgeIIoT_dataset(
    csv_path: str,
    label_col: str = "Attack_label",
    drop_cols: Optional[List[str]] = None,
):
    """Load the Edge-IIoT CSV and return (X, y) while printing basic stats.

    The function purposely keeps things lightweight and produces simple
    descriptive statistics that are useful before any heavy preprocessing.
    """

    logger = get_logger(__name__)

    if drop_cols is None:
        drop_cols = ["Attack_type"]  # category strings not used for modelling

    logger.info("Loading Edge-IIoT dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {csv_path}.")

    # Basic stats
    logger.info("Dataset shape : %s rows × %s columns", *df.shape)
    label_counts = df[label_col].value_counts().to_dict()
    logger.info("Label distribution (benign=0 vs. attack): %s", label_counts)

    # Numeric summary of the numeric subset
    numeric_summary = df.select_dtypes(include=[np.number]).describe().T
    logger.info("Numeric feature summary:\n%s", numeric_summary[['mean', 'std', 'min', 'max']])

    # Separate features / label
    X = df.drop(columns=[label_col] + drop_cols)
    y = df[label_col].values

    return X, y

# -----------------------------------------------------------------------------
# Quick demonstrator – hello-world style
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Default expected locations (relative to project root and this script)
    _candidate_paths = [
        "datasets/ML-EdgeIIoT-dataset.csv",  # most common when run from repo root
        "../datasets/ML-EdgeIIoT-dataset.csv",  # fallback when run from src/
    ]

    CSV_PATH = None
    for _p in _candidate_paths:
        if os.path.exists(_p):
            CSV_PATH = _p
            break

    if CSV_PATH is None:
        print("Could not locate ML-EdgeIIoT-dataset.csv. Checked paths:\n", _candidate_paths)
        sys.exit(1)

    try:
        X_raw, y = load_EdgeIIoT_dataset(CSV_PATH)

        preprocessor = IDSPreprocessor(
            scaler_type="standard",
            dim_reduction_type="none",
        )

        X_ready = preprocessor.fit_transform(X_raw)

        print("Hello, world! ✨  Preprocessing successful.")
        print(f"Processed feature matrix shape: {X_ready.shape}")
        print(f"Benign vs. attack counts         : {np.bincount(y.astype(int))}")

    except FileNotFoundError:
        print(
            "Edge-IIoT CSV not found at", CSV_PATH,
            "– please adjust the path or download the dataset first."
        )