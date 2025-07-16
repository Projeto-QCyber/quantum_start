"""Minimal hello-world demo using existing pipeline components.

This script intentionally keeps runtime VERY small:
1. Reads only the first 50 rows of the Edge-IIoT dataset.
2. Runs basic preprocessing (numeric scaling only).
3. Trains a single-trial Variational Quantum Classifier (VQC) if Qiskit is
   available; otherwise it falls back to the Naive Bayes trainer.
4. Prints the achieved accuracy – that's all!

Run:
    python src/hello_world.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import load_EdgeIIoT_dataset, IDSPreprocessor

# Try importing quantum components; if unavailable default to classical NB
USE_QUANTUM = True
try:
    from qiskit.providers.fake_provider import GenericBackendV2
    from quantumtrainers import VQCTrainer
except Exception:
    USE_QUANTUM = False
    from sklearn.naive_bayes import GaussianNB  # type: ignore


def _locate_dataset() -> str:
    """Return the first existing path for the Edge-IIoT CSV or raise."""
    candidate_paths = [
        "datasets/ML-EdgeIIoT-dataset.csv",
        "../datasets/ML-EdgeIIoT-dataset.csv",
    ]
    for p in candidate_paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "ML-EdgeIIoT-dataset.csv not found. Place it under ./datasets or "
        "adjust _locate_dataset()."
    )


def _prepare_data(max_rows: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a slice of the dataset and return train/test numpy arrays."""
    X_df, y = load_EdgeIIoT_dataset(_locate_dataset())
    X_df = X_df.iloc[:max_rows]
    y = y[:max_rows]

    pre = IDSPreprocessor(scaler_type="standard", dim_reduction_type="none")
    X_np = pre.fit_transform(X_df)

    return train_test_split(X_np, y, test_size=0.3, random_state=42, stratify=y)


def main() -> None:
    X_train, X_test, y_train, y_test = _prepare_data()

    if USE_QUANTUM:
        backend = GenericBackendV2(num_qubits=min(4, X_train.shape[1]))
        trainer = VQCTrainer(backend=backend, study_name="hello_vqc", n_trials=1)
        trainer.train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model_name="hello_vqc",
            feature_map="PauliFeatureMap",
            ansatz_type="EfficientSU2",
        )
        acc = trainer.metrics["hello_vqc"]["accuracy"]
        print(f"Hello, quantum world! Accuracy: {acc:.3f}")
    else:
        # Classic hello-world using plain scikit-learn GaussianNB
        from sklearn.naive_bayes import GaussianNB

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"Hello, classical world! GaussianNB accuracy: {acc:.3f}")


if __name__ == "__main__":
    main() 