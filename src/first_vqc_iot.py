import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_EdgeIIoT_dataset, IDSPreprocessor
from quantumtrainers import VQCTrainer
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_machine_learning.optimizers import SPSA
from logger import get_logger


class SimpleVQCTrainer(VQCTrainer):
    """VQCTrainer variant with fixed, small-iteration optimizer."""

    def _create_optimizer(self, trial):  # type: ignore[override]
        # Nenhuma otimização de hiperparâmetros do otimizador é feita agora.
        return SPSA(maxiter=20)


def main() -> None:
    logger = get_logger("first_vqc_iot")

    candidates = [
        "datasets/ML-EdgeIIoT-dataset.csv",
        "../datasets/ML-EdgeIIoT-dataset.csv",
    ]
    csv_path = None
    for p in candidates:
        if Path(p).exists():
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find ML-EdgeIIoT-dataset.csv in {candidates}"
        )

    # Carrega o dataset (apenas features numéricas)
    X_full, y_full = load_EdgeIIoT_dataset(csv_path)
    X_num = X_full.select_dtypes(include=[np.number])

    # Build a balanced (100 benign, 100 attack) subset
    df_full = X_num.copy()
    df_full["Attack_label"] = y_full

    benign_df = df_full[df_full["Attack_label"] == 0].sample(n=100, random_state=42)
    attack_df = df_full[df_full["Attack_label"] != 0].sample(n=100, random_state=42)
    df = (
        pd.concat([benign_df, attack_df])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Convert series to ndarray explicitly for type checking.
    y = np.asarray(df["Attack_label"], dtype=int)
    X_df = df.drop(columns=["Attack_label"]).iloc[:, :4]  # first 4 numeric features

    # Basic scaling (optional for hello-world)
    preprocessor = IDSPreprocessor(
        scaler_type="standard",
        dim_reduction_type="none",
    )

    # Split data (stratified to keep class balance)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, stratify=y, random_state=42
    )

    # Explicit ignores for pandas typing quirks.
    X_train = preprocessor.fit_transform(X_train_df)  # type: ignore[arg-type]
    X_test = preprocessor.transform(X_test_df)  # type: ignore[arg-type]

    # Setup tiny VQC trainer – only a single trial and ~10 optimiser iterations
    backend = GenericBackendV2(num_qubits=4)
    trainer = SimpleVQCTrainer(
        backend=backend,
        study_name="first_vqc_iot_demo",
        n_trials=1,  # single trial: pick first working configuration
    )

    trainer.train_model(
        X_train,
        X_test,
        np.asarray(y_train),  # ensure ndarray for type checker
        np.asarray(y_test),
        model_name="FirstVQC_IoT",
        preprocessor=None,  # already scaled
        feature_map="PauliFeatureMap",
        ansatz_type="EfficientSU2",
        patience=3,
        min_delta=1e-3,
    )

    logger.info(
        "First VQC IoT finalizada! Métricas: %s",
        trainer.metrics["FirstVQC_IoT"],
    )


if __name__ == "__main__":
    main() 