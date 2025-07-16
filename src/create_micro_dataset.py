import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------------------------------------------------------
# Updated script to create train/test splits of the full Edge-IIoT dataset
# (ML-EdgeIIoT-dataset.csv). The script performs a stratified 70/30 train/test
# split and stores the resulting files next to the original CSV.
# -----------------------------------------------------------------------------

def create_train_test_split_from_full_dataset(
    data_file: str = "ML-EdgeIIoT-dataset.csv",
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Create training and testing datasets from the full Edge-IIoT CSV file.

    Parameters
    ----------
    data_file : str
        Path to the full CSV dataset containing an `Attack_label` column where
        0 = benign traffic and any other value = attack.
    test_size : float
        Proportion of the dataset that will be reserved for the test
        split. The remainder is used for training.
    random_state : int
        Seed to ensure reproducible shuffling/sampling.
    """

    # Construct the absolute path to the data file
    datasets_dir = Path(__file__).parent.parent / "datasets"
    data_path = datasets_dir / data_file

    print("\n[1/4] Reading full dataset …")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please ensure the file '{data_file}' is in the 'datasets' directory."
        )
    full_df = pd.read_csv(data_path)

    if "Attack_label" not in full_df.columns:
        raise KeyError(
            "Column 'Attack_label' not found. Please verify that the provided "
            "CSV is the original ML-EdgeIIoT dataset."
        )

    # Identify benign vs. attack samples using Attack_label
    print("[2/4] Analysing class distribution …")
    normal_df = full_df[full_df["Attack_label"] == 0]
    attack_df = full_df[full_df["Attack_label"] != 0]

    total = len(full_df)
    normal_ratio = len(normal_df) / total

    print(f"   Total samples         : {total:,}")
    print(f"   Benign (label==0)     : {len(normal_df):,} ({normal_ratio:.2%})")
    print(f"   Attack (label!=0)     : {len(attack_df):,} ({1-normal_ratio:.2%})")

    # Perform stratified split using Attack_label on the full dataset
    print("[3/4] Creating stratified train/test split …")
    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["Attack_label"],
        random_state=random_state,
    )

    # Build output file names next to the original dataset for convenience
    base_path = data_path.parent
    train_path = base_path / f"{data_path.stem}-training.csv"
    test_path = base_path / f"{data_path.stem}-testing.csv"

    print("[4/4] Saving train/test datasets …")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Report final stats
    def _describe(split_name: str, df):
        total_split = len(df)
        benign = (df["Attack_label"] == 0).sum()
        attack = total_split - benign
        print(
            f" → {split_name:<7}: {total_split:4d} rows | "
            f"Benign: {benign:4d} ({benign/total_split:.2%}) | "
            f"Attack: {attack:4d} ({attack/total_split:.2%})"
        )

    print("\nTrain/test datasets created successfully!")
    _describe("Train", train_df)
    _describe("Test", test_df)

    # Optional: show attack-type distribution in the full dataset
    if "Attack_type" in full_df.columns:
        print("\nAttack type distribution (full dataset):")
        attack_samples_count = len(attack_df)
        dist = attack_df["Attack_type"].value_counts()
        for attack_type, count in dist.items():
            print(f"   {attack_type:<20}: {count:4d} ({count/attack_samples_count:.2%})")


if __name__ == "__main__":
    # If the default path does not exist, inform the user gracefully.
    try:
        create_train_test_split_from_full_dataset()
    except FileNotFoundError as e:
        print(e) 