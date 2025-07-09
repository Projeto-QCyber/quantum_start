import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------------------------------------------------------
# Updated script to create a micro (reduced) version of the Edge-IIoT dataset
# (ML-EdgeIIoT-dataset.csv). The script keeps the original benign/attack ratio
# found in the full dataset, then performs a stratified 70/30 train/test split
# and stores the resulting files next to the original csv.
# -----------------------------------------------------------------------------

# Total number of samples to keep in the micro version
TOTAL_SAMPLES = 1500  # adjust as required


def create_micro_dataset_edgeiot(
    data_file: str = "../datasets/ML-EdgeIIoT-dataset.csv",
    total_samples: int = TOTAL_SAMPLES,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Create a reduced micro dataset from the Edge-IIoT CSV file.

    Parameters
    ----------
    data_file : str
        Path to the full CSV dataset containing an `Attack_label` column where
        0 = benign traffic and any other value = attack.
    total_samples : int
        How many samples (rows) the micro dataset should contain in total.
    test_size : float
        Proportion of the micro dataset that will be reserved for the test
        split. The remainder is used for training.
    random_state : int
        Seed to ensure reproducible shuffling/sampling.
    """

    print("\n[1/5] Reading dataset …")
    full_df = pd.read_csv(data_file)

    if "Attack_label" not in full_df.columns:
        raise KeyError(
            "Column 'Attack_label' not found. Please verify that the provided "
            "CSV is the original ML-EdgeIIoT dataset."
        )

    # Identify benign vs. attack samples using Attack_label
    print("[2/5] Analysing class distribution …")
    normal_df = full_df[full_df["Attack_label"] == 0]
    attack_df = full_df[full_df["Attack_label"] != 0]

    total = len(full_df)
    normal_ratio = len(normal_df) / total

    print(f"   Total samples         : {total:,}")
    print(f"   Benign (label==0)     : {len(normal_df):,} ({normal_ratio:.2%})")
    print(f"   Attack (label!=0)     : {len(attack_df):,} ({1-normal_ratio:.2%})")

    # Determine how many samples to draw from each class so that the micro
    # dataset preserves the original ratio.
    benign_samples = int(total_samples * normal_ratio)
    attack_samples = total_samples - benign_samples  # ensure exact total

    print("[3/5] Sampling …")
    sampled_benign = normal_df.sample(n=benign_samples, random_state=random_state)
    sampled_attack = attack_df.sample(n=attack_samples, random_state=random_state)

    micro_df = (
        pd.concat([sampled_benign, sampled_attack])
        .sample(frac=1, random_state=random_state)  # shuffle
        .reset_index(drop=True)
    )

    # Perform stratified split using Attack_label
    print("[4/5] Creating stratified train/test split …")
    train_df, test_df = train_test_split(
        micro_df,
        test_size=test_size,
        stratify=micro_df["Attack_label"],
        random_state=random_state,
    )

    # Build output file names next to the original dataset for convenience
    base_path = data_file.rsplit("/", 1)[0] or "."
    train_path = f"{base_path}/ML-EdgeIIoT-training.micro.csv"
    test_path = f"{base_path}/ML-EdgeIIoT-testing.micro.csv"

    print("[5/5] Saving micro datasets …")
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

    print("\nMicro dataset created successfully!")
    _describe("Train", train_df)
    _describe("Test", test_df)

    # Optional: show attack-type distribution in the micro dataset
    if "Attack_type" in micro_df.columns:
        print("\nAttack type distribution (micro dataset):")
        dist = micro_df[micro_df["Attack_label"] != 0]["Attack_type"].value_counts()
        for attack_type, count in dist.items():
            print(f"   {attack_type:<20}: {count:4d} ({count/attack_samples:.2%})")


if __name__ == "__main__":
    # If the default path does not exist, inform the user gracefully.
    try:
        create_micro_dataset_edgeiot()
    except FileNotFoundError as e:
        print(
            "Could not locate 'datasets/ML-EdgeIIoT-dataset.csv'. "
            "Please provide the correct path when calling "
            "create_micro_dataset_edgeiot(data_file=…)."
        ) 