import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Sample size for micro dataset (adjust as needed)
TOTAL_SAMPLES = 1500  # This will be split into train/test

def create_merged_micro_dataset(train_file, test_file, total_samples):
    # Read both datasets
    print("Reading datasets...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Merge datasets
    print("Merging datasets...")
    full_df = pd.concat([train_df, test_df], axis=0)
    print(f"Total samples in merged dataset: {len(full_df)}")
    
    # Separate normal and attack samples
    normal = full_df[full_df['label'] == 0]
    attack = full_df[full_df['label'] != 0]
    
    # Calculate proportions from the full dataset
    total = len(full_df)
    normal_prop = len(normal) / total
    
    print(f"\nOriginal class distribution:")
    print(f"Normal samples: {len(normal)} ({normal_prop:.2%})")
    print(f"Attack samples: {len(attack)} ({1-normal_prop:.2%})")
    
    # Calculate samples needed for each class
    normal_samples = int(total_samples * normal_prop)
    attack_samples = total_samples - normal_samples
    
    # Sample from each class
    sampled_normal = normal.sample(n=normal_samples)
    sampled_attack = attack.sample(n=attack_samples)
    
    # Combine and shuffle
    micro_df = pd.concat([sampled_normal, sampled_attack])
    micro_df = micro_df.sample(frac=1).reset_index(drop=True)
    
    # Split into train and test sets (70/30 split)
    train_micro, test_micro = train_test_split(
        micro_df, 
        test_size=0.3, 
        stratify=micro_df['label'],
        random_state=42
    )
    
    # Save micro datasets
    print("\nSaving micro datasets...")
    train_micro.to_csv('UNSW_NB15_training-set.micro.csv', index=False)
    test_micro.to_csv('UNSW_NB15_testing-set.micro.csv', index=False)
    
    # Print statistics
    print("\nMicro dataset statistics:")
    print("Training set:")
    print(f"Total samples: {len(train_micro)}")
    print(f"Normal samples: {len(train_micro[train_micro['label'] == 0])} ({len(train_micro[train_micro['label'] == 0])/len(train_micro):.2%})")
    print(f"Attack samples: {len(train_micro[train_micro['label'] != 0])} ({len(train_micro[train_micro['label'] != 0])/len(train_micro):.2%})")
    
    print("\nTest set:")
    print(f"Total samples: {len(test_micro)}")
    print(f"Normal samples: {len(test_micro[test_micro['label'] == 0])} ({len(test_micro[test_micro['label'] == 0])/len(test_micro):.2%})")
    print(f"Attack samples: {len(test_micro[test_micro['label'] != 0])} ({len(test_micro[test_micro['label'] != 0])/len(test_micro):.2%})")
    
    # Print attack categories distribution
    if 'attack_cat' in micro_df.columns:
        print("\nAttack categories distribution in micro dataset:")
        attack_dist = micro_df[micro_df['label'] != 0]['attack_cat'].value_counts()
        for cat, count in attack_dist.items():
            print(f"{cat}: {count} ({count/len(micro_df[micro_df['label'] != 0]):.2%})")

if __name__ == "__main__":
    create_merged_micro_dataset(
        'UNSW_NB15_training-set.csv',
        'UNSW_NB15_testing-set.csv',
        TOTAL_SAMPLES
    ) 