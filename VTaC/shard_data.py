"""
Data Sharding Script for Federated Learning
Divides the preprocessed VTaC data into multiple hospital silos
"""
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split

def shard_data(num_hospitals=3, val_split=0.1, random_seed=42):
    """
    Shard the training data into multiple hospitals
    Each hospital gets a portion of training data and its own validation split
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load data
    data_dir = "data/out/sample-norm"
    train_x, train_y, train_names = torch.load(os.path.join(data_dir, "train.pt"))
    val_x, val_y, val_names = torch.load(os.path.join(data_dir, "val.pt"))
    test_x, test_y, test_names = torch.load(os.path.join(data_dir, "test.pt"))

    # Create hospital directories
    for i in range(num_hospitals):
        hospital_dir = f"data/hospitals/hospital_{i+1:02d}"
        os.makedirs(hospital_dir, exist_ok=True)

    # Split training data into hospitals
    n_samples = len(train_x)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    samples_per_hospital = n_samples // num_hospitals
    remainder = n_samples % num_hospitals

    start_idx = 0
    for i in range(num_hospitals):
        # Calculate end index, distribute remainder
        extra = 1 if i < remainder else 0
        end_idx = start_idx + samples_per_hospital + extra

        hospital_indices = indices[start_idx:end_idx]

        # Get hospital data
        hospital_train_x = train_x[hospital_indices]
        hospital_train_y = train_y[hospital_indices]
        hospital_train_names = [train_names[idx] for idx in hospital_indices]

        # Split into train/val for this hospital
        train_indices, val_indices = train_test_split(
            np.arange(len(hospital_train_x)),
            test_size=val_split,
            random_state=random_seed + i,
            stratify=hospital_train_y.numpy()
        )

        hospital_local_train_x = hospital_train_x[train_indices]
        hospital_local_train_y = hospital_train_y[train_indices]
        hospital_local_train_names = [hospital_train_names[idx] for idx in train_indices]

        hospital_local_val_x = hospital_train_x[val_indices]
        hospital_local_val_y = hospital_train_y[val_indices]
        hospital_local_val_names = [hospital_train_names[idx] for idx in val_indices]

        # Save hospital data
        hospital_dir = f"data/hospitals/hospital_{i+1:02d}"

        torch.save(
            (hospital_local_train_x, hospital_local_train_y, hospital_local_train_names),
            os.path.join(hospital_dir, "train.pt")
        )
        torch.save(
            (hospital_local_val_x, hospital_local_val_y, hospital_local_val_names),
            os.path.join(hospital_dir, "val.pt")
        )
        # Each hospital also gets a copy of the global test set for evaluation
        torch.save(
            (test_x, test_y, test_names),
            os.path.join(hospital_dir, "test.pt")
        )

        print(f"Hospital {i+1}: {len(hospital_local_train_x)} train, {len(hospital_local_val_x)} val samples")
        start_idx = end_idx

    print(f"\nData sharding complete! Created {num_hospitals} hospital silos.")
    print("Each hospital has:")
    print("- Local training data")
    print("- Local validation data")
    print("- Global test data for evaluation")

if __name__ == "__main__":
    shard_data(num_hospitals=3, val_split=0.1)