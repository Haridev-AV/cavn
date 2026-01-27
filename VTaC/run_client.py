"""
Federated Learning Client Runner for VTaC
Run this script for each hospital to participate in federated learning
"""
import flwr as fl
import sys
import os
import argparse
from fl_client import VTaCClient


def main():
    """Main function to run a hospital client"""
    parser = argparse.ArgumentParser(description="VTaC Federated Learning Client")
    parser.add_argument("hospital_id", type=int, help="Hospital ID (1-3)")
    parser.add_argument("--mode", type=str, choices=["cnn", "hybrid"], default="cnn", 
                       help="Model type to use (default: cnn)")
    args = parser.parse_args()

    hospital_id = args.hospital_id
    mode = args.mode

    # Validate hospital_id
    if hospital_id < 1 or hospital_id > 3:
        print("Hospital ID must be between 1 and 3")
        sys.exit(1)

    # Set data directory for this hospital
    data_dir = f"data/hospitals/hospital_{hospital_id:02d}"

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        print("Please run shard_data.py first to create hospital data.")
        sys.exit(1)

    print("=" * 60)
    print(f"Starting VTaC Federated Learning Client - Hospital {hospital_id}")
    print(f"Model Mode: {mode.upper()}")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Create client parameters
    params = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "differ_loss_weight": 0.0,  # Reset to prevent over-fitting
        "adam_weight_decay": 0.005,
        "weighted_class": 2.0,  # Reset to prevent always-positive predictions
        "dropout": 0.3,
        "local_epochs": 1,
        "seed": 42 + hospital_id
    }

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=VTaCClient(hospital_id, data_dir, params, model_type=mode).to_client(),
    )


if __name__ == "__main__":
    main()