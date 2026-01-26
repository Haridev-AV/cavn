"""
Federated Learning Client Runner for VTaC
Run this script for each hospital to participate in federated learning
"""
import flwr as fl
import sys
import os
from fl_client import VTaCClient


def main():
    """Main function to run a hospital client"""
    if len(sys.argv) != 2:
        print("Usage: python run_client.py <hospital_id>")
        print("Example: python run_client.py 1")
        sys.exit(1)

    hospital_id = int(sys.argv[1])

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
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Create client parameters
    params = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "differ_loss_weight": 0.2,
        "adam_weight_decay": 0.005,
        "weighted_class": 6.0,
        "dropout": 0.3,
        "local_epochs": 1,
        "seed": 42 + hospital_id
    }

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=VTaCClient(hospital_id, data_dir, params).to_client(),
    )


if __name__ == "__main__":
    main()