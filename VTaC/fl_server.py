"""
Federated Learning Server for VTaC CNN Model
Coordinates training across multiple hospital clients using FedAvg
"""
import flwr as fl
import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate metrics weighted by the number of samples for AUC, TPR, TNR, ACC, and Score"""
    if not metrics:
        return {}

    # Extract weights and metrics
    weights = [num_samples for num_samples, _ in metrics]
    total_weight = sum(weights)

    # Initialize aggregated metrics for specific keys
    aggregated = {}
    target_keys = ['AUC', 'TPR', 'TNR', 'ACC', 'Score']

    for _, metric_dict in metrics:
        for key in target_keys:
            if key in metric_dict:
                if key not in aggregated:
                    aggregated[key] = 0.0
                # Weight by the number of samples
                weight_idx = metrics.index((_, metric_dict))
                aggregated[key] += (metric_dict[key] * weights[weight_idx]) / total_weight

    return aggregated


def fit_config(server_round: int):
    """Return training configuration dict for each round"""
    config = {
        "server_round": server_round,
        "local_epochs": 1,  # One epoch per round
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round"""
    return {"server_round": server_round}


class VTaCServer:
    def __init__(self, num_rounds=10, num_clients=3, fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3):
        self.num_rounds = num_rounds
        self.num_clients = num_clients

        # Define strategy
        self.strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,  # Sample 100% of available clients for training
            fraction_evaluate=fraction_evaluate,  # Sample 100% of available clients for evaluation
            min_fit_clients=min_fit_clients,  # Never sample less than 3 clients for training
            min_evaluate_clients=min_evaluate_clients,  # Never sample less than 3 clients for evaluation
            min_available_clients=min_available_clients,  # Wait until all 3 clients are available
            evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate evaluation metrics
            fit_metrics_aggregation_fn=weighted_average,  # Aggregate training metrics
            on_fit_config_fn=fit_config,  # Configuration for training
            on_evaluate_config_fn=evaluate_config,  # Configuration for evaluation
        )

    def start_server(self):
        """Start the federated learning server"""
        print("=" * 60)
        print("Starting VTaC Federated Learning Server")
        print("=" * 60)
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Number of clients: {self.num_clients}")
        print("=" * 60)

        # Start Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )


def main():
    """Main function to run the FL server"""
    # Create and start server
    server = VTaCServer(
        num_rounds=10,  # 10 rounds of federated learning
        num_clients=3,  # 3 hospital clients
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )

    server.start_server()


if __name__ == "__main__":
    main()