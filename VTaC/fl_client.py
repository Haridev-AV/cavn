"""
Federated Learning Client for VTaC CNN Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import flwr as fl
from sklearn.metrics import roc_auc_score

# Import the model and utilities
from models.cnn.realtime.nets import CNNClassifier
from models.cnn.realtime.tools import Dataset_train, train_model, eval_model, evaluation


class VTaCClient(fl.client.NumPyClient):
    def __init__(self, hospital_id, data_dir, params):
        self.hospital_id = hospital_id
        self.data_dir = data_dir
        self.params = params

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds for reproducibility
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])

        # Load data
        self.load_data()

        # Create model
        self.model = CNNClassifier(
            inputs=self.num_channels,
            dropout=params["dropout"]
        ).to(self.device)

        # Create optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["adam_weight_decay"]
        )
        self.loss_ce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([params["weighted_class"]]).to(self.device)
        )

    def load_data(self):
        """Load hospital-specific data"""
        train_x, train_y, _ = torch.load(os.path.join(self.data_dir, "train.pt"))
        val_x, val_y, _ = torch.load(os.path.join(self.data_dir, "val.pt"))
        test_x, test_y, _ = torch.load(os.path.join(self.data_dir, "test.pt"))

        # Apply data preprocessing (same as original)
        zero_nans = lambda x: torch.nan_to_num(x, 0)
        clip_value = 10.0

        train_x = zero_nans(train_x)
        val_x = zero_nans(val_x)
        test_x = zero_nans(test_x)

        train_x = torch.clamp(train_x, -clip_value, clip_value)
        val_x = torch.clamp(val_x, -clip_value, clip_value)
        test_x = torch.clamp(test_x, -clip_value, clip_value)

        self.num_channels = train_x.shape[1]

        # Create datasets
        self.train_dataset = Dataset_train(train_x, train_y)
        self.val_dataset = Dataset_train(val_x, val_y)
        self.test_dataset = Dataset_train(test_x, test_y)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=0
        )

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.parameters(), parameters)
        with torch.no_grad():
            for param, param_np in params_dict:
                param.copy_(torch.from_numpy(param_np))

    def fit(self, parameters, config):
        """Train the model on local data"""
        # Set model parameters
        self.set_parameters(parameters)

        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.params["local_epochs"]):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch in self.train_loader:
                loss, differ_loss, _, _ = train_model(
                    batch,
                    self.model,
                    self.loss_ce,
                    self.device,
                    weight=self.params["differ_loss_weight"]
                )

                # Skip if NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                combined_loss = loss + differ_loss

                if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                combined_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                # Handle NaN gradients
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad = torch.where(
                                torch.isnan(param.grad) | torch.isinf(param.grad),
                                torch.zeros_like(param.grad),
                                param.grad
                            )
                            has_nan_grad = True

                if not has_nan_grad:
                    self.optimizer.step()

                epoch_loss += combined_loss.item()
                epoch_batches += 1

            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                print(f"Hospital {self.hospital_id} - Epoch {epoch+1}/{self.params['local_epochs']}: Loss = {avg_epoch_loss:.4f}")

        # Return updated parameters, number of training samples, and metrics
        num_samples = len(self.train_dataset)

        # Calculate training metrics on a subset of training data for global tracking
        train_metrics = self.calculate_training_metrics()

        metrics = {
            "loss": avg_epoch_loss if epoch_batches > 0 else 0.0,
            "TPR": train_metrics["TPR"],
            "AUC": train_metrics["AUC"]
        }

        return self.get_parameters({}), num_samples, metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on local validation data"""
        # Set model parameters
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # For metrics calculation
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                loss, predictions, targets = eval_model(
                    batch,
                    self.model,
                    self.loss_ce,
                    self.device
                )

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                # Collect predictions and targets for metrics
                all_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Calculate TPR, TNR, etc.
        predictions_binary = np.array(all_predictions) >= 0.5
        targets = np.array(all_targets)

        TP = np.sum((predictions_binary == 1) & (targets == 1))
        FP = np.sum((predictions_binary == 1) & (targets == 0))
        TN = np.sum((predictions_binary == 0) & (targets == 0))
        FN = np.sum((predictions_binary == 0) & (targets == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

        # Calculate AUC
        try:
            auc = roc_auc_score(targets, all_predictions)
        except:
            auc = 0.5

        # Calculate Score (from original paper)
        score = 100 * (TP + TN) / (TP + TN + FP + 5 * FN) if (TP + TN + FP + 5 * FN) > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "TPR": TPR * 100,
            "TNR": TNR * 100,
            "PPV": PPV * 100,
            "AUC": auc,
            "Score": score,
            "ACC": ACC * 100
        }

        num_samples = len(self.val_dataset)

        print(f"Hospital {self.hospital_id} - Eval Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Score: {score:.2f}")

        return avg_loss, num_samples, metrics

    def calculate_training_metrics(self):
        """Calculate TPR and AUC on a subset of training data for global tracking"""
        self.model.eval()

        # Use a small subset of training data for metrics calculation
        subset_size = min(1000, len(self.train_dataset))  # Use up to 1000 samples
        indices = torch.randperm(len(self.train_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(self.train_dataset, indices)
        subset_loader = DataLoader(subset_dataset, batch_size=self.params["batch_size"], shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in subset_loader:
                _, predictions, targets = eval_model(
                    batch,
                    self.model,
                    self.loss_ce,
                    self.device
                )
                all_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate TPR and AUC
        predictions_binary = np.array(all_predictions) >= 0.5
        targets = np.array(all_targets)

        TP = np.sum((predictions_binary == 1) & (targets == 1))
        FN = np.sum((predictions_binary == 0) & (targets == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        try:
            auc = roc_auc_score(targets, all_predictions)
        except:
            auc = 0.5

        return {
            "TPR": TPR * 100,
            "AUC": auc
        }


def create_client(hospital_id, data_dir):
    """Factory function to create a client for a specific hospital"""
    params = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "differ_loss_weight": 0.2,  # Further increase contrastive learning for better TPR
        "adam_weight_decay": 0.005,
        "weighted_class": 6.0,  # Higher weight to further penalize missed alarms
        "dropout": 0.3,
        "local_epochs": 1,  # One epoch per round
        "seed": 42 + hospital_id  # Different seed per hospital
    }

    return VTaCClient(hospital_id, data_dir, params)