import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import flwr as fl
from sklearn.metrics import roc_auc_score
from scipy.stats import kurtosis, skew

# Import Models
from models.cnn.realtime.nets import CNNClassifier
from models.hybrid.realtime.nets import HybridCAVN

# Conditionally import tools based on model type
def get_model_tools(model_type):
    if model_type.lower() == "hybrid":
        from models.hybrid.realtime.tools import Dataset_train, train_model, eval_model
    else:  # default to cnn
        from models.cnn.realtime.tools import Dataset_train, train_model, eval_model
    return Dataset_train, train_model, eval_model

# Import tools will be done dynamically in __init__ 

class VTaCClient(fl.client.NumPyClient):
    def __init__(self, hospital_id, data_dir, params, model_type="cnn"):
        self.hospital_id = hospital_id
        self.data_dir = data_dir
        self.params = params
        self.model_type = model_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dynamically import tools based on model type
        self.Dataset_train, self.train_model, self.eval_model = get_model_tools(self.model_type)

        # Reproducibility
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])

        self.load_data()
        self.initialize_model()

    def initialize_model(self):
        """Clean initialization logic"""
        # 1. Base CNN
        base_cnn = CNNClassifier(
            inputs=self.num_channels,
            dropout=self.params["dropout"]
        ).to(self.device)

        if self.model_type == "hybrid":
            print(f"Hospital {self.hospital_id}: Initializing HybridCAVN")
            # Wrap the CNN in the Hybrid architecture
            self.model = HybridCAVN(base_cnn, num_context_features=4).to(self.device)
        else:
            print(f"Hospital {self.hospital_id}: Initializing Baseline CNN")
            self.model = base_cnn

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["adam_weight_decay"]
        )
        
        # Loss Function (BCEWithLogitsLoss includes Sigmoid)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.params["weighted_class"]]).to(self.device)
        )

    def load_data(self):
        # ... (Keep your existing data loading logic exactly as is) ...
        # Ensure drop_last=True for train_loader
        train_x, train_y, _ = torch.load(os.path.join(self.data_dir, "train.pt"))
        val_x, val_y, _ = torch.load(os.path.join(self.data_dir, "val.pt"))
        
        # Basic Preprocessing
        zero_nans = lambda x: torch.nan_to_num(x, 0)
        train_x = torch.clamp(zero_nans(train_x), -10, 10)
        val_x = torch.clamp(zero_nans(val_x), -10, 10)

        self.num_channels = train_x.shape[1]
        self.train_loader = DataLoader(
            self.Dataset_train(train_x, train_y), 
            batch_size=self.params["batch_size"], shuffle=True, drop_last=True
        )
        self.val_loader = DataLoader(
            self.Dataset_train(val_x, val_y), 
            batch_size=self.params["batch_size"], shuffle=False
        )

    def get_context_features(self, x_batch):
        """Extract statistical features on CPU"""
        x_np = x_batch.cpu().numpy()
        features = []
        for i in range(x_np.shape[0]):
            ecg = x_np[i, 0, :]
            # Safety check for constant signals (prevents NaN in kurtosis/skew)
            if np.std(ecg) < 1e-6:
                features.append([0, 0, 0, 0])
                continue
                
            k = kurtosis(ecg, fisher=True, bias=False)
            s = skew(ecg, bias=False)
            # Simple zero crossing rate
            zcr = ((ecg[:-1] * ecg[1:]) < 0).sum()
            # Placeholder for HR diff or other features
            hr = 0.0 
            features.append([k, s, zcr, hr])
            
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def forward_pass(self, batch):
        """Handles data routing for both CNN and Hybrid modes"""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.model_type == "hybrid":
            context = self.get_context_features(inputs)
            outputs = self.model(inputs, context)
        else:
            outputs = self.model(inputs)
            
        return outputs, targets

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        epoch_loss = 0.0
        batches = 0
        
        for epoch in range(self.params["local_epochs"]):
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                # Use the proper training pipeline from tools.py
                loss, _, _, _ = self.train_model(
                    batch, 
                    self.model, 
                    self.criterion, 
                    self.device, 
                    weight=self.params.get("differ_loss_weight", 0.0)
                )
                
                # Ensure gradient calculation with tiny epsilon
                loss = loss + 1e-6
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1

        avg_loss = epoch_loss / batches if batches > 0 else 0.0
        
        # Calculate training metrics
        train_metrics = self.evaluate_metrics(self.train_loader, subset=True)
        print(f"Hospital {self.hospital_id} Train: Loss {avg_loss:.4f}, AUC {train_metrics['AUC']:.4f}")
        
        return self.get_parameters({}), len(self.train_loader.dataset), {
            "loss": avg_loss, 
            "AUC": train_metrics["AUC"]
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        val_metrics = self.evaluate_metrics(self.val_loader)
        print(f"Hospital {self.hospital_id} Eval: AUC {val_metrics['AUC']:.4f}, Score {val_metrics['Score']:.2f}")
        
        return val_metrics["loss"], len(self.val_loader.dataset), val_metrics

    def evaluate_metrics(self, loader, subset=False):
        """Unified metric calculation for train/val"""
        total_loss = 0.0
        all_preds = []
        all_targets = []
        batches = 0
        
        # If subset is True, only check first 50 batches to save time
        limit = 50 if subset else float('inf')

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= limit: break
                
                # Use the proper evaluation pipeline from tools.py
                loss, predictions, targets = self.eval_model(
                    batch, 
                    self.model, 
                    self.criterion, 
                    self.device
                )
                
                total_loss += loss.item()
                # predictions are already sigmoid-applied in eval_model
                all_preds.extend(torch.sigmoid(predictions).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                batches += 1
        
        # Metrics Logic
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
            
        preds_bin = np.array(all_preds) >= 0.3
        targs = np.array(all_targets)
        
        TP = ((preds_bin == 1) & (targs == 1)).sum()
        TN = ((preds_bin == 0) & (targs == 0)).sum()
        FP = ((preds_bin == 1) & (targs == 0)).sum()
        FN = ((preds_bin == 0) & (targs == 1)).sum()
        
        TPR = TP / (TP + FN) if (TP+FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN+FP) > 0 else 0
        Score = 100 * (TP+TN) / (TP+TN+FP+5*FN) if (TP+TN+FP+5*FN) > 0 else 0
        
        return {
            "loss": total_loss / batches if batches > 0 else 0,
            "AUC": auc,
            "TPR": TPR * 100,
            "TNR": TNR * 100,
            "Score": Score
        }

    # Helper methods for Flower
    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.parameters(), parameters)
        with torch.no_grad():
            for param, param_np in params_dict:
                param.copy_(torch.from_numpy(param_np))

def create_client(hospital_id, data_dir, model_type="cnn"):
    params = {
        "batch_size": 32,
        "learning_rate": 0.001, 
        "adam_weight_decay": 0.0001,
        "weighted_class": 2.0, # Conservative start to prevent collapse
        "dropout": 0.3,
        "local_epochs": 1,
        "seed": 42 + hospital_id
    }
    return VTaCClient(hospital_id, data_dir, params, model_type)