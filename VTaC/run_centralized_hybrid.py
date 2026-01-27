import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from scipy.stats import kurtosis, skew
from models.cnn.realtime.nets import CNNClassifier
from models.hybrid.realtime.nets import HybridCAVN
from models.cnn.realtime.tools import Dataset_train

# --- MEDICAL IMBALANCE CONFIGURATION ---
PARAMS = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "dropout": 0.5,
    "seed": 42,
    "fn_penalty": 5.0,        # Cost of missing an alarm
    "fp_penalty": 1.0,        # Cost of false alarm
    "oversample_positives": True,  # Balance training data
    "focal_alpha": 0.75,      # Favor positive class in Focal Loss
    "focal_gamma": 2.0,       # Focus on hard examples
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONTEXT FEATURES (Fixed Logic) ---
def get_raw_context_features(ecg):
    if np.std(ecg) < 1e-6: return [0.0, 0.0, 0.0, 0.0]
    k = kurtosis(ecg, fisher=True, bias=False)
    s = skew(ecg, bias=False)
    zcr = ((ecg[:-1] * ecg[1:]) < 0).sum()
    
    # Simple HR estimate
    signal_power = np.mean(ecg ** 2)
    high_freq_power = np.mean(ecg[::5] ** 2)
    hr_proxy = 60.0 + (signal_power - high_freq_power) * 20.0
    return [k, s, float(zcr), np.clip(hr_proxy, 40.0, 180.0)]

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical data"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def cost_sensitive_score(y_true, y_pred, fn_penalty=5.0, fp_penalty=1.0):
    """
    Cost-sensitive scoring for medical alarm detection
    Penalizes missed alarms (FN) more heavily than false alarms (FP)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix elements
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    # Cost-sensitive score: minimize total cost
    total_cost = (fn_penalty * FN) + (fp_penalty * FP)
    total_samples = len(y_true)

    # Normalize by maximum possible cost (all FN + all FP)
    max_cost = (fn_penalty * (y_true == 1).sum()) + (fp_penalty * (y_true == 0).sum())
    score = 100 * (1 - total_cost / max_cost) if max_cost > 0 else 50.0

    return {
        'score': score,
        'cost': total_cost,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'TPR': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'TNR': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'PPV': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'NPV': TN / (TN + FN) if (TN + FN) > 0 else 0
    }

def optimize_threshold(y_true, y_scores, fn_penalty=5.0, fp_penalty=1.0):
    """
    Find optimal threshold that maximizes cost-sensitive score
    """
    best_score = -np.inf
    best_thresh = 0.5
    best_metrics = None

    # Try thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        metrics = cost_sensitive_score(y_true, y_pred, fn_penalty, fp_penalty)

        if metrics['score'] > best_score:
            best_score = metrics['score']
            best_thresh = thresh
            best_metrics = metrics

    return best_thresh, best_metrics
    if np.std(ecg) < 1e-6: return [0.0, 0.0, 0.0, 0.0]
    k = kurtosis(ecg, fisher=True, bias=False)
    s = skew(ecg, bias=False)
    zcr = ((ecg[:-1] * ecg[1:]) < 0).sum()
    
    # Simple HR estimate
    signal_power = np.mean(ecg ** 2)
    high_freq_power = np.mean(ecg[::5] ** 2)
    hr_proxy = 60.0 + (signal_power - high_freq_power) * 20.0
    return [k, s, float(zcr), np.clip(hr_proxy, 40.0, 180.0)]

class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.context_features = []
        self.targets = []
        
        print(f"   -> Processing {len(original_dataset)} samples...", end="\r")
        for i in range(len(original_dataset)):
            signal, target = original_dataset[i]
            feats = get_raw_context_features(signal[0, :].numpy())
            self.context_features.append(feats)
            self.targets.append(target.item())
            
        self.context_features = torch.tensor(self.context_features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def normalize_features(self, mean, std):
        self.context_features = (self.context_features - mean) / (std + 1e-6)

    def __len__(self): return len(self.original_dataset)
    def __getitem__(self, index):
        s, t = self.original_dataset[index]
        return s, t, self.context_features[index]

def load_data_and_stats():
    train_datasets, val_datasets = [], []
    all_train_targets = []

    for hid in [1, 2, 3]:
        data_dir = f"data/hospitals/hospital_{hid:02d}"
        if not os.path.exists(data_dir): continue

        tx, ty, _ = torch.load(os.path.join(data_dir, "train.pt"))
        vx, vy, _ = torch.load(os.path.join(data_dir, "val.pt"))

        zero_nans = lambda x: torch.nan_to_num(x, 0)
        tx = torch.clamp(zero_nans(tx), -10, 10)[:, :, 72500:75000]
        vx = torch.clamp(zero_nans(vx), -10, 10)[:, :, 72500:75000]

        train_datasets.append(PrecomputedDataset(Dataset_train(tx, ty)))
        val_datasets.append(PrecomputedDataset(Dataset_train(vx, vy)))

        # Collect targets for oversampling
        all_train_targets.extend(ty.tolist())

    full_train = ConcatDataset(train_datasets)
    full_val = ConcatDataset(val_datasets)

    # Normalize context features
    all_feats = torch.cat([ds.context_features for ds in train_datasets], dim=0)
    mean, std = all_feats.mean(dim=0), all_feats.std(dim=0)
    for ds in train_datasets + val_datasets: ds.normalize_features(mean, std)

    return full_train, full_val, all_train_targets

def create_balanced_sampler(targets):
    """Create weighted sampler to balance positive/negative classes"""
    targets = np.array(targets)
    pos_count = (targets == 1).sum()
    neg_count = (targets == 0).sum()

    # Weight positives more heavily
    pos_weight = len(targets) / (2 * pos_count)
    neg_weight = len(targets) / (2 * neg_count)

    sample_weights = np.where(targets == 1, pos_weight, neg_weight)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets, context in loader:
        inputs, targets, context = inputs.to(DEVICE), targets.to(DEVICE), context.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs, context).squeeze(-1), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_medical(model, loader, fn_penalty=5.0, fp_penalty=1.0):
    """Medical evaluation with cost-sensitive scoring and threshold optimization"""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets, context in loader:
            inputs, context = inputs.to(DEVICE), context.to(DEVICE)
            outputs = model(inputs, context)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    # Optimize threshold for cost-sensitive score
    best_thresh, best_metrics = optimize_threshold(
        all_targets, all_preds, fn_penalty, fp_penalty
    )

    # Also evaluate at default 0.5 threshold for comparison
    default_pred = (all_preds >= 0.5).astype(int)
    default_metrics = cost_sensitive_score(
        all_targets, default_pred, fn_penalty, fp_penalty
    )

    return {
        'auc': auc,
        'optimal_threshold': best_thresh,
        'optimal_metrics': best_metrics,
        'default_metrics': default_metrics,
        'predictions': all_preds,
        'targets': all_targets
    }

def main():
    torch.manual_seed(PARAMS["seed"])
    np.random.seed(PARAMS["seed"])

    # Load data with oversampling support
    train_data, val_data, train_targets = load_data_and_stats()

    # Create balanced sampler if requested
    sampler = None
    if PARAMS["oversample_positives"]:
        sampler = create_balanced_sampler(train_targets)
        print(f"Using balanced sampling: {len(train_targets)} samples, {(np.array(train_targets) == 1).sum()} positives")

    train_loader = DataLoader(
        train_data,
        batch_size=PARAMS["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None)
    )
    val_loader = DataLoader(val_data, batch_size=PARAMS["batch_size"], shuffle=False)

    # Initialize model
    sample_x, _, sample_c = train_data[0]
    model = HybridCAVN(
        CNNClassifier(inputs=sample_x.shape[0], dropout=PARAMS["dropout"]),
        num_context_features=sample_c.shape[0]
    ).to(DEVICE)

    # Use Focal Loss for better imbalance handling
    criterion = FocalLoss(alpha=PARAMS["focal_alpha"], gamma=PARAMS["focal_gamma"])
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

    print("="*70)
    print(f"Focal Loss (α={PARAMS['focal_alpha']}, γ={PARAMS['focal_gamma']})")
    print(f"Cost weights: FN_penalty={PARAMS['fn_penalty']}, FP_penalty={PARAMS['fp_penalty']}")
    print(f"Oversampling: {'Enabled' if PARAMS['oversample_positives'] else 'Disabled'}")
    print("="*70)

    best_score = -np.inf

    for epoch in range(PARAMS["epochs"]):
        # Train
        loss = train_epoch(model, train_loader, optimizer, criterion)

        # Evaluate with cost-sensitive metrics
        eval_results = evaluate_medical(
            model, val_loader,
            fn_penalty=PARAMS["fn_penalty"],
            fp_penalty=PARAMS["fp_penalty"]
        )

        print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | AUC: {eval_results['auc']:.4f}")
        print(f"  Optimal Threshold: {eval_results['optimal_threshold']:.3f}")
        print(f"  Cost Score: {eval_results['optimal_metrics']['score']:.2f}%")
        print(f"  TPR: {eval_results['optimal_metrics']['TPR']*100:.1f}% | "
              f"TNR: {eval_results['optimal_metrics']['TNR']*100:.1f}%")
        print(f"  FN: {eval_results['optimal_metrics']['FN']} | "
              f"FP: {eval_results['optimal_metrics']['FP']}")

        # Save best model
        if eval_results['optimal_metrics']['score'] > best_score:
            best_score = eval_results['optimal_metrics']['score']
            torch.save(model.state_dict(), "models/medical_hybrid_best.pt")
            print("New best model saved!")

    print("="*70)
    print(f"Training complete. Best cost score: {best_score:.2f}%")
    print("Model saved as: models/medical_hybrid_best.pt")

if __name__ == "__main__":
    main()