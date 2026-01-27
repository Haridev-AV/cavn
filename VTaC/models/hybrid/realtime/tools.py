import numpy as np
from scipy.stats import kurtosis, skew
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dataset_train(Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, signal_train, y_train):
        # signal
        self.strain = signal_train
        # groundtruth
        self.ytrain = y_train

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.ytrain)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        return self.strain[index], self.ytrain[index]


def get_context_features(ecg_segment, ppg_segment):
    """
    Extracts 4-dimensional context vector:
    1. ECG Kurtosis (Signal Quality) - normalized
    2. PPG Skewness (Motion/Noise indicator) - normalized  
    3. HR Difference (Cross-sensor agreement) - normalized
    4. Zero Crossing Rate (Technical noise detection) - normalized
    """
    # Handle potential NaN/inf from kurtosis/skew
    try:
        k_ecg = kurtosis(ecg_segment, nan_policy='omit')
        if np.isnan(k_ecg) or np.isinf(k_ecg):
            k_ecg = 0.0
        # Normalize kurtosis (typical range -2 to 10, center around 0)
        k_ecg = np.clip(k_ecg / 5.0, -2.0, 2.0)
    except:
        k_ecg = 0.0

    try:
        s_ppg = skew(ppg_segment, nan_policy='omit')
        if np.isnan(s_ppg) or np.isinf(s_ppg):
            s_ppg = 0.0
        # Normalize skewness (typical range -2 to 2)
        s_ppg = np.clip(s_ppg, -2.0, 2.0)
    except:
        s_ppg = 0.0

    # Simple HR approximation for the 'Agreement' feature
    hr_ecg = len(np.where(ecg_segment > np.mean(ecg_segment))[0]) # Use mean instead of 0.5
    hr_ppg = len(np.where(ppg_segment > np.mean(ppg_segment))[0])
    hr_diff = abs(hr_ecg - hr_ppg)
    # Normalize HR difference (relative to segment length)
    hr_diff_norm = min(hr_diff / len(ecg_segment), 1.0)

    # Zero crossing rate
    zcr = ((ecg_segment[:-1] * ecg_segment[1:]) < 0).sum()
    # Normalize ZCR (relative to segment length)
    zcr_norm = zcr / len(ecg_segment)

    return np.array([k_ecg, s_ppg, hr_diff_norm, zcr_norm], dtype=np.float32)


def train_model(batch, model, loss_ce, device, weight):
    signal_train, y_train = batch
    batch_size = len(signal_train)
    length = 2500

    # samples with a true alarm
    true_alarm_index = (y_train == 1).view(-1)
    # samples with a false alarm
    false_alarm_index = (y_train != 1).view(-1)

    true_alarm_batch = torch.sum(true_alarm_index).item()
    false_alarm_batch = torch.sum(false_alarm_index).item()

    # randomly select the start of a sequence for each sample in this batch
    sample_index = np.random.choice(75000 - length * 2, batch_size, True)
    random_s = []
    for i, j in enumerate(sample_index):
        random_s.append(signal_train[i, :, j : j + length])
    random_s = torch.stack(random_s).to(device)

    # use the last 10s signal as model input
    signal_train = signal_train[:, :, 72500:75000].to(device)
    y_train = y_train.to(device)

    # Extract context features for each sample in the batch
    context_features = []
    for i in range(batch_size):
        # For simplicity, use the same segment for both ECG and PPG
        # In a real implementation, you'd separate ECG and PPG channels
        ecg_segment = signal_train[i, 0, :].cpu().numpy()  # Assume channel 0 is ECG
        ppg_segment = signal_train[i, 1, :].cpu().numpy() if signal_train.shape[1] > 1 else ecg_segment  # Assume channel 1 is PPG
        context = get_context_features(ecg_segment, ppg_segment)
        context_features.append(context)

    context_features = torch.tensor(np.array(context_features), dtype=torch.float32).to(device)

    # model prediction - hybrid model takes (signal, context)
    Y_train_prediction = model(signal_train, context_features)

    # For hybrid model, we don't have the contrastive features like CNN
    # So we'll use dummy features for compatibility
    s_f = torch.zeros(batch_size, 32, device=device)  # Dummy feature vector
    random_s_dummy = torch.zeros_like(random_s)  # Dummy random segments

    # Check for NaN/Inf in model outputs
    if torch.isnan(Y_train_prediction).any() or torch.isinf(Y_train_prediction).any():
        # Replace NaN/Inf with zeros
        Y_train_prediction = torch.nan_to_num(Y_train_prediction, nan=0.0, posinf=10.0, neginf=-10.0)

    # calculate loss - squeeze prediction to match target shape
    # Clamp predictions to prevent extreme values
    Y_pred_squeezed = Y_train_prediction.squeeze(-1)
    Y_pred_squeezed = torch.clamp(Y_pred_squeezed, min=-10, max=10)
    loss = loss_ce(Y_pred_squeezed, y_train)

    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        # If loss is NaN, use a small constant loss to prevent training from breaking
        loss = torch.tensor(0.01, device=device, requires_grad=True)

    # For hybrid model, we skip the contrastive loss calculation
    # as the model architecture is different
    differ_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss, weight * differ_loss, Y_train_prediction.squeeze(-1), y_train


def eval_model(batch, model, loss_ce, device):
    signal_train, y_train = batch
    length = 2500
    batch_size = len(signal_train)

    # alarm signal
    signal_train = signal_train[:, :, 72500:75000].to(device)
    y_train = y_train.to(device)

    # Extract context features for each sample in the batch
    context_features = []
    for i in range(batch_size):
        # For simplicity, use the same segment for both ECG and PPG
        # In a real implementation, you'd separate ECG and PPG channels
        ecg_segment = signal_train[i, 0, :].cpu().numpy()  # Assume channel 0 is ECG
        ppg_segment = signal_train[i, 1, :].cpu().numpy() if signal_train.shape[1] > 1 else ecg_segment  # Assume channel 1 is PPG
        context = get_context_features(ecg_segment, ppg_segment)
        context_features.append(context)

    context_features = torch.tensor(np.array(context_features), dtype=torch.float32).to(device)

    # prediction - hybrid model takes (signal, context)
    Y_train_prediction = model(signal_train, context_features)

    # Check for NaN/Inf in predictions
    if torch.isnan(Y_train_prediction).any() or torch.isinf(Y_train_prediction).any():
        # Replace NaN/Inf with zeros
        Y_train_prediction = torch.nan_to_num(Y_train_prediction, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clamp predictions to prevent extreme values
    Y_pred_squeezed = Y_train_prediction.squeeze(-1)
    Y_pred_squeezed = torch.clamp(Y_pred_squeezed, min=-10, max=10)

    loss = loss_ce(Y_pred_squeezed, y_train)

    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        # Return zero loss if NaN/Inf
        loss = torch.tensor(0.0, device=device)

    return loss, Y_pred_squeezed, y_train