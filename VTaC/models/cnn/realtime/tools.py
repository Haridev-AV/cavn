import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


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

    # model prediction, feature of alarm signal, feature of randomly selected signal
    Y_train_prediction, s_f, random_s = model(signal_train, random_s)
    
    # Check for NaN/Inf in model outputs
    if torch.isnan(Y_train_prediction).any() or torch.isinf(Y_train_prediction).any():
        # Replace NaN/Inf with zeros
        Y_train_prediction = torch.nan_to_num(Y_train_prediction, nan=0.0, posinf=10.0, neginf=-10.0)
    if torch.isnan(s_f).any() or torch.isinf(s_f).any():
        s_f = torch.nan_to_num(s_f, nan=0.0, posinf=1.0, neginf=-1.0)
    if torch.isnan(random_s).any() or torch.isinf(random_s).any():
        random_s = torch.nan_to_num(random_s, nan=0.0, posinf=1.0, neginf=-1.0)

    feature_size = s_f.shape[1]
    # calculate loss - squeeze prediction to match target shape
    # Clamp predictions to prevent extreme values
    Y_pred_squeezed = Y_train_prediction.squeeze(-1)
    Y_pred_squeezed = torch.clamp(Y_pred_squeezed, min=-10, max=10)
    loss = loss_ce(Y_pred_squeezed, y_train)
    
    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        # If loss is NaN, use a small constant loss to prevent training from breaking
        loss = torch.tensor(0.01, device=device, requires_grad=True)

    # Initialize differ_loss
    differ_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Only calculate differ_loss if we have samples of both classes
    if false_alarm_batch > 0:
        # feature of randomly selected signal from a sample with a false alarm
        false_random_s = random_s[false_alarm_index]
        # feature of alarm signal from a sample with a false alarm
        false_s_f = s_f[false_alarm_index]
        
        # Check for NaN/Inf before calculation
        if not (torch.isnan(false_s_f).any() or torch.isnan(false_random_s).any() or 
                torch.isinf(false_s_f).any() or torch.isinf(false_random_s).any()):
            false_dot = torch.bmm(
                false_s_f.view(false_alarm_batch, 1, feature_size),
                false_random_s.view(false_alarm_batch, feature_size, 1),
            )
            # Clamp to prevent extreme values
            false_dot = torch.clamp(false_dot, min=-10, max=10)
            differ_loss = differ_loss + (-torch.mean(F.logsigmoid(false_dot)))
    
    if true_alarm_batch > 0:
        # feature of randomly selected signal from a sample with a true alarm
        true_random_s = random_s[true_alarm_index]
        # feature of alarm signal from a sample with a true alarm
        true_s_f = s_f[true_alarm_index]
        
        # Check for NaN/Inf before calculation
        if not (torch.isnan(true_s_f).any() or torch.isnan(true_random_s).any() or 
                torch.isinf(true_s_f).any() or torch.isinf(true_random_s).any()):
            true_dot = torch.bmm(
                true_s_f.view(true_alarm_batch, 1, feature_size),
                true_random_s.view(true_alarm_batch, feature_size, 1),
            )
            # Clamp to prevent extreme values
            true_dot = torch.clamp(true_dot, min=-10, max=10)
            differ_loss = differ_loss + (-torch.mean(F.logsigmoid(-true_dot)))
    
    # If differ_loss is NaN or Inf, set to zero
    if torch.isnan(differ_loss) or torch.isinf(differ_loss):
        differ_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss, weight * differ_loss, Y_train_prediction.squeeze(-1), y_train


def eval_model(batch, model, loss_ce, device):
    signal_train, y_train = batch
    length = 2500
    # alarm signal
    signal_train = signal_train[:, :, 72500:75000].to(device)

    y_train = y_train.to(device)

    # prediction
    Y_train_prediction = model(signal_train)
    
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


def evaluation(Y_eval_prediction, y_test, TP, FP, TN, FN):
    # set 0 if false alarm else 1
    pre = (Y_eval_prediction >= 0).int()
    # pair: (prediction, groundtruth)
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:
            TP += 1
        if i.item() == 1 and j.item() == 0:
            FP += 1
        if i.item() == 0 and j.item() == 0:
            TN += 1
        if i.item() == 0 and j.item() == 1:
            FN += 1
    return TP, FP, TN, FN


def evaluate_rule_based(rule_based_results, y_test):
    TP = FP = TN = FN = 0
    # pair: (prediction, groundtruth)
    for i, j in zip(rule_based_results, y_test):
        if i.item() == 1 and j.item() == 1:
            TP += 1
        if i.item() == 1 and j.item() == 0:
            FP += 1
        if i.item() == 0 and j.item() == 0:
            TN += 1
        if i.item() == 0 and j.item() == 1:
            FN += 1
    return (
        100 * TP / (TP + FN),
        100 * TN / (TN + FP),
        100 * (TP + TN) / (TP + TN + FP + 5 * FN),
        100 * (TP + TN) / (TP + TN + FP + FN),
    )


def evaluation_test(Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN):
    pre = (Y_eval_prediction >= 0).int()
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:
            types_TP += 1
        if i.item() == 1 and j.item() == 0:
            types_FP += 1
        if i.item() == 0 and j.item() == 0:
            types_TN += 1
        if i.item() == 0 and j.item() == 1:
            types_FN += 1
    return types_TP, types_FP, types_TN, types_FN


def evaluate_raise_threshold(
    prediction, groundtruth, types_TP, types_FP, types_TN, types_FN, threshold
):
    prediction = torch.sigmoid(prediction)

    pre = 1 if prediction >= threshold else 0
    # identify if a sample is TP, FP, TN, or FN
    if pre == 1 and groundtruth == 1:
        types_TP += 1
    elif pre == 1 and groundtruth == 0:
        types_FP += 1
    elif pre == 0 and groundtruth == 1:
        types_FN += 1
    elif pre == 0 and groundtruth == 0:
        types_TN += 1

    return types_TP, types_FP, types_TN, types_FN


def eval(batch, model, device):
    signal_train, y_train = batch
    length = 2500

    signal_train = signal_train[:, :, 72500:75000].to(device)

    y_train = y_train.to(device)

    Y_train_prediction = model(signal_train)

    return Y_train_prediction.squeeze(-1), y_train
