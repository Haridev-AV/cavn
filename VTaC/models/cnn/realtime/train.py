import torch
import os
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import *
import time
from tools import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import sklearn
import sys

if __name__ == "__main__":
    SEED = 1 if len(sys.argv) <= 6 else int(sys.argv[6])
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the project root directory (3 levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, "data", "out", "sample-norm")
    os.chdir(data_dir)
    # load preprocessed dataset
    trainset_x, trainset_y, train_names = torch.load("train.pt")
    valset_x, valset_y, val_names = torch.load("val.pt")
    testset_x, testset_y, test_names = torch.load("test.pt")
    num_channels = trainset_x.shape[1]

    zero_nans = lambda x: torch.nan_to_num(x, 0)

    trainset_x = zero_nans(trainset_x)
    testset_x = zero_nans(testset_x)
    valset_x = zero_nans(valset_x)
    
    # Clip extreme values to prevent overflow/underflow
    clip_value = 10.0  # Adjust based on your data scale
    trainset_x = torch.clamp(trainset_x, -clip_value, clip_value)
    valset_x = torch.clamp(valset_x, -clip_value, clip_value)
    testset_x = torch.clamp(testset_x, -clip_value, clip_value)
    
    # Check for NaN/Inf in data (before logger is created)
    if torch.isnan(trainset_x).any() or torch.isinf(trainset_x).any():
        print("Warning: NaN or Inf values found in training data")
    if torch.isnan(valset_x).any() or torch.isinf(valset_x).any():
        print("Warning: NaN or Inf values found in validation data")

    batch_size = int(sys.argv[1])
    lr = float(sys.argv[2])
    dl = float(sys.argv[3])
    dropout_probability = float(sys.argv[4])
    positive_class_weight = float(sys.argv[5])
    max_epoch = int(sys.argv[6]) if len(sys.argv) > 6 else 50

    params_training = {
        "framework": "textcnn",
        "differ_loss_weight": dl,
        "weighted_class": positive_class_weight,
        "learning_rate": lr,
        "adam_weight_decay": 0.005,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "data_length": 2500,
    }

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    # save path of trained model

    tuning_name = (
        f"{batch_size}-{lr}-{dl}-{dropout_probability}-{positive_class_weight}-{SEED}"
    )

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", tuning_name
    )

    if not any(
        os.path.exists(os.path.join(model_path, x)) for x in ["", "auc", "score"]
    ):
        # os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, "auc"))
        os.makedirs(os.path.join(model_path, "score"))
    save_path = os.path.join(model_path, "results.txt")

    logger = get_logger(logpath=save_path, filepath=os.path.abspath(__file__))
    logger.info(params_training)
    logger.info(f"Training samples: {len(trainset_x)}, Validation: {len(valset_x)}, Test: {len(testset_x)}")
    logger.info(f"Positive samples - Train: {trainset_y.sum().item()}, Val: {valset_y.sum().item()}, Test: {testset_y.sum().item()}")
    
    # Diagnostic code for problem batches
    problem_batches = [3, 10, 30, 32, 38, 56, 60, 65, 70, 73, 78, 81, 83, 86, 97, 98, 103, 120]
    
    for batch_idx in problem_batches:
        start_idx = (batch_idx - 1) * 32
        end_idx = batch_idx * 32
        batch_data = trainset_x[start_idx:end_idx]
        batch_labels = trainset_y[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Data - Min: {batch_data.min():.4f}, Max: {batch_data.max():.4f}")
        print(f"  Data - Mean: {batch_data.mean():.4f}, Std: {batch_data.std():.4f}")
        print(f"  NaNs: {torch.isnan(batch_data).sum()}, Infs: {torch.isinf(batch_data).sum()}")
        print(f"  Label distribution: {batch_labels.sum()}/{len(batch_labels)} positive")
    
    model_save_path = os.path.join(
        model_path, str(params_training["learning_rate"]) + ".pt"
    )

    dataset_train = Dataset_train(trainset_x, trainset_y)
    dataset_eval = Dataset_train(valset_x, valset_y)
    dataset_test = Dataset_train(testset_x, testset_y)

    params = {
        "batch_size": params_training["batch_size"],
        "shuffle": False,
        "num_workers": 0,
    }

    iterator_train = DataLoader(dataset_train, **params)
    iterator_test = DataLoader(dataset_eval, **params)
    iterator_heldout = DataLoader(dataset_test, **params)

    model = CNNClassifier(inputs=num_channels, dropout=dropout_probability)

    logger.info(model)
    logger.info(
        "Num of Parameters: {}M".format(
            sum(x.numel() for x in model.parameters()) / 1000000
        )
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params_training["learning_rate"],
        weight_decay=params_training["adam_weight_decay"],
    )  # optimize all cnn parameters
    loss_ce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([params_training["weighted_class"]]).to(device)
    )

    num_epochs = params_training["max_epoch"]

    results_trainloss = []
    results_evalloss = []
    results_score = []
    results_TPR = []
    results_TNR = []
    results_acc = []
    max_score, max_auc = 0, 0
    min_eval_loss = float("inf")

    for t in range(1, 1 + num_epochs):
        train_loss = 0
        differ_loss_val = 0
        model = model.train()
        train_TP, train_FP, train_TN, train_FN = 0, 0, 0, 0

        for b, batch in enumerate(
            iterator_train, start=1
        ):  # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
            loss, differ_loss, Y_train_prediction, y_train = train_model(
                batch,
                model,
                loss_ce,
                device,
                weight=params_training["differ_loss_weight"],
            )

            # Check for NaN/Inf in losses before proceeding
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf in loss at batch {b}, skipping...")
                continue
            if torch.isnan(differ_loss) or torch.isinf(differ_loss):
                logger.warning(f"NaN/Inf in differ_loss at batch {b}, setting to 0...")
                differ_loss = torch.tensor(0.0, device=device, requires_grad=True)

            train_loss += loss.item()
            differ_loss_val += differ_loss.item()
            loss += differ_loss
            
            # Check combined loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf in combined loss at batch {b}, skipping update...")
                continue

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()
            
            # Gradient clipping BEFORE checking for NaN (clipping can fix some NaN issues)
            # Use a smaller max_norm for more aggressive clipping
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            except RuntimeError:
                # If clipping fails, zero out gradients and skip
                optimizer.zero_grad()
                logger.warning(f"Gradient clipping failed at batch {b}, skipping update...")
                continue
            
            # Check for NaN/Inf gradients after clipping
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    # Replace NaN/Inf gradients with zeros
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param.grad = torch.where(
                            torch.isnan(param.grad) | torch.isinf(param.grad),
                            torch.zeros_like(param.grad),
                            param.grad
                        )
                        has_nan_grad = True
            
            if has_nan_grad:
                logger.warning(f"NaN/Inf gradients detected at batch {b}, zeroed out...")
            
            # Check if gradient norm is still reasonable
            if grad_norm > 10.0:
                logger.warning(f"Large gradient norm ({grad_norm:.2f}) at batch {b}, but proceeding...")

            # Update parameters
            optimizer.step()
        train_loss /= b
        differ_loss_val /= b

        eval_loss = 0
        model = model.eval()
        types_TP = 0
        types_FP = 0
        types_TN = 0
        types_FN = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for b, batch in enumerate(iterator_test, start=1):
                loss, Y_eval_prediction, y_test = eval_model(
                    batch, model, loss_ce, device
                )
                
                # Skip if loss is still NaN after eval_model fixes
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss in eval at batch {b}, skipping...")
                    continue
                
                types_TP, types_FP, types_TN, types_FN = evaluation_test(
                    Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN
                )
                eval_loss += loss.item()
                # Accumulate predictions and targets for AUC calculation
                all_predictions.append(Y_eval_prediction.cpu().detach())
                all_targets.append(y_test.cpu().detach())

        eval_loss /= b
        acc = 100 * (types_TP + types_TN) / (types_TP + types_TN + types_FP + types_FN)
        score = (
            100
            * (types_TP + types_TN)
            / (types_TP + types_TN + types_FP + 5 * types_FN)
        )
        TPR = 100 * types_TP / (types_TP + types_FN)
        TNR = 100 * types_TN / (types_TN + types_FP)

        if types_TP + types_FP == 0:
            ppv = 1
        else:
            ppv = types_TP / (types_TP + types_FP)
        
        # Concatenate all predictions and targets, handle NaNs
        all_predictions = torch.cat(all_predictions).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Remove NaN values and ensure we have both classes
        valid_mask = ~(np.isnan(all_predictions) | np.isnan(all_targets))
        if valid_mask.sum() > 0 and len(np.unique(all_targets[valid_mask])) > 1:
            auc = sklearn.metrics.roc_auc_score(
                all_targets[valid_mask], all_predictions[valid_mask]
            )
        else:
            auc = 0.0
        f1 = types_TP / (types_TP + 0.5 * (types_FP + types_FN))
        sen = types_TP / (types_TP + types_FN)
        spec = types_TN / (types_TN + types_FP)

        if auc > max_auc:
            max_auc = auc
            torch.save(
                model.state_dict(), os.path.join(model_path, "auc", str(t) + ".pt")
            )

        if score > max_score:
            max_score = score
            torch.save(
                model.state_dict(), os.path.join(model_path, "score", str(t) + ".pt")
            )

        logger.info(20 * "-")

        logger.info(params_training["framework"] + " Epoch " + str(t))

        logger.info(
            "total_loss: "
            + str(round(train_loss + differ_loss_val, 5))
            + " train_loss: "
            + str(round(train_loss, 5))
            + " differ_loss: "
            + str(round(differ_loss_val, 5))
            + " eval_loss: "
            + str(round(eval_loss, 5))
        )

        logger.info(
            "TPR: "
            + str(round(TPR, 3))
            + " TNR: "
            + str(round(TNR, 3))
            + " Score: "
            + str(round(score, 3))
            + " Acc: "
            + str(round(acc, 3))
        )

        logger.info(
            "PPV: "
            + str(round(ppv, 3))
            + " AUC: "
            + str(round(auc, 3))
            + " F1: "
            + str(round(f1, 3))
        )

        if t == num_epochs:
            # logger.info(f'total_loss: {total_losses} train_loss: {train_losses} eval_loss: {eval_losses}')

            types_TP = 0
            types_FP = 0
            types_TN = 0
            types_FN = 0
            y_pred_final_all = []
            y_test_final_all = []
            with torch.no_grad():
                for b, batch in enumerate(iterator_heldout, start=1):
                    # batch is a list of two tensors, the second tensor is y_test
                    # prediction
                    loss, y_prediction_final, y_test_final = eval_model(
                        batch, model, loss_ce, device
                    )
                    # loss, Y_eval_prediction is different every time, y_test is same everytime

                    types_TP, types_FP, types_TN, types_FN = evaluation_test(
                        y_prediction_final,
                        y_test_final,
                        types_TP,
                        types_FP,
                        types_TN,
                        types_FN,
                    )
                    y_pred_final_all.extend(y_prediction_final.tolist())
                    y_test_final_all.extend(y_test_final.tolist())

                acc = (100 * types_TP + types_TN) / (
                    types_TP + types_TN + types_FP + types_FN
                )
                score = (
                    100
                    * (types_TP + types_TN)
                    / (types_TP + types_TN + types_FP + 5 * types_FN)
                )
                TPR = 100 * types_TP / (types_TP + types_FN)
                TNR = 100 * types_TN / (types_TN + types_FP)

                if types_TP + types_FP == 0:
                    ppv = 1
                else:
                    ppv = types_TP / (types_TP + types_FP)
                
                # Handle NaN values in predictions
                y_pred_final_all = np.array(y_pred_final_all)
                y_test_final_all = np.array(y_test_final_all)
                valid_mask = ~(np.isnan(y_pred_final_all) | np.isnan(y_test_final_all))
                if valid_mask.sum() > 0 and len(np.unique(y_test_final_all[valid_mask])) > 1:
                    auc = sklearn.metrics.roc_auc_score(y_test_final_all[valid_mask], y_pred_final_all[valid_mask])
                else:
                    auc = 0.0
                f1 = types_TP / (types_TP + 0.5 * (types_FP + types_FN))
                sen = types_TP / (types_TP + types_FN)
                spec = types_TN / (types_TN + types_FP)

                logger.info("final_test")
                logger.info(
                    "TPR: "
                    + str(round(TPR, 3))
                    + " TNR: "
                    + str(round(TNR, 3))
                    + " Score: "
                    + str(round(score, 3))
                    + " Acc: "
                    + str(round(acc, 3))
                )

                logger.info(
                    "PPV: "
                    + str(round(ppv, 3))
                    + " AUC: "
                    + str(round(auc, 3))
                    + " F1: "
                    + str(round(f1, 3))
                    + " SEN: "
                    + str(round(sen, 3))
                    + " SPEC: "
                    + str(round(spec, 3))
                )
