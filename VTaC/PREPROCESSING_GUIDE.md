# Preprocessing and Baseline Verification Guide

This guide helps you complete the last two steps of Phase 1:
1. Run Filtering & Standardization
2. Verify the Baseline

## Prerequisites

Before running these scripts, ensure you have:

1. **Installed dependencies**: 
   ```bash
   pip install torch wfdb scipy rich matplotlib sklearn
   ```
   Note: If you're using Python 3.11, torch 1.9.0 may not be available. Use a compatible version (torch 2.x should work).

2. **Prepared input data**: The filtering script expects the following files:
   - `data/out/lead_selected/train.pt`
   - `data/out/lead_selected/val.pt`
   - `data/out/lead_selected/test.pt`
   
   These files should be created from the PhysioNet VTaC dataset in a previous preprocessing step.

## Step 1: Run Filtering

The filtering script applies signal filters to ECG, PPG, and ABP waveforms.

Run for each split:
```bash
python preprocessing/filtering.py train
python preprocessing/filtering.py val
python preprocessing/filtering.py test
```

Or use the automated script:
```bash
python run_preprocessing.py
```

This will create:
- `data/out/train-filtered.pt`
- `data/out/val-filtered.pt`
- `data/out/test-filtered.pt`

## Step 2: Run Standardization

The standardization script normalizes the filtered data. You can choose between:
- **Population normalization** (option 1): Uses mean/std from the training set
- **Per-sample normalization** (option 2): Normalizes each sample individually

Run:
```bash
python preprocessing/standardize.py
```

When prompted, enter:
- `1` for population normalization → outputs to `data/out/population-norm/`
- `2` for per-sample normalization → outputs to `data/out/sample-norm/`

**Note**: The training script expects per-sample normalized data in `data/out/sample-norm/`, so choose option 2.

Or use the automated script (defaults to per-sample normalization):
```bash
python run_preprocessing.py
```

## Step 3: Verify Baseline

Run the CNN realtime training script to verify baseline performance:

```bash
python models/cnn/realtime/train.py <batch_size> <learning_rate> <differ_loss_weight> <dropout> <positive_class_weight> [seed]
```

Example:
```bash
python models/cnn/realtime/train.py 64 0.0001 0.1 0.5 4.0 1
```

Parameters:
- `batch_size`: Batch size (e.g., 64)
- `learning_rate`: Learning rate (e.g., 0.0001)
- `differ_loss_weight`: Difference loss weight (e.g., 0.1)
- `dropout`: Dropout probability (e.g., 0.5)
- `positive_class_weight`: Positive class weight (e.g., 4.0)
- `seed`: Random seed (optional, default: 1)

The script will:
- Load data from `data/out/sample-norm/`
- Train for up to 500 epochs
- Save models to `models/cnn/realtime/models/<tuning_name>/`
- Log results to `models/cnn/realtime/models/<tuning_name>/results.txt`

## Troubleshooting

### Missing input files
If you get an error about missing `data/out/lead_selected/*.pt` files:
- Ensure you've downloaded the VTaC dataset from PhysioNet
- Run the initial data preparation steps to create the lead_selected files

### Torch installation issues
- For Python 3.11, use: `pip install torch` (will install latest compatible version)
- For older Python versions, you may need: `pip install torch==1.9.0`

### Path issues
The `train.py` script has been fixed to use the correct path (`data/out/sample-norm` instead of `../dataset/sample-norm`).

