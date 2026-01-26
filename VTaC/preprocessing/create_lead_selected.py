"""
Creates lead_selected files from PhysioNet VTaC dataset.

This script processes the raw waveform files and creates the input files
needed for filtering.py. It extracts:
- 2 ECG leads (II and V, or best available)
- PPG/PLETH signal
- ABP signal (if available, otherwise zeros)

Output format: (samples, ys, names) where:
- samples: torch.Tensor of shape [N, 4, signal_length]
- ys: torch.Tensor of shape [N] with labels (1 for True, 0 for False)
- names: list of event names
"""
import os
import sys
import pandas as pd
import numpy as np
# Delay torch import until we actually need it (at save time)
# This avoids DLL loading issues during processing
import wfdb
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

# Paths
PHYSIONET_BASE = "physionet.org/files/vtac/1.0"
WAVEFORMS_DIR = os.path.join(PHYSIONET_BASE, "waveforms")
SPLIT_CSV = os.path.join(PHYSIONET_BASE, "benchmark_data_split.csv")
LABELS_CSV = os.path.join(PHYSIONET_BASE, "event_labels.csv")
OUTPUT_DIR = "data/out/lead_selected"

SAMPLING_FREQ = 250  # Hz
# Signal length may vary, but we'll use 90000 as standard (6 minutes at 250 Hz)
# Some files may be shorter (72000 = 4.8 minutes) or longer
SIGNAL_LENGTH = 90000  # Target length - will pad or truncate as needed


def find_signal_indices(sig_names):
    """
    Find indices for ECG leads, PLETH, and ABP in the signal names.
    Returns: (ecg_idx1, ecg_idx2, pleth_idx, abp_idx)
    """
    ecg_idx1 = None
    ecg_idx2 = None
    pleth_idx = None
    abp_idx = None
    
    # Preferred ECG leads (in order of preference)
    ecg_preferences = ['II', 'V', 'I', 'aVR', 'aVL', 'aVF']
    
    # Find ECG leads
    for pref in ecg_preferences:
        for i, name in enumerate(sig_names):
            if name.upper() == pref and ecg_idx1 is None:
                ecg_idx1 = i
                break
        if ecg_idx1 is not None:
            break
    
    # Find second ECG lead
    for pref in ecg_preferences:
        for i, name in enumerate(sig_names):
            if name.upper() == pref and i != ecg_idx1 and ecg_idx2 is None:
                ecg_idx2 = i
                break
        if ecg_idx2 is not None:
            break
    
    # Find PLETH/PPG
    for i, name in enumerate(sig_names):
        if 'PLETH' in name.upper() or 'PPG' in name.upper():
            pleth_idx = i
            break
    
    # Find ABP
    for i, name in enumerate(sig_names):
        if 'ABP' in name.upper() or 'ART' in name.upper():
            abp_idx = i
            break
    
    return ecg_idx1, ecg_idx2, pleth_idx, abp_idx


def load_waveform(record_path, event_name):
    """
    Load waveform data for a specific event.
    Returns: (ecg1, ecg2, pleth, abp) as numpy arrays, or None if failed
    """
    waveform_path = os.path.join(WAVEFORMS_DIR, record_path, event_name)
    
    try:
        # Read the waveform file
        record = wfdb.rdrecord(waveform_path)
        
        # Get signal names and data
        sig_names = [sig_name.strip() for sig_name in record.sig_name]
        signals = record.p_signal  # Shape: [n_samples, n_signals]
        
        # Find indices for each signal type
        ecg_idx1, ecg_idx2, pleth_idx, abp_idx = find_signal_indices(sig_names)
        
        # Extract signals (handle missing signals)
        actual_length = signals.shape[0]
        ecg1 = signals[:, ecg_idx1].copy() if ecg_idx1 is not None else np.zeros(actual_length)
        ecg2 = signals[:, ecg_idx2].copy() if ecg_idx2 is not None else np.zeros(actual_length)
        pleth = signals[:, pleth_idx].copy() if pleth_idx is not None else np.zeros(actual_length)
        abp = signals[:, abp_idx].copy() if abp_idx is not None else np.zeros(actual_length)
        
        # Ensure correct length (pad or truncate if needed)
        def ensure_length(sig, target_len):
            if len(sig) > target_len:
                return sig[:target_len]
            elif len(sig) < target_len:
                padded = np.zeros(target_len)
                padded[:len(sig)] = sig
                return padded
            return sig
        
        ecg1 = ensure_length(ecg1, SIGNAL_LENGTH)
        ecg2 = ensure_length(ecg2, SIGNAL_LENGTH)
        pleth = ensure_length(pleth, SIGNAL_LENGTH)
        abp = ensure_length(abp, SIGNAL_LENGTH)
        
        return ecg1, ecg2, pleth, abp
        
    except Exception as e:
        print(f"Error loading {waveform_path}: {e}")
        return None


def create_lead_selected_file(split_name):
    """
    Create lead_selected file for a specific split (train/val/test).
    """
    print(f"\nProcessing {split_name} split...")
    
    # Load split assignments
    splits_df = pd.read_csv(SPLIT_CSV)
    split_df = splits_df[splits_df['split'] == split_name]
    
    # Load labels
    labels_df = pd.read_csv(LABELS_CSV)
    labels_dict = dict(zip(labels_df['event'], labels_df['decision']))
    
    # Initialize lists
    samples_list = []
    ys_list = []
    names_list = []
    
    # Process each event
    failed_count = 0
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
        record = row['record']
        event = row['event']
        
        # Get label (True = 1, False = 0)
        label = labels_dict.get(event, False)
        y = 1 if label else 0
        
        # Load waveform
        result = load_waveform(record, event)
        if result is None:
            failed_count += 1
            continue
        
        ecg1, ecg2, pleth, abp = result
        
        # Stack into [4, signal_length] format
        sample = np.stack([ecg1, ecg2, pleth, abp], axis=0)
        
        samples_list.append(sample)
        ys_list.append(y)
        names_list.append(event)
    
    if failed_count > 0:
        print(f"Warning: Failed to load {failed_count} events")
    
    # Convert to numpy arrays first
    samples_array = np.array(samples_list, dtype=np.float32)
    ys_array = np.array(ys_list, dtype=np.float32)
    
    # Try to save with torch, fallback to numpy if torch fails
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.pt")
    
    # Try to import and use torch
    torch_available = False
    try:
        import torch
        torch_available = True
    except (OSError, ImportError, Exception) as e:
        print(f"Warning: Could not import torch ({type(e).__name__}: {e})")
        print("Will save as numpy arrays. Run convert_to_torch.py after fixing torch.")
    
    if torch_available:
        try:
            # Convert to tensors
            samples = torch.from_numpy(samples_array).float()
            ys = torch.tensor(ys_list, dtype=torch.float32)
            
            # Save with torch
            torch.save((samples, ys, names_list), output_path)
            print(f"[OK] Saved {len(samples_list)} samples to {output_path} (torch format)")
            print(f"  Shape: {samples.shape}")
            print(f"  Labels: {ys.sum().item()} True, {len(ys) - ys_array.sum()} False")
            return True
        except Exception as e:
            print(f"Warning: Error saving with torch ({e})")
            print("Falling back to numpy format...")
    
    # Fallback: save as numpy arrays
    np_output_path = output_path.replace('.pt', '_numpy.npz')
    np.savez(np_output_path, samples=samples_array, ys=ys_array, names=names_list)
    print(f"[OK] Saved {len(samples_list)} samples to {np_output_path} (numpy format)")
    print(f"  Shape: {samples_array.shape}")
    print(f"  Labels: {ys_array.sum()} True, {len(ys_array) - ys_array.sum()} False")
    print(f"  Run: python preprocessing/convert_to_torch.py to convert after fixing torch")
    return False
    
    print(f"✓ Saved {len(samples_list)} samples to {output_path}")
    print(f"  Shape: {samples.shape}")
    print(f"  Labels: {ys.sum().item()} True, {len(ys) - ys.sum().item()} False")


if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        create_lead_selected_file(split)
    
    print("\n[OK] All lead_selected files created successfully!")
    print(f"Output directory: {OUTPUT_DIR}")

