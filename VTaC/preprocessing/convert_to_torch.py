"""
Convert numpy-format lead_selected files to torch format.
Run this script after fixing torch installation issues.
"""
import os
import numpy as np
import torch

OUTPUT_DIR = "data/out/lead_selected"

def convert_split(split_name):
    """Convert a numpy-format file to torch format"""
    np_path = os.path.join(OUTPUT_DIR, f"{split_name}_numpy.npz")
    torch_path = os.path.join(OUTPUT_DIR, f"{split_name}.pt")
    
    if not os.path.exists(np_path):
        print(f"File not found: {np_path}")
        return False
    
    print(f"Converting {split_name}...")
    data = np.load(np_path, allow_pickle=True)
    
    samples = torch.from_numpy(data['samples']).float()
    ys = torch.from_numpy(data['ys']).float()
    names = data['names'].tolist()
    
    torch.save((samples, ys, names), torch_path)
    print(f"[OK] Converted to {torch_path}")
    print(f"  Shape: {samples.shape}")
    print(f"  Labels: {ys.sum().item()} True, {len(ys) - ys.sum().item()} False")
    
    # Optionally remove numpy file
    # os.remove(np_path)
    
    return True

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)
    print("\n[OK] All files converted!")

