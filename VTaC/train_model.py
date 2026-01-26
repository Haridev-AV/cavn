"""
Simple training script for CNN realtime model
Usage: python train_model.py
"""
import subprocess
import sys
import os

# Change to the train.py directory
script_dir = os.path.dirname(os.path.abspath(__file__))
train_script = os.path.join(script_dir, "models", "cnn", "realtime", "train.py")

# Parameters: batch_size, lr, dl, dropout, pos_weight, seed
# Note: max_epoch is hardcoded in train.py as 500
params = ["32", "0.001", "0.0", "0.3", "2.0", "10"]

print("=" * 60)
print("Training CNN Realtime Model")
print("=" * 60)
print(f"Parameters: batch_size={params[0]}, lr={params[1]}, dl={params[2]}")
print(f"           dropout={params[3]}, pos_weight={params[4]}, seed={params[5]}")
print(f"           max_epochs=500 (hardcoded in train.py)")
print("=" * 60)
print()

# Run the training script
cmd = [sys.executable, train_script] + params
subprocess.run(cmd)