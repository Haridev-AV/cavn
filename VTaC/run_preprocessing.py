"""
Script to run filtering and standardization steps for VTaC dataset.
This script automates the preprocessing pipeline.
"""
import subprocess
import sys
import os

def run_filtering():
    """Run filtering.py for train, val, and test splits"""
    print("=" * 60)
    print("Step 1: Running filtering.py for all splits")
    print("=" * 60)
    
    splits = ["train", "val", "test"]
    for split in splits:
        print(f"\nFiltering {split} split...")
        try:
            result = subprocess.run(
                [sys.executable, "preprocessing/filtering.py", split],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Successfully filtered {split} split")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error filtering {split} split:")
            print(e.stderr)
            return False
        except FileNotFoundError:
            print(f"✗ Error: Could not find data/out/lead_selected/{split}.pt")
            print("  Please ensure you have run the initial data preparation steps.")
            return False
    
    return True

def run_standardization(choice="2"):
    """
    Run standardize.py with the specified normalization choice.
    choice: "1" for population normalization, "2" for per-sample normalization
    """
    print("\n" + "=" * 60)
    print("Step 2: Running standardize.py")
    print("=" * 60)
    
    print(f"\nUsing {'population' if choice == '1' else 'per-sample'} normalization...")
    try:
        # Run standardize.py and provide input
        process = subprocess.Popen(
            [sys.executable, "preprocessing/standardize.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=choice + "\n", timeout=3600)
        
        if process.returncode == 0:
            print("✓ Successfully standardized data")
            if stdout:
                print(stdout)
            return True
        else:
            print("✗ Error during standardization:")
            print(stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Standardization timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("VTaC Preprocessing Pipeline")
    print("=" * 60)
    
    # Check if input files exist
    required_files = [
        "data/out/lead_selected/train.pt",
        "data/out/lead_selected/val.pt",
        "data/out/lead_selected/test.pt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n⚠ Warning: Missing required input files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you have:")
        print("  1. Downloaded the VTaC dataset from PhysioNet")
        print("  2. Run the initial data preparation to create lead_selected files")
        print("\nProceeding anyway...\n")
    
    # Run filtering
    if not run_filtering():
        print("\n❌ Filtering failed. Please check the errors above.")
        sys.exit(1)
    
    # Run standardization (default to per-sample normalization)
    # Change to "1" for population normalization
    if not run_standardization(choice="2"):
        print("\n❌ Standardization failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run models/cnn/realtime/train.py to verify baseline performance")
    print("  - Example: python models/cnn/realtime/train.py 64 0.0001 0.1 0.5 4.0")

