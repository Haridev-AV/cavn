import torch
import time
import numpy as np
from models.cnn.realtime.nets import CNNClassifier
from models.hybrid.realtime.nets import HybridCAVN  # Ensure this import matches your file structure

def benchmark_inference(model, device, is_hybrid=False, num_runs=1000):
    """
    Measures the average inference time of the model.
    
    Args:
        model: The PyTorch model to test.
        device: 'cpu' or 'cuda'.
        is_hybrid: Boolean, set True if model needs context input.
        num_runs: Number of forward passes to average.
    """
    model.eval()
    model.to(device)

    # 1. Create Dummy Input (1 Batch, 2 Channels, 2500 Samples for 10s @ 250Hz)
    dummy_wave = torch.randn(1, 2, 2500).float().to(device)
    dummy_context = torch.randn(1, 4).float().to(device) if is_hybrid else None

    # 2. Warm-up Phase (Critical for GPU)
    # The first few passes are always slower due to memory allocation/caching
    print(f"Warming up ({device})...")
    with torch.no_grad():
        for _ in range(50):
            if is_hybrid:
                _ = model(dummy_wave, dummy_context)
            else:
                _ = model(dummy_wave)
    
    # Synchronize if on GPU to finish all async tasks before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 3. Timing Phase
    timings = []
    print(f"Benchmarking over {num_runs} runs...")
    
    with torch.no_grad():
        for _ in range(num_runs):
            # Start timer
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                
                # Inference
                if is_hybrid:
                    _ = model(dummy_wave, dummy_context)
                else:
                    _ = model(dummy_wave)
                
                end.record()
                torch.cuda.synchronize()
                curr_time = start.elapsed_time(end) # Returns ms directly
                timings.append(curr_time)
            else:
                # CPU Timing
                start = time.perf_counter()
                
                # Inference
                if is_hybrid:
                    _ = model(dummy_wave, dummy_context)
                else:
                    _ = model(dummy_wave)
                    
                end = time.perf_counter()
                timings.append((end - start) * 1000) # Convert s to ms

    # 4. Calculate Statistics
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    
    return avg_latency, std_latency

if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CHANNELS = 2  # ECG + PPG
    
    print("="*60)
    print("VTaC SYSTEM LATENCY BENCHMARK")
    print("="*60)

    # --- Test 1: Baseline CNN ---
    print("\n[TEST 1] Baseline CNN Latency")
    cnn_model = CNNClassifier(inputs=NUM_CHANNELS, dropout=0.3)
    avg_cnn, std_cnn = benchmark_inference(cnn_model, DEVICE, is_hybrid=False)
    print(f"Result: {avg_cnn:.4f} ms ± {std_cnn:.4f} ms")

    # --- Test 2: Hybrid CAVN ---
    print("\n[TEST 2] Hybrid CAVN Latency")
    try:
        # Create CNN backbone first, then wrap it in Hybrid
        cnn_backbone = CNNClassifier(inputs=NUM_CHANNELS, dropout=0.3)
        hybrid_model = HybridCAVN(cnn_backbone, num_context_features=4)
        avg_hyb, std_hyb = benchmark_inference(hybrid_model, DEVICE, is_hybrid=True)
        print(f"Result: {avg_hyb:.4f} ms ± {std_hyb:.4f} ms")
        
        # --- Comparison ---
        impact = avg_hyb - avg_cnn
        print("-" * 30)
        print(f"Time-to-Detection Impact: +{impact:.4f} ms")
        print("Verdict: " + ("NEGLIGIBLE" if impact < 10 else "SIGNIFICANT"))
        print("-" * 30)

    except ImportError:
        print("HybridCAVN class not found. Make sure models/hybrid/realtime/nets.py exists.")
    except Exception as e:
        print(f"Error testing Hybrid model: {e}")