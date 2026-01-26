"""
Federated Learning Simulation for VTaC
Runs the complete federated learning process with multiple hospital clients
"""
import subprocess
import sys
import os
import time
import threading


def run_server():
    """Run the FL server in a separate thread"""
    print("Starting FL Server...")
    cmd = [sys.executable, "fl_server.py"]
    return subprocess.Popen(cmd, cwd=os.getcwd())


def run_client(hospital_id):
    """Run a client for a specific hospital"""
    print(f"Starting Client for Hospital {hospital_id}...")
    cmd = [sys.executable, "run_client.py", str(hospital_id)]
    return subprocess.Popen(cmd, cwd=os.getcwd())


def main():
    """Main simulation function"""
    print("=" * 80)
    print("VTaC Federated Learning Simulation")
    print("=" * 80)
    print("This will simulate 3 hospitals participating in federated learning")
    print("Each hospital trains on its local data and shares model updates")
    print("=" * 80)

    # Check if data exists
    for i in range(1, 4):
        data_dir = f"data/hospitals/hospital_{i:02d}"
        if not os.path.exists(data_dir):
            print(f"Error: Hospital data not found at {data_dir}")
            print("Please run 'python shard_data.py' first to create hospital data.")
            return

    print("Starting simulation...")

    # Start server
    server_process = run_server()
    time.sleep(2)  # Give server time to start

    # Start clients
    client_processes = []
    for i in range(1, 4):
        client_process = run_client(i)
        client_processes.append(client_process)
        time.sleep(1)  # Stagger client starts

    # Wait for all processes to complete
    print("Waiting for federated learning to complete...")
    for process in client_processes:
        process.wait()

    # Stop server
    server_process.terminate()
    server_process.wait()

    print("=" * 80)
    print("Federated Learning Simulation Complete!")
    print("=" * 80)
    print("The global model has been trained across all hospitals.")
    print("Check the server output for final aggregated metrics.")
    print("=" * 80)


if __name__ == "__main__":
    main()