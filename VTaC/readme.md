# Clinical Alarm Validation Network (CAVN)

This repository implements a **Hybrid CAVN model** designed to distinguish between true life-threatening arrhythmias and false clinical alarms. The project is built directly on top of the **[VTaC (Ventricular Tachycardia annotated alarms from ICUs)](https://github.com/ML-Health/VTaC/tree/main)** benchmark and utilizes its core data processing and modeling principles as a foundation. By extending VTaC's deep learning approaches, this implementation introduces a privacy-preserving federated framework and a context-aware architecture to improve patient safety.

---

## Project Background & Implementation
Traditional ICU monitoring systems suffer from high false alarm rates, leading to "alert fatigue" where clinicians may become desensitized to important warnings. While deep learning models can reduce these alarms, they often require centralized data, which is difficult to obtain due to privacy regulations. 

This project addresses these challenges by:
* **Building on VTaC:** Utilizing the [VTaC dataset and repository](https://github.com/ML-Health/VTaC/tree/main) for standardized ICU alarm validation and signal processing. VTaC is a benchmark dataset introduced in the NeurIPS 2023 paper by Lehman et al., containing over 5,000 annotated VT alarm recordings from three major US hospitals. The dataset is publicly available on [PhysioNet](https://physionet.org/content/vtac/1.0/).
* **Privacy-First Federated Learning:** Implementing the **Flower (flwr)** framework to train models across "hospital silos" without ever moving raw patient signals.
* **Context-Aware Validation:** Moving beyond simple waveform analysis by integrating signal quality indices (SQI) and cross-sensor agreement.

---

## Project Architecture
The system is built on a **Late-Fusion Hybrid Architecture** that combines deep learning with clinical domain knowledge.

### 1. Model Components
* **CNN Branch (Backbone):** A multi-scale 1D Convolutional Neural Network that extracts local spatial and temporal features directly from raw physiological signals (ECG/PPG).
* **Context Branch (Statistical):** A parallel network that processes hand-engineered statistical features:
    * **Kurtosis & Skewness:** Used to identify signal peakedness and asymmetry to detect noise or artifacts.
    * **Zero Crossing Rate (ZCR):** Captures frequency-domain characteristics for noise identification.
    * **Heart Rate Proxy:** Provides clinical heart rate context to verify cross-sensor agreement (e.g., comparing ECG heart rate to PPG pulse rate).
* **Fusion Layer:** Merges the deep features and statistical features into a final classification head for the final decision.

### 2. Learning Frameworks
* **Federated Learning:** Built using the Flower (flwr) framework. It simulates a server-client architecture where hospitals (clients) compute weight updates locally and a central server aggregates them using the **FedAvg** strategy over 10 rounds.
* **Centralized Learning:** A standalone benchmark script used to establish the performance "Upper Bound" by aggregating all data into a single training pool.

---

## Execution Guide

### Federated Learning Simulation
To run the full simulation with 3 hospitals and a central server:

**To run the Hybrid CAVN (Proposed):**
```powershell
python run_fl_simulation.py --mode hybrid
```

**To run the Baseline CNN (Comparison):**
```powershell
python run_fl_simulation.py --mode cnn
```

---

## Repository Structure
```
models/hybrid/: Contains the HybridCAVN architecture.
models/cnn/: Contains the baseline CNN architecture and VTaC data utilities.
fl_server.py: Implementation of the Flower Server and FedAvg strategy.
fl_client.py: Federated Learning client logic and local training loops.
run_fl_simulation.py: Orchestration script to launch the server and multiple clients.
run_centralized_hybrid.py: Benchmark script for non-federated training.
```

---

## Evaluation Metrics
The system is evaluated using the VTaC Clinical Utility Score, which balances safety and alarm reduction:

$$Score = 100 \times \frac{TP + TN}{TP + TN + FP + 5 \times FN}$$

* **TPR (Sensitivity):** Percentage of true emergencies caught.
* **TNR (Specificity):** Percentage of false alarms suppressed.
* **AUC-ROC:** Measures overall diagnostic potential across all thresholds.

---

## Acknowledgments
This project is built on top of the [VTaC repository](https://github.com/ML-Health/VTaC/tree/main) and extends its core methodologies with federated learning and hybrid architectures.

**VTaC Citation:**
> Li-wei Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, Roger Mark. *VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors.* Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track. [DOI: 10.13026/z4f3-1f07](https://doi.org/10.13026/z4f3-1f07)
