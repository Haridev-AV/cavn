# Context-Aware Alert Validation Network (CAVN)

This repository implements a **Hybrid CAVN model** designed to distinguish between clinically meaningful alerts and false alarms in IoMT (Internet of Medical Things) monitoring systems. The project is built directly on top of the **[VTaC (Ventricular Tachycardia annotated alarms from ICUs)](https://github.com/ML-Health/VTaC/tree/main)** benchmark and utilizes its core data processing and modeling principles as a foundation. By extending VTaC's deep learning approaches, this implementation introduces a privacy-preserving federated framework that captures physiological trends and multi-sensor contextual information to improve patient safety and reduce alert fatigue.

---

## Project Background & Implementation

Traditional IoMT monitoring systems suffer from high false alarm rates, leading to "alert fatigue" where clinicians may become desensitized to important warnings. These false alerts arise due to patient movement, poor sensor contact, signal drift, or short-term physiological changes that do not require medical intervention. While deep learning models can reduce these alarms, they often require centralized data, which is difficult to obtain due to privacy regulations.

This project addresses these challenges by:
* **Building on VTaC:** Utilizing the [VTaC dataset and repository](https://github.com/ML-Health/VTaC/tree/main) for standardized ICU alarm validation and signal processing. VTaC is a benchmark dataset introduced in the NeurIPS 2023 paper by Lehman et al., containing over 5,000 annotated VT alarm recordings from three major US hospitals. The dataset is publicly available on [PhysioNet](https://physionet.org/content/vtac/1.0/).
* **Privacy-First Federated Learning:** Implementing the **Flower (flwr)** framework to train models across hospital silos without ever moving raw patient data, enabling collaborative learning while strictly maintaining patient privacy.
* **Context-Aware Validation:** Moving beyond simple waveform analysis by integrating signal quality indices (SQI), physiological trends, and multi-sensor agreement to assess the clinical relevance of alerts.

---

## Project Objectives

1. **Mitigate Alarm Fatigue:** Suppress non-actionable alerts while preserving clinically significant events through a context-aware alert validation mechanism.

2. **Model Clinical Relevance:** Capture physiological trends and multi-sensor agreement to determine whether an alert is clinically meaningful before it reaches the clinician.

3. **Empirical Robustness Evaluation:** Test the system's effectiveness under realistic, noisy IoMT sensing conditions using publicly available clinical datasets.

4. **Privacy-Preserving Collaboration:** Enable collaborative learning across healthcare institutions without centralized data sharing through federated learning.

---

## Project Architecture
The system is built on a **Late-Fusion Hybrid Architecture** that combines deep learning with clinical domain knowledge.

### 1. Model Components
* **CNN Branch (Backbone):** A multi-scale 1D Convolutional Neural Network that extracts local spatial and temporal features directly from raw physiological signals (ECG/PPG) and device alerts.
* **Context Branch (Statistical):** A parallel network that processes hand-engineered contextual features to model clinical relevance:
    * **Kurtosis & Skewness:** Used to identify signal peakedness and asymmetry to detect noise or artifacts.
    * **Zero Crossing Rate (ZCR):** Captures frequency-domain characteristics for noise identification and signal quality assessment.
    * **Heart Rate Proxy:** Provides clinical heart rate context to verify multi-sensor agreement (e.g., comparing ECG heart rate to PPG pulse rate).
    * **Physiological Trends:** Captures temporal patterns to differentiate short-term changes from clinically significant events.
* **Fusion Layer:** Merges the deep features and statistical features into a final classification head with confidence scoring mechanism.
* **Confidence Scoring:** Post-inference filter that ensures only high-confidence, clinically actionable alerts reach the clinician, directly addressing alert fatigue.

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

## Expected Outcomes

* **Reduces Alert Fatigue:** By filtering false and non-actionable alerts, clinicians receive fewer but more meaningful notifications, improving their ability to respond effectively to real emergencies.

* **Improves Patient Safety:** Important alerts are preserved with high sensitivity, increasing the chance of timely intervention for life-threatening events.

* **Preserves Privacy:** Federated learning enables collaborative model training across institutions without sharing sensitive patient data—raw patient data never leaves the local node.

* **Scalable Across Hospitals:** The system can be deployed across multiple healthcare institutions and improved continuously through federated updates.

* **Real-World Applicability:** Evaluated under realistic, noisy IoMT sensing conditions including signal drift, sensor detachment, and patient movement artifacts.

---

## Evaluation Metrics

To ensure clinical relevance and system robustness, the system is evaluated using multiple complementary metrics that address both safety and operational efficiency:

**Primary Metric: VTaC Clinical Utility Score**
The model is evaluated using the **VTaC Clinical Utility Score**, which penalizes missed life-threatening events five times more heavily than false alarms, ensuring patient safety remains the top priority.

### Clinical Performance Indicators

### 1. Alarm Burden Reduction (False Alarm Rate)

The percentage of false alarms successfully suppressed by the model. This corresponds to the **Specificity (True Negative Rate)** and directly measures the reduction in alert fatigue. A high value indicates clinicians are not overwhelmed by non-actionable warnings, allowing them to focus on genuine emergencies.

### 2. Sensitivity on High-Priority VT Alarms (Recall)

The ability of the system to correctly identify true, life-threatening Ventricular Tachycardia events. This corresponds to the **Recall (True Positive Rate)** and is critical for patient safety. The system prioritizes recall over aggressive alert suppression to ensure no genuine cardiac arrests are missed during validation.

### 3. Robustness Under Noisy Sensor Conditions

The model's resilience against real-world artifacts including signal drift, sensor detachment, patient movement, and poor sensor contact. This metric evaluates performance degradation when tested against noisy, realistic IoMT sensing conditions, ensuring the system maintains accuracy in clinical deployment scenarios.

### 4. Time-to-Detection Impact

The computational latency introduced by the inference and confidence scoring process. This metric measures the average time (in milliseconds) required to process a 10-second window of vital signs, ensuring the model is computationally efficient enough for real-time deployment on bedside monitors without introducing dangerous delays to clinical intervention.

---

## Project Team

**Group Number:** B14

**Team Members:**
- Haridev A. V. (AM.EN.U4CSE22126)
- Navjyoth Pradeep (AM.EN.U4CSE22139)
- Prisha Singh (AM.EN.U4CSE22143)
- Rohit Kamal V (AM.EN.U4CSE22145)

**Project Guide:** Dr. Lekshmi S. Nair

**Institution:** Department of Computer Science, Amrita School of Computing, Amritapuri Campus  
**Program:** B.Tech CSE (2022 Admission)  
**Course:** 19CSE499 PROJECT PHASE-II

---

## Acknowledgments

This project is built on top of the [VTaC repository](https://github.com/ML-Health/VTaC/tree/main) and extends its core methodologies with federated learning and hybrid architectures.

**VTaC Citation:**

> Li-wei Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, Roger Mark. *VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors.* Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track. [DOI: 10.13026/z4f3-1f07](https://doi.org/10.13026/z4f3-1f07)