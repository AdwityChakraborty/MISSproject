# MISSproject
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/AdwityChakraborty/MISSproject)

This repository contains a comprehensive framework for building, evaluating, and optimizing Network Intrusion Detection Systems (IDS). The project focuses on cross-domain performance, concept drift adaptation, and multi-objective optimization, with a special emphasis on IoT environments.

## Overview

The framework provides a complete end-to-end pipeline for network anomaly detection research, starting from raw data ingestion to advanced model optimization and explainability. It leverages multiple public datasets (KDD'99, NSL-KDD, UNSW-NB15, ToN-IoT) and transforms them into a unified feature space to enable robust cross-dataset evaluation.

## Key Features

*   **Multi-Dataset Preprocessing:** Ingests and cleans four major IDS datasets, handling inconsistencies, duplicates, and faulty column names.
*   **Harmonized Feature Space:** Creates a unified meta-schema of 32 common features, enabling models to be trained on one dataset and tested on another.
*   **IoT-Specific Feature Engineering:** Implements a pipeline to generate five novel IoT-specific features related to device behavior, such as periodicity, burstiness, and service sparsity.
*   **Multi-View Representation Learning:** Transforms tabular data into sequence and graph-based representations to capture temporal and relational patterns using sliding windows and Node2Vec.
*   **Multi-Objective Optimization (NIO Framework):** Utilizes NSGA-II, Particle Swarm Optimization (PSO), and a novel hybrid approach to simultaneously optimize model hyperparameters, feature subsets, and alert thresholds for competing objectives like high PR-AUC, high Recall, low False Positive Rate (FPR), and low latency.
*   **Comprehensive Benchmarking:** Evaluates a wide range of models across different data views:
    *   **Tabular:** LightGBM, XGBoost, MLP, Isolation Forest.
    *   **Deep Anomaly:** Autoencoder, VAE, Deep SVDD.
    *   **Sequence:** 1D-CNN.
    *   **Graph:** Node2Vec with LightGBM/MLP classifiers.
*   **Concept Drift Management:** Detects data and concept drift using ADWIN and KS-Tests, and implements adaptation strategies like threshold recalibration, ensemble reweighting, and incremental retraining.
*   **Explainability (XAI):** Integrates SHAP and LIME to explain model predictions, providing both global and local feature importance.
*   **Performance Evaluation:** Includes benchmarks for model latency and memory usage to assess feasibility for deployment on constrained IoT gateways.

## Project Workflow

The project is structured as a series of sequential scripts. Running them in the specified order reproduces the entire experimental pipeline.

1.  **Initial Setup and Inspection**
    *   `dataset_set.py`: Configure paths and perform an initial inspection of raw dataset files.
    *   `Dataset Inspection + Column Fixing + Sanity Check (3 datasets).py`: Corrects known issues in the raw CSV files, like broken column headers.

2.  **Phase 1: Data Preprocessing**
    *   `Preprocessing.py`: Cleans, normalizes, and splits the four datasets into processed `.parquet` files.
    *   `SMOTE-based class imbalance handling stage...py`: Applies SMOTE to the training splits to address class imbalance.

3.  **Phase 2: Feature Engineering & Representation Learning**
    *   `IoT Feature Engineering Pipeline + dataset merging.py`: Computes IoT-specific features for UNSW-NB15 and ToN-IoT and merges them.
    *   `Single Unified Feature Space (Harmonized Meta-Schema).py`: Maps all datasets to the common 32-feature schema, creating the `H_*.parquet` files for cross-domain experiments.
    *   `Temporal transformation of tabular intrusion detection data into sequence data.py`: Creates sequence data using a sliding window approach, saving `seq_*.npy` files.
    *   `Graph Representation Learning pipeline for intrusion detection data.py`: Creates graph embeddings using Node2Vec, saving `graph_*.npy` files.
    *   `Dataset Pipeline Integrity Checker.py`: Verifies that all necessary intermediate data files have been created.

4.  **Phase 3: Baseline Model Benchmarking**
    *   `Cross-Domain IDS benchmarking Framework.py`: Benchmarks tabular models (LGBM, XGBoost, MLP, iForest) on the harmonized data.
    *   `Deep Anomaly Detection Benchmark for Multi-Dataset Network IDS.py`: Benchmarks unsupervised deep learning models (AE, VAE, Deep SVDD).
    *   `1D-CNN-Based Sequence Intrusion Detection Benchmark.py`: Benchmarks a 1D-CNN on the sequence data.
    *   `Graph-Based Intrusion Detection Benchmark (Node2Vec + ML Classifier.py`: Benchmarks classifiers on the graph embeddings.

5.  **Phase 4: Multi-Objective Optimization (NIO)**
    *   `Multi-objective Optimization using NSGA-II (NIO Framework).py`: Runs the core NSGA-II algorithm to find a Pareto front of optimal model configurations.
    *   `PSO-Based Threshold and Ensemble Calibration Module.py`: Uses PSO to refine the alert thresholds of the best solutions found by NSGA-II.
    *   `Hybrid Optimization Framework (NSGA-II + PSO).py`: Executes a hybrid algorithm combining NSGA-II's global search with PSO's local refinement.
    *   `Generating Pareto Front Plots.py`: Creates visualizations of the optimization results, including Pareto fronts and convergence curves.

6.  **Phase 5: Drift Analysis and Adaptation**
    *   `Drift Detection (ADWIN + KS).py`: Analyzes temporal datasets for concept drift, evaluates its impact on model performance, and tests adaptation mechanisms. Also includes ablation studies.

7.  **Phase 6: Explainability and Final Benchmarks**
    *   `Explainability & Benchmarks.py`: Generates SHAP and LIME explanations for the best-optimized model and runs final latency/memory benchmarks.

## Core Components

### Data Processing and Harmonization

Four datasets (KDD'99, NSL-KDD, UNSW-NB15, ToN-IoT) are processed through a multi-stage pipeline. The key output is a set of harmonized data files (`H_*.parquet`) where each dataset is represented by a shared set of 32 features. This allows for training a model on one network environment (e.g., KDD'99) and evaluating its generalization ability on another (e.g., NSL-KDD or UNSW-NB15). The pipeline also generates IoT-specific features based on traffic periodicity, burstiness, and communication patterns (fan-in/fan-out).

### Multi-View Benchmarking Framework

The project benchmarks models across three distinct "views" of the network data:
*   **Tabular View:** Standard ML models applied to the flat feature vectors.
*   **Sequence View:** A 1D-CNN model trained on data transformed into sequences of consecutive flows, capturing temporal dependencies.
*   **Graph View:** A graph is constructed with IPs and services as nodes. Node2Vec embeddings are generated and fed into ML classifiers to leverage relational information.

### NIO: Neuro-evolutionary Optimization Framework

This is the core optimization engine of the project. It uses multi-objective evolutionary algorithms to find models that balance competing goals. The framework optimizes:
*   **Model Choice:** Selects between LightGBM and XGBoost.
*   **Hyperparameters:** `n_estimators`, `max_depth`, `learning_rate`.
*   **Feature Subset:** Selects from groups of features to find the most informative subset.
*   **Alert Threshold:** Calibrates the decision threshold for flagging anomalies.

The objectives are to maximize PR-AUC and Recall while minimizing False Positive Rate (FPR) and inference latency.

### Concept Drift and Adaptation

The framework includes modules to simulate and mitigate concept drift, a common problem where network traffic patterns change over time. Using `ADWIN` and the Kolmogorov-Smirnov test, the system detects drift in model output scores and feature distributions. It then evaluates three adaptation strategies:
1.  **Threshold Recalibration:** Adjusting the alert threshold on new data.
2.  **Ensemble Reweighting:** Blending an old model with a new one trained on recent data.
3.  **Incremental Retraining:** Fine-tuning the existing model with new data.

The results demonstrate a significant recovery in performance (e.g., recall) after applying adaptation.