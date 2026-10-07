# Automated Identification of Gait Abnormalities

An automated end-to-end spatiotemporal deep learning framework for predicting the Gait Deviation Index (GDI) from monocular video sequences using DensePose representations and deep neural networks.

---

## Academic Metadata

* **Course:** UE24CS352A – Machine Learning Mini-Project
* **Institution:** PES University, Ring Road Campus
* **Team Members:**
  * Nimay Ballal (`PES1UG24CS301`)
  * Nishanth T Chigilar (`PES1UG24CS302`)
* **Repository Link:** [ML-MiniProject-GaitAnalysis](https://github.com/ChigilarNishanth/ML-MiniProject-GaitAnalysis)

---

## Project Overview

Neuromuscular pathologies such as Cerebral Palsy, Parkinson's disease, and stroke frequently manifest in abnormal walking gait patterns. The **Gait Deviation Index (GDI)** is a standardized clinical numerical score (ranging from 0 to 100) used by biomechanists and orthopedic surgeons to evaluate gait pathology severity.

The standard gold-standard method for measuring GDI relies on marker-based 3D motion capture in specialized laboratories, which requires expensive multi-camera array equipment, calibration, and long patient evaluation sessions.

This project implements **GDI-Net**, an automated machine learning pipeline that predicts continuous GDI scores directly from monocular RGB video feeds captured on commodity devices. Raw video frames are featurized into 3D body surface representations using **DensePose IUV coordinates**, removing background clutter and isolating anatomical movement kinematics.

---

## Model Architecture

The network combines spatial feature extraction with temporal sequence modeling across consecutive frame sequences:

1. **Spatial Module (Custom 2D CNN):** Processes individual DensePose frames of dimensions (480 x 640 x 3) through three repeating blocks consisting of Convolutional layers, Batch Normalization, Max Pooling, and 0.7 Dropout. The spatial outputs are flattened and compressed into a compact 16-dimensional latent vector per frame.
2. **Temporal Module (LSTM):** Sequences 10 consecutive frame feature vectors into a 256-unit Recurrent LSTM network to track gait trajectories, step cadence, and motion dynamics across time.
3. **Regression Output Layer:** A dense linear layer mapping the temporal features to a single continuous scalar GDI prediction.

### Performance Benchmarks
* **Validation RMSE:** 4.4
* **Coefficient of Determination ($R^2$):** 0.86

---

## Repository Structure

```text
ML/
├── data/                  # Ground truth GDI mapping CSVs and data samples
├── docs/                  # Project Write-Up PDF and Presentation Deck
│   ├── Mini_Project_Writeup.pdf
│   └── Presentation_Slides.pdf
├── DataProcessingCode/    # DensePose frame extraction scripts
├── Code/
│   └── ModelCode/         # Model training and prediction pipeline
│       └── FGDI_S10.py    # Main 2D CNN + LSTM execution script
├── .gitignore             # Environment exclusions
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

