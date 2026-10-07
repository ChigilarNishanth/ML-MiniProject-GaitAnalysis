# Automated Identification of Gait Abnormalities

An automated end-to-end spatiotemporal deep learning framework for predicting the Gait Deviation Index (GDI) from monocular video sequences using DensePose representations and deep neural networks.

---

## Academic Metadata

- **Course:** UE24CS352A – Machine Learning Mini-Project
- **Institution:** PES University, Ring Road Campus
- **Team Members:**
  - Nimay Ballal (`PES1UG24CS301`)
  - Nishanth T Chigilar (`PES1UG24CS302`)
- **Repository:** [ML-MiniProject-GaitAnalysis](https://github.com/ChigilarNishanth/ML-MiniProject-GaitAnalysis)

---

## Project Overview

Neuromuscular pathologies such as cerebral palsy, Parkinson's disease, and stroke frequently manifest in abnormal walking gait patterns. The **Gait Deviation Index (GDI)** is a standardized clinical measure used to quantify gait abnormalities.

The standard gold-standard method for measuring GDI relies on marker-based 3D motion capture in specialized laboratories. This approach requires expensive multi-camera equipment, careful calibration, and trained personnel, making it difficult to use in many practical settings.

This project implements **GDI-Net**, an automated machine learning pipeline that predicts continuous GDI scores directly from monocular RGB video feeds captured using commodity devices. Raw video frames are converted into DensePose representations, which are then processed by a spatiotemporal deep learning model.

The proposed system combines a custom 2D convolutional neural network for spatial feature extraction with an LSTM-based temporal module for learning gait dynamics across consecutive frames.

---

## Objectives

- Develop an automated system for identifying gait abnormalities from monocular video.
- Extract meaningful human-pose representations using DensePose.
- Learn spatial gait features using a custom 2D CNN.
- Model temporal gait patterns using an LSTM network.
- Predict a continuous GDI score for each gait sequence.
- Evaluate the model using RMSE and the coefficient of determination (`R²`).

---

## Model Architecture

The network combines spatial feature extraction with temporal sequence modeling across consecutive frame sequences:

1. **Spatial Module — Custom 2D CNN**
   - Processes individual DensePose frames with dimensions `(480 × 640 × 3)`.
   - Uses three repeating convolutional blocks.
   - Each block includes convolutional layers, batch normalization, max pooling, and nonlinear activation functions.
   - Produces a compact feature representation for every frame.

2. **Temporal Module — LSTM**
   - Groups 10 consecutive frame feature vectors into a sequence.
   - Uses a 256-unit LSTM network to learn gait trajectories, step cadence, and motion dynamics over time.

3. **Regression Output Layer**
   - Maps the temporal representation to a single continuous GDI prediction using a dense linear layer.

### Performance Benchmarks

- **Validation RMSE:** 4.4
- **Coefficient of Determination (`R²`):** 0.86

---

## Repository Structure

```text
ML-MiniProject-GaitAnalysis/
├── data/                  # Ground-truth GDI mappings and data samples
├── docs/                  # Project write-up and presentation deck
│   ├── Mini_Project_Writeup.pdf
│   └── Presentation_Slides.pdf
├── DataProcessingCode/    # DensePose frame extraction scripts
├── Code/
│   └── ModelCode/         # Model training and prediction pipeline
│       └── FGDI_S10.py    # Main 2D CNN + LSTM execution script
├── .gitignore             # Environment exclusions
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Setup and Execution Instructions

### 1. Prerequisites

Ensure that the following software is installed:

- Python 3.8 or later
- Git
- A compatible TensorFlow/Keras installation
- Sufficient storage for the DensePose frame data and trained model files

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ChigilarNishanth/ML-MiniProject-GaitAnalysis.git
cd ML-MiniProject-GaitAnalysis
pip install -r requirements.txt
```

> **Recommendation:** Use a Python virtual environment to keep project dependencies isolated.

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

Place the processed DensePose frame arrays or video sequences inside the `data/` directory.

The dataset mapping file, `imagepath_gdi.csv`, should contain the relative paths to the frame sequences and their corresponding ground-truth GDI labels. A typical mapping file may contain columns similar to the following:

```csv
image_path,gdi
path/to/sequence_001,75.4
path/to/sequence_002,82.1
```

Before training, verify that:

- All paths in `imagepath_gdi.csv` are correct relative to the repository.
- Every referenced sequence is available in the `data/` directory.
- DensePose frames have the expected dimensions and format.
- Ground-truth GDI values are numeric and free from invalid or missing entries.

### 4. Training the Model

To initiate training of the GDI-Net architecture, run:

```bash
python Code/ModelCode/FGDI_S10.py
```

The script performs the following operations:

- Loads the dataset and GDI labels.
- Reads the DensePose frame sequences.
- Extracts spatial features using the custom 2D CNN.
- Generates sequences of 10 consecutive frames.
- Trains the 256-unit LSTM temporal model.
- Applies learning-rate decay during training.
- Saves the trained model weights and relevant outputs.

### 5. Evaluation and Live Demo

To evaluate the trained model on validation sequences and display prediction metrics such as RMSE and `R²`, run:

```bash
python Code/ModelCode/FGDI_S10.py --eval
```

The evaluation step uses the trained model to generate GDI predictions for validation data and reports the resulting performance metrics.

> **Note:** The exact output and model checkpoint locations depend on the configuration implemented in `Code/ModelCode/FGDI_S10.py`.

---

## Deliverables

- **Mini-Project Write-Up:** `docs/Mini_Project_Writeup.pdf`
- **Presentation Slide Deck:** `docs/Presentation_Slides.pdf`

---

## Individual Contributions

- **Nimay Ballal (`PES1UG24CS301`):** Neural network architecture development, Keras/TensorFlow model implementation, LSTM sequence tracking, and training optimization.
- **Nishanth T Chigilar (`PES1UG24CS302`):** DensePose dataset preprocessing, data sequence loading pipelines, evaluation metric calculation, repository maintenance, and documentation.

---

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- OpenCV
- DensePose
- Convolutional Neural Networks
- Long Short-Term Memory Networks

---

## License

This project was developed as part of the UE24CS352A Machine Learning Mini-Project at PES University. Refer to the repository for the applicable project usage and distribution terms.
