\# Automated Identification of Gait Abnormalities (GDI Prediction)



\*\*Course:\*\* UE24CS352A - Machine Learning Mini-Project  

\*\*Institution:\*\* PES University, Ring Road Campus  



\## Team Members

\* \*\*Nimay Ballal\*\* — `PES1UG24CS301`

\* \*\*Nishanth T Chigilar\*\* — `PES1UG24CS302`



\---



\## 1. Problem Overview

Pathologies such as Cerebral Palsy, Parkinson's disease, and stroke manifest in abnormal walking gait patterns. The \*\*Gait Deviation Index (GDI)\*\* is a continuous score (`0–100`) quantifying gait pathology severity. This project implements a spatiotemporal deep learning model (\*\*GDI-Net\*\*) to predict GDI scores automatically from video sequences featurized using \*\*DensePose (IUV coordinates)\*\*.



\---



\## 2. Model Architecture

The network combines spatial feature extraction with sequence modeling:

1\. \*\*Spatial Module (2D CNN):\*\* Extracts spatial features from individual DensePose frames (`480 x 640 x 3`) across 3 Convolutional-BatchNorm-MaxPool blocks with `0.7` Dropout, flattening frames into compact 16-dimensional latent vectors.

2\. \*\*Temporal Module (LSTM):\*\* Sequences 10 consecutive frame feature vectors into a 256-unit LSTM network to track motion trajectories across time.

3\. \*\*Regression Output:\*\* Dense linear layer predicting scalar GDI values.



\### Benchmark Results

\* \*\*Top Validation RMSE:\*\* `4.4`

\* \*\*R² Score:\*\* `0.86`



\---



\## 3. Environment \& Execution



\### Setup

```bash

pip install -r requirements.txt

python Code/ModelCode/FGDI\_S10.py

