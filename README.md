# STAE-Net: Structural–Temporal Autoencoder Network for Wireless Intrusion Detection

This repository provides a complete implementation of **STAE-Net**, a structural–temporal autoencoder framework for wireless intrusion detection.  
The pipeline is designed for the **AWID3** dataset and includes:

- Sequential AWID3 data loading & Normal-only constrained splitting  
- Feature expansion, normalization, categorical encoding, variance-based sorting  
- SRF image construction using improved iGASF / iGADF / iMTF  
- Temporal segment construction via sliding windows  
- ResAE spatial reconstruction branch  
- TMAE temporal masked modeling branch  
- Adaptive score fusion with full evaluation (Precision, Recall, F1)

All steps are implemented in the seven Python scripts included in this repository.

---

## Contents

1. Project Overview  
2. Repository Structure (matches uploaded scripts exactly)  
3. Environment Setup  
4. Data Preparation  
5. Preprocessing Pipeline  
6. Image & Sequence Construction  
7. ResAE Spatial Branch  
8. TMAE Temporal Branch  
9. Adaptive Fusion & Evaluation  

---

## 1. Project Overview

STAE-Net is a dual-branch anomaly detection model designed for wireless intrusion detection.  
The model transforms tabular features into **SRF (Structured Representation Features)** images, constructs temporal sequences, and trains:

- **ResAE** for spatial anomaly reconstruction from RGB SRF images  
- **TMAE** for temporal masked sequence modeling from SRF segments  

Finally, an adaptive fusion module integrates both scores and produces final anomaly predictions.

---

## 2. Repository Structure

This structure **exactly matches the seven uploaded Python scripts**.

STAE-Net/

│

├── dataLoad.py # Load AWID3 CSV folders & perform Normal-only 60/40 split

├── dataPreprocess.py # Feature expansion, categorical encoding, normalization, variance sorting

├── imageConstruction.py # Build SRF RGB images (iGASF + iGADF + iMTF)

├── segmentConstruction.py # Sliding-window SRF segment construction (.npz)

├── resae_branch.py # ResAE model + training + scoring

├── tmae_branch.py # TMAE model + training + scoring

├── fusion.py # Adaptive fusion + thresholding + evaluation

│

├── CSV/ # Place AWID3 dataset here (CSV folders)

├── train_images/ # Generated SRF images (train)

├── test_images/ # Generated SRF images (test)

├── train_segments/ # Generated SRF segments (train)

├── test_segments/ # Generated SRF segments (test)

├── *.csv # Output score files

├── *.pth # Saved model weights

└── README.md

---

## 3. Environment Setup

Requires:

- `python >= 3.8`
- `numpy`
- `pandas`
- `scikit-learn`
- `pytorch >= 1.12`
- `opencv-python`
- `Pillow`
- `tqdm`

Install:

- `pip install -r requirements.txt`

---

## 4. Data Preparation

Place AWID3 CSV folders into:

CSV/

│

├── 1.Deauth/

│   ├──Deauth_0.csv

│   ├──Deauth_1.csv

│   ├──...

├── 2.Disas/


│   ├──...

├── ...

Run:

- `python dataLoad.py`

This script (**according to actual code**)：

1. Reads folders in numeric order  
2. Reads CSV files inside folders in numeric order  
3. Preserves row order  
4. Selects **only Normal samples** for the first 60% training split  
5. Outputs:

- `awid3_train.csv`
- `awid3_test.csv`

---

## 5. Preprocessing Pipeline

Run:

- `python dataPreprocess.py`

This script (✔ real behavior):

1. Expands feature dimension to **256**  
2. Detects numeric vs categorical features  
3. Categorical features → label encoding  
4. Computes train-only min/max  
5. Normalizes both train & test using train statistics  
6. Computes variance on normalized train features  
7. Sorts features by descending variance  
8. Saves:

- `awid3_train_preprocessed.csv`
- `awid3_test_preprocessed.csv`

---

## 6. SRF Image & Sequence Construction

### 6.1 SRF RGB Images

Run:

- `python imageConstruction.py`

This script:

1. Computes iGASF, iGADF, and iMTF  
2. Builds **RGB images (256 × 256)**  
3. Saves:

- `train_images/train_idxXXXXXX_label-XXX.png`
- `test_images/test_idxXXXXXX_label-XXX.png`

### 6.2 Sliding-Window Temporal Segments

Run:

- `python segmentConstruction.py`

This script:

1. Loads SRF images  
2. Builds **segments of length T = 8**  
3. Each segment saved as:

- `train_segments/train_seg_endidxXXXXXX_fromXXXXXX_toXXXXXX_label-XXX.npz`
- `test_segments/test_seg_endidxXXXXXX_fromXXXXXX_toXXXXXX_label-XXX.npz`

---

## 7. ResAE Spatial Branch

Run:

- `python resae_branch.py`

This script trains a **Residual Autoencoder with SE blocks** (matches uploaded code).

Outputs:

- `resae_awid3.pth`
- `resae_scores_train.csv`
- `resae_scores_test.csv`

CSV format:

- `idx, label, E_img`

---

## 8. TMAE Temporal Branch

Run:

- `python tmae_branch.py`

This script trains a **Temporal Masked Autoencoder** on sliding-window segments (matches uploaded code).

Outputs:

- `tmae_awid3.pth`
- `tmae_scores_train.csv`
- `tmae_scores_test.csv`

CSV format:

- `idx, label, E_seq`

---

## 9. Adaptive Fusion & Evaluation

Run:

- `python fusion.py`

This script:

1. Merges ResAE & TMAE scores  
2. Computes adaptive λ based on variance of Normal training samples  
3. Computes fused anomaly score  
4. Selects threshold τ from training 99th percentile  
5. Applies τ to test samples  
6. Computes:

- Precision  
- Recall  
- F1-score  

Outputs:

- `fusion_scores_train.csv`
- `fusion_scores_test.csv`
- `fusion_metrics.txty`

Example metrics file:

- `Precision : 0.98xxxx`
- `Recall : 0.97xxxx`
- `F1-score : 0.97xxxx`
