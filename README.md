# DS Group Project

A machine learning project focused on neuroimaging data analysis using neural networks and the OASIS and ADNI brain imaging dataset.

## Overview

This project implements deep learning models to analyze brain imaging data from the OASIS (Open Access Series of Imaging Studies) and ADNI (Alzheimer's Disease Neuroimaging Initiative) datasets. The goal is to perform classification tasks on neuroimaging features using PyTorch-based neural networks.

## Dataset

The project uses MRI derivatives from obtained by running Freesurfer's (7.4.1) SynthSeg on volumes from the OASIS and ADNI datasets.

## Project Structure

```
.
├── notebook.ipynb                      # Main analysis notebook
├── train_nn.py                         # Initial neural network training script
├── train_nn2.py                        # Second iteration training script
├── train_nn_final.py                   # Final optimized training script
├── oasis_cross-sectional.csv          # OASIS cross-sectional data
├── oasis_roi_volumes.tsv              # ROI volume measurements
├── baseline_data.csv                   # Baseline comparison data
├── nn_input_data.pkl                   # Preprocessed neural network input
├── nn_input_data_option2.pkl          # Alternative preprocessed input
├── final_nn_model_f1macro.pth         # Trained model (F1-macro optimized)
├── final_nn_model_f1macro_optimized.pth  # Optimized model variant
└── final_nn_model_option2_f1macro.pth # Alternative model architecture
```

## Technologies

- **Python 3.10**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities and preprocessing
- **Pandas** - Data manipulation (implied)
- **PyTorch** - GPU acceleration library

## Features

- Neural network architecture optimized for F1-macro score
- Cross-validation with stratified k-fold splitting
- Feature attention mechanisms
- Label smoothing and class balancing
- Threshold optimization for classification
- GPU/CUDA support for accelerated training

## Setup

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install torch numpy scikit-learn pandas
   ```

## Usage

### Training Models
These have been run for you, and outputs are stored in the pkl files. However should you want to do it yourself.

Best model training script:
```bash
python train_nn_final.py
```

Alternative architectures:
```bash
python train_nn.py
python train_nn2.py
```

### Analysis

Open and run the Jupyter notebook for exploratory analysis:
```bash
jupyter notebook notebook.ipynb
```

## Model Features

- Wide-shallow architecture with feature attention
- BCEWithLogitsLoss with positive class weighting
- Label smoothing for regularization
- Proper train/validation/test splitting
- Comprehensive evaluation metrics tracking

## License

This project uses MRI-derived features from the OASIS and ADNI datasets. Please refer to the respective dataset licensing terms for the original MRI voumes.
