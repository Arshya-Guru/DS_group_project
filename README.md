# DS Group Project

A machine learning project focused on neuroimaging data analysis using neural networks and the OASIS brain imaging dataset.

## Overview

This project implements deep learning models to analyze brain imaging data from the OASIS (Open Access Series of Imaging Studies) dataset. The goal is to perform classification tasks on neuroimaging features using PyTorch-based neural networks.

## Dataset

The project uses the OASIS dataset, which contains:
- Cross-sectional brain imaging data (`oasis_cross-sectional.csv`)
- Region of interest (ROI) volume measurements (`oasis_roi_volumes.tsv`)

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

- **Python 3.x**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities and preprocessing
- **Pandas** - Data manipulation (implied)

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

Run the final optimized training script:
```bash
python train_nn_final.py
```

Or explore alternative approaches:
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

This project uses the OASIS dataset. Please refer to the OASIS dataset licensing terms for data usage.
