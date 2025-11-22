# Predicting Early Cognitive Impairment Using MRI and Biologically-Guided Feature Engineering

## Overview

This project addresses a critical research gap in screening for cognitive impairment by developing machine learning models to predict early signs of mild cognitive impairment (MCI) and Alzheimer's disease (AD) risk. Rather than using all available brain imaging data, we focus on biologically relevant feature engineering using specific brain regions known to be involved in cognitive impairment.

**Course:** 2025 Fall DS9000: Introduction To Machine Learning

**Team Members:**
- Arshya Pooladi-Darvish
- David Wu
- Victor Lau

## Research Question

Can we improve early detection of cognitive impairment by using targeted feature engineering based on neurobiologically relevant brain regions rather than all available neuroimaging data?

## Dataset

### OASIS-1 (Open Access Series of Imaging Studies)

The project uses the **OASIS-1** cross-sectional dataset, which contains MRI data and clinical assessments for subjects aged 18-96 years.

**Data Sources:**
- Primary: [Kaggle OASIS-1 Dataset](https://www.kaggle.com/datasets/ninadaithal/oasis-1-shinohara/data)
- Alternative: [OASIS Project Website](https://sites.wustl.edu/oasisbrains/)
- Additional: [ADNI (Alzheimer's Disease Neuroimaging Initiative)](https://adni.loni.usc.edu/)

**Dataset Files:**
- `oasis_cross-sectional.csv` (436 records) - Demographic and clinical data
- `oasis_roi_volumes.tsv` (406 records) - Brain region volumes from MRI
- `baseline_data.csv` (768 records) - Additional ADNI baseline data

**Target Variable:**
- **CDR (Clinical Dementia Rating)**: Binary classification
  - CDR = 0: Cognitively normal (725 subjects)
  - CDR > 0: Cognitive impairment (444 subjects)

## Methodology

### 1. Feature Engineering

We engineered **8 biologically-motivated features** based on brain regions affected early in Alzheimer's disease:

#### Ratio Features:
1. **L_Hippocampus_L_Ventricle_Ratio** - Left hippocampus to left lateral ventricle
2. **L_Entorhinal_L_Ventricle_Ratio** - Left entorhinal cortex to left lateral ventricle
3. **R_Hippocampus_R_Ventricle_Ratio** - Right hippocampus to right lateral ventricle
4. **R_Entorhinal_R_Ventricle_Ratio** - Right entorhinal cortex to right lateral ventricle
5. **Lateral_Ventricle_Volume** - Combined lateral ventricle volume
6. **Age** - Subject age
7. **M/F** - Sex (binary encoded)
8. **nWBV** - Normalized whole brain volume

**Biological Rationale:**
- **Hippocampus**: Critical for memory formation; shows early atrophy in AD
- **Entorhinal cortex**: One of the first regions affected in AD
- **Lateral ventricles**: Enlarge as brain tissue degenerates
- **Ratios**: Normalize for individual brain size variations and capture relative atrophy

### 2. Statistical Analysis

- Group mean differences between CDR=0 vs CDR>0
- T-tests with FDR (False Discovery Rate) correction
- Chi-square tests for categorical variables

### 3. Machine Learning Models

We compared three classification algorithms:

1. **Logistic Regression**
   - Class-weighted for imbalanced data
   - L2 regularization
   - Max iterations: 10,000

2. **Support Vector Machine (SVM)**
   - RBF kernel with class weighting
   - Hyperparameter tuning via GridSearchCV (C, gamma)
   - Probability estimates enabled

3. **XGBoost Classifier**
   - Binary logistic objective
   - Extensive hyperparameter tuning
   - Scale_pos_weight: 1.6338 for class imbalance

4. **Neural Network (Advanced)**
   - F1-Maximizing architecture with feature attention
   - Wide-shallow network (256 hidden units)
   - Focal loss and label smoothing
   - Optimal threshold tuning for F1 score

### 4. Model Evaluation

- **Nested Cross-Validation**: 5 outer folds, 3 inner folds
- **Metrics**: Macro F1, AUC-ROC, Accuracy
- **Test Set**: 20% holdout for final evaluation
- **Threshold Optimization**: Fine-tuned decision boundaries for maximizing F1 score

## Project Structure

```
DS_group_project/
├── notebook.ipynb                    # Main analysis notebook
├── train_nn.py                       # Neural network training script (v1)
├── train_nn2.py                      # Neural network training script (v2)
├── train_nn_final.py                 # Final optimized neural network
├── oasis_cross-sectional.csv         # OASIS clinical data
├── oasis_roi_volumes.tsv             # Brain region volumes
├── baseline_data.csv                 # ADNI baseline data
├── nn_input_data.pkl                 # Preprocessed neural network input (v1)
├── nn_input_data_option2.pkl         # Preprocessed neural network input (v2)
├── nn_results.pkl                    # Neural network results (v1)
├── nn_results_option2.pkl            # Neural network results (v2)
├── nn_results_f1_optimized.pkl       # Optimized neural network results
└── README.md                         # This file
```

## Installation

### Requirements

```bash
# Python 3.8+
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
torch
scipy
statsmodels
```

### Setup

```bash
# Clone the repository
git clone https://github.com/Arshya-Guru/DS_group_project.git
cd DS_group_project

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch scipy statsmodels

# For Jupyter notebook
pip install jupyter
```

## Usage

### Running the Main Analysis

```bash
# Open and run the main notebook
jupyter notebook notebook.ipynb
```

### Training Neural Network Models

```bash
# Train the final optimized neural network
python train_nn_final.py

# This will:
# - Perform nested cross-validation (5 outer, 3 inner folds)
# - Optimize F1 score with threshold tuning
# - Train final model on full training set
# - Evaluate on test set
# - Save results to nn_results_f1_optimized.pkl
# - Save model to final_nn_model_f1_optimized.pth
```

**Expected Output:**
- Training progress with F1 scores per epoch
- Nested CV summary with mean ± std metrics
- Test set performance metrics
- Confusion matrix and classification report

## Results

### Model Performance Summary

| Model | Macro F1 | AUC-ROC | Accuracy |
|-------|----------|---------|----------|
| **Logistic Regression** | 0.720 | 0.812 | 0.720 |
| **SVM** | 0.720 | 0.795 | 0.720 |
| **XGBoost** | 0.658 | 0.807 | 0.680 |

### Cross-Validation Results (Logistic Regression - Best Model)

- **Mean Macro F1**: 0.674 ± 0.015
- **Mean AUC-ROC**: 0.764 ± 0.018
- **Mean Accuracy**: 0.676 ± 0.016
- **Optimal Threshold**: 0.512

### Neural Network Performance

The F1-maximizing neural network with attention mechanism achieved:
- Improved F1 scores through threshold optimization
- Robust performance via nested cross-validation
- Feature attention weights for interpretability
- Training history saved for loss/metric visualization

## Key Findings

1. **Biologically-informed feature engineering is effective**: Using only 8 carefully selected features based on neurobiological knowledge of AD achieved strong performance (AUC ~0.80-0.81).

2. **Logistic Regression and SVM performed best**: Both achieved Macro F1 of 0.720 and strong AUC scores, demonstrating that simpler models can be highly effective with good feature engineering.

3. **Hippocampus and entorhinal cortex ratios are predictive**: The engineered ratio features capturing relative atrophy in these regions showed strong discriminative power.

4. **Threshold optimization improves F1**: Fine-tuning the decision threshold beyond the default 0.5 can meaningfully improve F1 score.

5. **Consistent generalization**: The nested cross-validation approach showed that models generalize well, with test set performance closely matching CV performance.

## Technical Highlights

### Neural Network Architecture
- **Wide-shallow design**: 256 hidden units in the first layer to capture feature interactions
- **Feature attention mechanism**: Learns to weight important features
- **Residual connections**: Improves gradient flow and training stability
- **Focal loss option**: Focuses learning on hard examples
- **Label smoothing**: Prevents overconfident predictions
- **OneCycle learning rate scheduler**: Faster convergence

### Best Practices Implemented
- Nested cross-validation for unbiased performance estimation
- Stratified splits to maintain class balance
- Feature scaling (StandardScaler)
- Class weighting for imbalanced data
- Hyperparameter tuning via GridSearchCV
- Threshold optimization for F1 maximization
- Early stopping with patience
- Gradient clipping for stable training

## Future Work

- Incorporate longitudinal data for progression prediction
- Explore additional brain regions and neuroimaging modalities
- Test ensemble methods combining multiple models
- Investigate deep learning architectures for direct MRI image analysis
- Validate on external datasets for generalizability
- Clinical deployment considerations and interpretability

## Acknowledgments

- **OASIS Project**: Washington University School of Medicine for making the dataset publicly available
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative for additional baseline data
- **Course Instructors**: DS9000 Introduction To Machine Learning

## References

1. Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.

2. LaMontagne, P. J., et al. (2019). OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease. *medRxiv*.

## License

This project is for educational purposes as part of DS9000 coursework. The OASIS dataset is subject to its own terms of use. Please refer to the [OASIS website](https://sites.wustl.edu/oasisbrains/) for dataset licensing information.

## Contact

For questions or collaboration inquiries, please contact the team members through the course platform or create an issue in this repository.

---

**Note**: This project demonstrates machine learning applications in medical research. The models are for research and educational purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.
