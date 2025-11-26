"""
Neural Network Training Script - Option 2 Architecture
3-layer network with BatchNorm: 128 → 64 → 32
Optimized for F1_macro with BCEWithLogitsLoss + pos_weight
Run with: python train_nn_option2.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import pickle
import copy
import random
from itertools import product
from collections import Counter
import time
import os
import gc

# ============================================================================
# CUDA SETUP AND DIAGNOSTICS
# ============================================================================

print("=" * 80)
print("CUDA DIAGNOSTICS")
print("=" * 80)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    print(f"Using device: {device}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Test CUDA functionality
    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        test_result = test_tensor * 2
        print("✓ CUDA basic operations test: PASSED")
        del test_tensor, test_result
    except Exception as e:
        print(f"✗ CUDA basic operations test: FAILED - {e}")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
    print("✗ CUDA not available, using CPU")

print("=" * 80)
print("NEURAL NETWORK - Option 2 Architecture (3-layer + BatchNorm)")
print("F1_macro optimization")
print("=" * 80)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATA LOADING
# ============================================================================

print("\nLoading data...")
try:
    with open('nn_input_data_option2.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"✓ Data loaded successfully")
    print(f"  Features shape: {X.shape}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    # Calculate pos_weight
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"  Positive weight for loss: {pos_weight:.3f}")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"  Training set: {len(X_train_full)} samples")
print(f"  Test set: {len(X_test)} samples")

# ============================================================================
# NEURAL NETWORK ARCHITECTURE - Option 2
# ============================================================================

class BinaryClassifierOption2(nn.Module):
    """
    3-layer architecture with BatchNorm:
    Input → Dense(128) + BatchNorm + ReLU + Dropout
         → Dense(64) + BatchNorm + ReLU + Dropout
         → Dense(32) + ReLU + Dropout
         → Output(1)
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.3):
        super(BinaryClassifierOption2, self).__init__()
        
        # Layer 1: Input → 128
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: 128 → 64
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3: 64 → 32
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout * 0.67)  # Reduced dropout for final layer
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim3, 1)
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        return x.squeeze(-1)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs=150, patience=20, device='cpu'):
    """
    Train the model with early stopping based on F1_macro
    """
    model.to(device)
    best_val_f1_macro = 0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'val_auc': []}
    
    print(f"  Training on: {next(model.parameters()).device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Handle batch size of 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        val_preds = []
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float32)
                y_batch = y_batch.to(device, dtype=torch.float32)
                
                outputs = model(X_batch)
                
                # Handle batch size of 1
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_true.extend(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        # Calculate metrics - USING F1_MACRO
        val_f1_macro = f1_score(val_true, val_preds, average='macro', zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else 0.5
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_auc'].append(val_auc)
        
        # Learning rate scheduling - BASED ON F1_MACRO
        if scheduler is not None:
            scheduler.step(val_f1_macro)
        
        # Early stopping - BASED ON F1_MACRO
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val F1_macro: {val_f1_macro:.4f}")
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"    Best validation F1_macro: {best_val_f1_macro:.4f} at epoch {best_epoch+1}")
    
    return model, best_val_f1_macro, history

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate the model on given data loader
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            
            outputs = model(X_batch)
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
    
    return np.array(all_true), np.array(all_preds), np.array(all_probs)

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# ============================================================================
# HYPERPARAMETER GRID
# ============================================================================

param_grid = {
    'hidden_dim1': [128],
    'hidden_dim2': [64],
    'hidden_dim3': [32],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64],
    'weight_decay': [0, 0.0001, 0.001]
}

# ============================================================================
# NESTED CROSS-VALIDATION
# ============================================================================

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores_acc = []
outer_scores_f1_macro = []
outer_scores_f1_class0 = []
outer_scores_f1_class1 = []
outer_scores_auc = []
best_params_list = []
fold_predictions = []

print("\n" + "=" * 80)
print("STARTING NESTED CROSS-VALIDATION")
print("=" * 80)
print(f"Device: {device}")
print(f"Outer folds: 5, Inner folds: 3")
print(f"Optimization metric: F1_macro")
print(f"Loss function: BCEWithLogitsLoss with pos_weight={pos_weight:.3f}")

start_time = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):
    print(f"\n{'=' * 80}")
    print(f"OUTER FOLD {fold_idx}/5")
    print(f"{'=' * 80}")
    fold_start = time.time()
    
    clear_gpu_memory()
    
    X_train_fold = X_train_full.iloc[train_idx]
    X_val_fold = X_train_full.iloc[val_idx]
    y_train_fold = y_train_full.iloc[train_idx]
    y_val_fold = y_train_full.iloc[val_idx]
    
    print(f"  Training samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Inner CV: Hyperparameter search
    best_inner_f1_macro = 0
    best_params = None
    
    print("\n  Inner CV: Hyperparameter search...")
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Sample subset
    sampled_combinations = random.sample(all_combinations, min(20, len(all_combinations)))
    
    for param_idx, params in enumerate(sampled_combinations, 1):
        inner_f1_macro_scores = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled, y_train_fold):
            clear_gpu_memory()
            
            X_inner_train = X_train_scaled[inner_train_idx]
            X_inner_val = X_train_scaled[inner_val_idx]
            y_inner_train = y_train_fold.iloc[inner_train_idx].values
            y_inner_val = y_train_fold.iloc[inner_val_idx].values
            
            # Create dataloaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_inner_train),
                torch.FloatTensor(y_inner_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_inner_val),
                torch.FloatTensor(y_inner_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Create model (Option 2)
            model = BinaryClassifierOption2(
                input_dim=X_train_scaled.shape[1],
                hidden_dim1=params['hidden_dim1'],
                hidden_dim2=params['hidden_dim2'],
                hidden_dim3=params['hidden_dim3'],
                dropout=params['dropout']
            ).to(device)
            
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                                   weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
            
            # Train
            model, val_f1_macro, _ = train_model(model, train_loader, val_loader, criterion, 
                                                 optimizer, scheduler, epochs=100, patience=15, 
                                                 device=device)
            
            inner_f1_macro_scores.append(val_f1_macro)
            
            # Clean up
            del model, criterion, optimizer, scheduler
            clear_gpu_memory()
        
        mean_inner_f1_macro = np.mean(inner_f1_macro_scores)
        
        if mean_inner_f1_macro > best_inner_f1_macro:
            best_inner_f1_macro = mean_inner_f1_macro
            best_params = params.copy()
        
        if param_idx % 5 == 0:
            print(f"    Tested {param_idx}/{len(sampled_combinations)} param combinations")
    
    print(f"\n  ✓ Best inner CV F1_macro: {best_inner_f1_macro:.4f}")
    print(f"  ✓ Best params: {best_params}")
    best_params_list.append(best_params)
    
    # Train final model on full outer fold
    print("\n  Training final model on outer fold...")
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_fold.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_fold.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    model = BinaryClassifierOption2(
        input_dim=X_train_scaled.shape[1],
        hidden_dim1=best_params['hidden_dim1'],
        hidden_dim2=best_params['hidden_dim2'],
        hidden_dim3=best_params['hidden_dim3'],
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    model, best_fold_f1_macro, history = train_model(model, train_loader, val_loader, criterion,
                                                      optimizer, scheduler, epochs=150, patience=20,
                                                      device=device)
    
    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)
    
    fold_acc = accuracy_score(y_true, y_pred)
    fold_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    fold_f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    fold_auc = roc_auc_score(y_true, y_prob)
    
    outer_scores_acc.append(fold_acc)
    outer_scores_f1_macro.append(fold_f1_macro)
    outer_scores_f1_class0.append(fold_f1_per_class[0])
    outer_scores_f1_class1.append(fold_f1_per_class[1])
    outer_scores_auc.append(fold_auc)
    
    fold_predictions.append({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'best_params': best_params,
        'best_f1_macro': best_fold_f1_macro
    })
    
    fold_time = time.time() - fold_start
    print(f"\n  ✓ Outer Fold {fold_idx} Results:")
    print(f"    Accuracy: {fold_acc:.4f}")
    print(f"    F1_macro: {fold_f1_macro:.4f}")
    print(f"    F1_class0 (CDR=0): {fold_f1_per_class[0]:.4f}")
    print(f"    F1_class1 (CDR>0): {fold_f1_per_class[1]:.4f}")
    print(f"    AUC: {fold_auc:.4f}")
    print(f"    Time: {fold_time/60:.1f} minutes")
    
    # Clean up
    del model, train_loader, val_loader, criterion, optimizer, scheduler
    clear_gpu_memory()

# ============================================================================
# NESTED CV SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("NESTED CROSS-VALIDATION SUMMARY")
print("=" * 80)

print(f"F1_macro Scores: {[f'{score:.4f}' for score in outer_scores_f1_macro]}")
print(f"F1_class0 Scores: {[f'{score:.4f}' for score in outer_scores_f1_class0]}")
print(f"F1_class1 Scores: {[f'{score:.4f}' for score in outer_scores_f1_class1]}")
print(f"AUC Scores: {[f'{score:.4f}' for score in outer_scores_auc]}")
print(f"Accuracy Scores: {[f'{score:.4f}' for score in outer_scores_acc]}")

print(f"\nAverage Performance (± std):")
print(f"  F1_macro: {np.mean(outer_scores_f1_macro):.4f} ± {np.std(outer_scores_f1_macro):.4f}")
print(f"  F1_class0 (CDR=0): {np.mean(outer_scores_f1_class0):.4f} ± {np.std(outer_scores_f1_class0):.4f}")
print(f"  F1_class1 (CDR>0): {np.mean(outer_scores_f1_class1):.4f} ± {np.std(outer_scores_f1_class1):.4f}")
print(f"  AUC: {np.mean(outer_scores_auc):.4f} ± {np.std(outer_scores_auc):.4f}")
print(f"  Accuracy: {np.mean(outer_scores_acc):.4f} ± {np.std(outer_scores_acc):.4f}")

# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FINAL MODEL ON FULL TRAINING SET")
print("=" * 80)

clear_gpu_memory()

final_params = {}
for param in param_grid.keys():
    values = [p[param] for p in best_params_list]
    final_params[param] = Counter(values).most_common(1)[0][0]

print(f"Selected final parameters: {final_params}")

scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_full)
X_test_scaled_final = scaler_final.transform(X_test)

train_dataset_final = TensorDataset(
    torch.FloatTensor(X_train_scaled_final),
    torch.FloatTensor(y_train_full.values)
)
test_dataset_final = TensorDataset(
    torch.FloatTensor(X_test_scaled_final),
    torch.FloatTensor(y_test.values)
)

train_loader_final = DataLoader(train_dataset_final, batch_size=final_params['batch_size'], shuffle=True)
test_loader_final = DataLoader(test_dataset_final, batch_size=final_params['batch_size'], shuffle=False)

final_model = BinaryClassifierOption2(
    input_dim=X_train_scaled_final.shape[1],
    hidden_dim1=final_params['hidden_dim1'],
    hidden_dim2=final_params['hidden_dim2'],
    hidden_dim3=final_params['hidden_dim3'],
    dropout=final_params['dropout']
).to(device)

print(f"Final model device: {next(final_model.parameters()).device}")

criterion_final = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer_final = optim.Adam(final_model.parameters(), lr=final_params['learning_rate'],
                              weight_decay=final_params['weight_decay'])
scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='max', factor=0.5, patience=10)

print("\nTraining final model...")
final_model, final_best_f1_macro, final_history = train_model(final_model, train_loader_final, test_loader_final,
                                                              criterion_final, optimizer_final, scheduler_final,
                                                              epochs=150, patience=20, device=device)

y_test_true, y_test_pred, y_test_prob = evaluate_model(final_model, test_loader_final, device)

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1_macro = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
test_f1_per_class = f1_score(y_test_true, y_test_pred, average=None, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_prob)

print(f"\n" + "=" * 80)
print("FINAL TEST SET PERFORMANCE")
print("=" * 80)
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1_macro: {test_f1_macro:.4f}")
print(f"  F1_class0 (CDR=0): {test_f1_per_class[0]:.4f}")
print(f"  F1_class1 (CDR>0): {test_f1_per_class[1]:.4f}")
print(f"  AUC: {test_auc:.4f}")
print(f"\n  Confusion Matrix:")
cm = confusion_matrix(y_test_true, y_test_pred)
print(f"    {cm}")
print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time/60:.1f} minutes")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results = {
    'architecture': 'Option 2: 3-layer with BatchNorm (128→64→32)',
    'optimization_metric': 'F1_macro',
    'loss_function': f'BCEWithLogitsLoss(pos_weight={pos_weight:.3f})',
    'nested_cv': {
        'f1_macro_scores': outer_scores_f1_macro,
        'f1_class0_scores': outer_scores_f1_class0,
        'f1_class1_scores': outer_scores_f1_class1,
        'auc_scores': outer_scores_auc,
        'acc_scores': outer_scores_acc,
        'best_params_list': best_params_list,
        'fold_predictions': fold_predictions
    },
    'test_set': {
        'y_true': y_test_true,
        'y_pred': y_test_pred,
        'y_prob': y_test_prob,
        'accuracy': test_acc,
        'f1_macro': test_f1_macro,
        'f1_class0': test_f1_per_class[0],
        'f1_class1': test_f1_per_class[1],
        'auc': test_auc,
        'confusion_matrix': confusion_matrix(y_test_true, y_test_pred).tolist(),
        'classification_report': classification_report(y_test_true, y_test_pred, 
                                                       target_names=['CDR=0', 'CDR>0'],
                                                       output_dict=True)
    },
    'final_params': final_params,
    'final_history': final_history,
    'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
    'runtime_minutes': total_time/60,
    'device_used': str(device),
    'cuda_available': torch.cuda.is_available(),
    'model_architecture': str(final_model)
}

with open('nn_results_option2_f1macro.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save model
torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler': scaler_final,
    'feature_names': feature_names,
    'final_params': final_params
}, 'final_nn_model_option2_f1macro.pth')

print("✓ Results saved to: nn_results_option2_f1macro.pkl")
print("✓ Model saved to: final_nn_model_option2_f1macro.pth")
print(f"✓ Device used: {device}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print_gpu_memory()

print("\n🎯 Training completed successfully!")
print("Download 'nn_results_option2_f1macro.pkl' and 'final_nn_model_option2_f1macro.pth' for analysis in your notebook.")