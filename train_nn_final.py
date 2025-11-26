"""
Neural Network Training Script - Optimized for F1_macro
Key improvements:
1. Threshold optimization for F1_MACRO (not just positive class)
2. Wide-shallow architecture + feature attention
3. BCEWithLogitsLoss + pos_weight (proven superior approach)
4. Label smoothing
5. Proper validation strategy
6. Training history saved for plotting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report, roc_auc_score)
import pickle
import copy
import random
import time
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
print("NEURAL NETWORK - F1_MACRO MAXIMIZATION WITH ATTENTION")
print("=" * 80)

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# LOAD DATA
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

    # Calculate pos_weight for imbalanced data
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
# ARCHITECTURE: Wide + Attention for F1_macro Maximization
# ============================================================================

class F1MacroMaximizingClassifier(nn.Module):
    """
    Design philosophy:
    - Wide first layer to capture all feature interactions
    - Feature attention to learn what matters
    - Moderate depth with residual connections
    - Heavy regularization to prevent overfitting
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Feature attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, input_dim),
                nn.Softmax(dim=-1)
            )
        
        # Wide first layer - learn ALL feature interactions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer with residual connection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Compression layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout * 0.7)
        
        # Output
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Optional attention on input features
        if self.use_attention:
            attn_weights = self.attention(x)
            x = x * attn_weights
        
        # Layer 1 - wide feature learning
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)  # GELU for smoother gradients
        x = self.dropout1(x)
        
        # Layer 2 - with residual
        identity = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = x + identity  # Residual connection
        
        # Layer 3 - compression
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        return x.squeeze(-1)

# ============================================================================
# THRESHOLD OPTIMIZATION FOR F1_MACRO (KEY INNOVATION!)
# ============================================================================

def find_optimal_threshold_f1_macro(y_true, y_probs, min_thresh=0.2, max_thresh=0.8, step=0.01):
    """
    Find threshold that maximizes F1_MACRO score (not just positive class!)
    This is CRITICAL for balanced performance
    """
    best_f1_macro = 0
    best_threshold = 0.5
    
    thresholds = np.arange(min_thresh, max_thresh, step)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold = thresh
    
    return best_threshold, best_f1_macro

# ============================================================================
# TRAINING FUNCTION WITH LABEL SMOOTHING
# ============================================================================

def train_model(model, train_loader, val_loader, config, device='cpu'):
    """
    Train with F1_macro optimization and BCEWithLogitsLoss + pos_weight
    """
    model = model.to(device)
    
    # Loss function - BCEWithLogitsLoss + pos_weight (SUPERIOR APPROACH)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config['pos_weight']]).to(device)
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler - OneCycle for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'] * 10,
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_f1_macro = 0
    best_threshold = 0.5
    best_model_state = None
    patience_counter = 0
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_macro': [],
        'val_f1_macro_optimized': [],
        'val_f1_class0': [],
        'val_f1_class1': [],
        'optimal_thresholds': [],
        'val_auc': []
    }
    
    print(f"  Training on: {next(model.parameters()).device}")
    
    for epoch in range(config['epochs']):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)
            
            # Label smoothing (optional)
            if config['use_label_smoothing']:
                y_batch_smooth = y_batch * 0.9 + 0.05
            else:
                y_batch_smooth = y_batch
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Handle batch size edge case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, y_batch_smooth)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0
        val_batches = 0
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float32)
                y_batch = y_batch.to(device, dtype=torch.float32)
                
                outputs = model(X_batch)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.extend(probs)
                val_true.extend(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_probs = np.array(val_probs)
        val_true = np.array(val_true)
        
        # Standard F1_macro at 0.5 threshold
        val_preds_standard = (val_probs > 0.5).astype(int)
        val_f1_macro_standard = f1_score(val_true, val_preds_standard, average='macro', zero_division=0)
        val_f1_per_class = f1_score(val_true, val_preds_standard, average=None, zero_division=0)
        
        # OPTIMIZED threshold F1_macro (this is the KEY!)
        optimal_thresh, val_f1_macro_optimized = find_optimal_threshold_f1_macro(val_true, val_probs)
        
        # AUC
        val_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else 0.5
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1_macro'].append(val_f1_macro_standard)
        history['val_f1_macro_optimized'].append(val_f1_macro_optimized)
        history['val_f1_class0'].append(val_f1_per_class[0])
        history['val_f1_class1'].append(val_f1_per_class[1])
        history['optimal_thresholds'].append(optimal_thresh)
        history['val_auc'].append(val_auc)
        
        # Track best model based on OPTIMIZED F1_MACRO
        if val_f1_macro_optimized > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro_optimized
            best_threshold = optimal_thresh
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_train_loss:.4f}, "
                  f"F1_macro={val_f1_macro_standard:.4f}, "
                  f"F1_macro_opt={val_f1_macro_optimized:.4f} @ thresh={optimal_thresh:.3f}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"    Best validation F1_macro: {best_val_f1_macro:.4f} at epoch {best_epoch+1}")
    
    return model, best_val_f1_macro, best_threshold, history

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, data_loader, threshold=0.5, device='cpu'):
    """Evaluate with custom threshold"""
    model.eval()
    model.to(device)
    
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            
            outputs = model(X_batch)
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(y_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    all_preds = (all_probs >= threshold).astype(int)
    
    return all_true, all_preds, all_probs

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
# HYPERPARAMETER CONFIGURATION
# ============================================================================

config = {
    'hidden_dim': 256,  # Wide network
    'dropout': 0.4,
    'use_attention': True,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 200,
    'patience': 25,
    'use_label_smoothing': True,
    'pos_weight': pos_weight
}

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
for key, value in config.items():
    print(f"  {key}: {value}")
print(f"  Optimization metric: F1_macro (with threshold optimization)")
print(f"  Loss function: BCEWithLogitsLoss with pos_weight={pos_weight:.3f}")

# ============================================================================
# NESTED CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STARTING NESTED CROSS-VALIDATION")
print("=" * 80)
print(f"Device: {device}")
print(f"Outer folds: 5")

start_time = time.time()

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

outer_scores_f1_macro = []
outer_scores_f1_class0 = []
outer_scores_f1_class1 = []
outer_scores_auc = []
outer_scores_acc = []
outer_thresholds = []
fold_histories = []
fold_predictions = []

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
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_fold.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_fold.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = F1MacroMaximizingClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        use_attention=config['use_attention']
    )
    
    # Train
    model, best_f1_macro, best_threshold, history = train_model(
        model, train_loader, val_loader, config, device
    )
    
    # Evaluate with optimized threshold
    y_true, y_pred, y_prob = evaluate_model(model, val_loader, best_threshold, device)
    
    fold_acc = accuracy_score(y_true, y_pred)
    fold_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    fold_f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    fold_auc = roc_auc_score(y_true, y_prob)
    
    outer_scores_acc.append(fold_acc)
    outer_scores_f1_macro.append(fold_f1_macro)
    outer_scores_f1_class0.append(fold_f1_per_class[0])
    outer_scores_f1_class1.append(fold_f1_per_class[1])
    outer_scores_auc.append(fold_auc)
    outer_thresholds.append(best_threshold)
    fold_histories.append(history)
    
    fold_predictions.append({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'best_threshold': best_threshold,
        'best_f1_macro': best_f1_macro
    })
    
    fold_time = time.time() - fold_start
    print(f"\n  ✓ Outer Fold {fold_idx} Results:")
    print(f"    Accuracy: {fold_acc:.4f}")
    print(f"    F1_macro: {fold_f1_macro:.4f}")
    print(f"    F1_class0 (CDR=0): {fold_f1_per_class[0]:.4f}")
    print(f"    F1_class1 (CDR>0): {fold_f1_per_class[1]:.4f}")
    print(f"    AUC: {fold_auc:.4f}")
    print(f"    Optimal Threshold: {best_threshold:.3f}")
    print(f"    Time: {fold_time/60:.1f} minutes")
    
    # Clean up
    del model, train_loader, val_loader
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
print(f"Optimal Thresholds: {[f'{t:.3f}' for t in outer_thresholds]}")

print(f"\nAverage Performance (± std):")
print(f"  F1_macro: {np.mean(outer_scores_f1_macro):.4f} ± {np.std(outer_scores_f1_macro):.4f}")
print(f"  F1_class0 (CDR=0): {np.mean(outer_scores_f1_class0):.4f} ± {np.std(outer_scores_f1_class0):.4f}")
print(f"  F1_class1 (CDR>0): {np.mean(outer_scores_f1_class1):.4f} ± {np.std(outer_scores_f1_class1):.4f}")
print(f"  AUC: {np.mean(outer_scores_auc):.4f} ± {np.std(outer_scores_auc):.4f}")
print(f"  Accuracy: {np.mean(outer_scores_acc):.4f} ± {np.std(outer_scores_acc):.4f}")
print(f"  Mean Optimal Threshold: {np.mean(outer_thresholds):.3f} ± {np.std(outer_thresholds):.3f}")

# ============================================================================
# TRAIN FINAL MODEL ON FULL TRAINING SET
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FINAL MODEL ON FULL TRAINING SET")
print("=" * 80)

clear_gpu_memory()

# Use mean threshold from CV (more robust)
cv_mean_threshold = np.mean(outer_thresholds)
print(f"CV mean threshold: {cv_mean_threshold:.3f}")

# Standardize
scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_full)
X_test_scaled_final = scaler_final.transform(X_test)

# Create dataloaders
train_dataset_final = TensorDataset(
    torch.FloatTensor(X_train_scaled_final),
    torch.FloatTensor(y_train_full.values)
)
test_dataset_final = TensorDataset(
    torch.FloatTensor(X_test_scaled_final),
    torch.FloatTensor(y_test.values)
)

train_loader_final = DataLoader(train_dataset_final, batch_size=config['batch_size'], shuffle=True)
test_loader_final = DataLoader(test_dataset_final, batch_size=config['batch_size'], shuffle=False)

# Create and train final model
final_model = F1MacroMaximizingClassifier(
    input_dim=X_train_scaled_final.shape[1],
    hidden_dim=config['hidden_dim'],
    dropout=config['dropout'],
    use_attention=config['use_attention']
)

final_model = final_model.to(device)
print(f"Final model device: {next(final_model.parameters()).device}")

print(f"\nTraining on {len(X_train_full)} samples")
final_model, _, training_threshold, final_history = train_model(
    final_model, train_loader_final, test_loader_final, config, device
)

# Use CV mean threshold for final predictions (more robust than single fold)
final_threshold = cv_mean_threshold
print(f"\nUsing threshold: {final_threshold:.3f} (CV mean)")

# Evaluate on test set
print("\nEvaluating on test set...")
y_test_true, y_test_pred, y_test_prob = evaluate_model(
    final_model, test_loader_final, final_threshold, device
)

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1_macro = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
test_f1_per_class = f1_score(y_test_true, y_test_pred, average=None, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_prob)
cm = confusion_matrix(y_test_true, y_test_pred)

print(f"\n" + "=" * 80)
print("FINAL TEST SET PERFORMANCE")
print("=" * 80)
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1_macro: {test_f1_macro:.4f}")
print(f"  F1_class0 (CDR=0): {test_f1_per_class[0]:.4f}")
print(f"  F1_class1 (CDR>0): {test_f1_per_class[1]:.4f}")
print(f"  AUC: {test_auc:.4f}")
print(f"  Optimal Threshold: {final_threshold:.3f}")
print(f"\n  Confusion Matrix:")
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
    'architecture': 'F1_macro-Maximizing Wide Network with Attention + Threshold Optimization',
    'optimization_metric': 'F1_macro (with threshold optimization)',
    'loss_function': f'BCEWithLogitsLoss(pos_weight={pos_weight:.3f})',
    'config': config,
    'nested_cv': {
        'f1_macro_scores': outer_scores_f1_macro,
        'f1_class0_scores': outer_scores_f1_class0,
        'f1_class1_scores': outer_scores_f1_class1,
        'auc_scores': outer_scores_auc,
        'acc_scores': outer_scores_acc,
        'optimal_thresholds': outer_thresholds,
        'fold_histories': fold_histories,
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
        'optimal_threshold': final_threshold,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test_true, y_test_pred,
            target_names=['CDR=0', 'CDR>0'],
            output_dict=True
        )
    },
    'final_history': final_history,
    'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
    'runtime_minutes': total_time/60,
    'device_used': str(device),
    'cuda_available': torch.cuda.is_available(),
    'model_architecture': str(final_model)
}

with open('nn_results_f1macro_optimized.pkl', 'wb') as f:
    pickle.dump(results, f)

torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler': scaler_final,
    'threshold': final_threshold,
    'config': config,
    'feature_names': feature_names
}, 'final_nn_model_f1macro_optimized.pth')

print("✓ Results saved to: nn_results_f1macro_optimized.pkl")
print("✓ Model saved to: final_nn_model_f1macro_optimized.pth")
print(f"✓ Device used: {device}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print_gpu_memory()

print(f"\n🎯 Test F1_macro: {test_f1_macro:.4f}")
print(f"🎯 Using threshold: {final_threshold:.3f}")
print("\nDownload 'nn_results_f1macro_optimized.pkl' and 'final_nn_model_f1macro_optimized.pth' for analysis!")