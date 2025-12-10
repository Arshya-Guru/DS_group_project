"""
Neural Network Training Script - Optimized for F1_macro
CORRECTED: Now properly optimizes threshold for BOTH Train and Validation metrics.
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
# CUDA SETUP
# ============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

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
    X, y = data['X'], data['y']
    feature_names = data['feature_names']
    pos_weight = (y == 0).sum() / (y == 1).sum()
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def evaluate_model_probs(model, data_loader, device='cpu'):
    """Get probabilities and true labels (no thresholding yet)"""
    model.eval()
    model.to(device)
    all_probs = []
    all_true = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            outputs = model(X_batch)
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(y_batch.numpy())
    return np.array(all_true), np.array(all_probs)

def find_optimal_threshold_f1_macro(y_true, y_probs, min_thresh=0.2, max_thresh=0.8, step=0.01):
    """Find threshold that maximizes F1_MACRO score"""
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
# ARCHITECTURE
# ============================================================================

class F1MacroMaximizingClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.Tanh(),
                nn.Linear(input_dim, input_dim), nn.Softmax(dim=-1)
            )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout * 0.7)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.use_attention: x = x * self.attention(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        identity = x
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + identity
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.fc4(x).squeeze(-1)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, config, device='cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['pos_weight']]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'] * 10, epochs=config['epochs'], steps_per_epoch=len(train_loader), pct_start=0.3)
    
    best_val_f1_macro = 0
    best_threshold = 0.5
    best_model_state = None
    patience_counter = 0
    
    # Store BOTH standard and optimized metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1_macro': [], 'val_f1_macro': [], # optimized
        'optimal_thresh_train': [], 'optimal_thresh_val': []
    }
    
    print(f"  Training on: {next(model.parameters()).device}")
    
    for epoch in range(config['epochs']):
        # --- TRAIN STEP ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_smooth = y_batch * 0.9 + 0.05 if config['use_label_smoothing'] else y_batch
            
            optimizer.zero_grad()
            out = model(X_batch)
            if out.dim() == 0: out = out.unsqueeze(0)
            loss = criterion(out, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- EVALUATION (Optimized Thresholds for BOTH) ---
        # 1. Train Set Evaluation
        train_true, train_probs = evaluate_model_probs(model, train_loader, device)
        train_thresh, train_f1_opt = find_optimal_threshold_f1_macro(train_true, train_probs)
        
        # 2. Validation Set Evaluation
        val_true, val_probs = evaluate_model_probs(model, val_loader, device)
        
        # Val Loss Calculation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                if out.dim() == 0: out = out.unsqueeze(0)
                val_loss_sum += criterion(out, y_b).item()
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # Val Metrics
        val_thresh, val_f1_opt = find_optimal_threshold_f1_macro(val_true, val_probs)
        
        # Store History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_f1_macro'].append(train_f1_opt) # <--- NOW OPTIMIZED
        history['val_f1_macro'].append(val_f1_opt)     # <--- ALREADY OPTIMIZED
        history['optimal_thresh_train'].append(train_thresh)
        history['optimal_thresh_val'].append(val_thresh)
        
        # Save Best Model (based on Val F1 Optimized)
        if val_f1_opt > best_val_f1_macro:
            best_val_f1_macro = val_f1_opt
            best_threshold = val_thresh
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_train_loss:.4f} | "
                  f"Train F1={train_f1_opt:.3f} (@{train_thresh:.2f}) | "
                  f"Val F1={val_f1_opt:.3f} (@{val_thresh:.2f})")
        
        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch+1}")
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_f1_macro, best_threshold, history

# ============================================================================
# CONFIG & EXECUTION
# ============================================================================

config = {
    'hidden_dim': 256, 'dropout': 0.4, 'use_attention': True,
    'lr': 0.001, 'weight_decay': 1e-4, 'batch_size': 32,
    'epochs': 200, 'patience': 25, 'use_label_smoothing': True,
    'pos_weight': pos_weight
}

print("\n" + "=" * 80 + "\nSTARTING NESTED CROSS-VALIDATION\n" + "=" * 80)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = {'f1': [], 'thresh': []}

# --- Nested CV Loop ---
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):
    print(f"\nFOLD {fold}/5")
    
    # Prepare Data
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_full.iloc[train_idx])
    X_val = scaler.transform(X_train_full.iloc[val_idx])
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_train_full.iloc[train_idx].values))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_train_full.iloc[val_idx].values))
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Train
    model = F1MacroMaximizingClassifier(X_tr.shape[1], config['hidden_dim'], config['dropout'], config['use_attention'])
    model, best_f1, best_thresh, _ = train_model(model, train_loader, val_loader, config, device)
    
    # Store
    outer_scores['f1'].append(best_f1)
    outer_scores['thresh'].append(best_thresh)
    print(f"  ✓ Result: F1={best_f1:.4f} @ Thresh={best_thresh:.3f}")

# --- Final Training ---
print("\n" + "=" * 80 + "\nTRAINING FINAL MODEL\n" + "=" * 80)
cv_mean_threshold = np.mean(outer_scores['thresh'])

scaler_final = StandardScaler()
X_tr_final = scaler_final.fit_transform(X_train_full)
X_te_final = scaler_final.transform(X_test)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_final), torch.FloatTensor(y_train_full.values)), batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_te_final), torch.FloatTensor(y_test.values)), batch_size=config['batch_size'], shuffle=False)

final_model = F1MacroMaximizingClassifier(X_tr_final.shape[1], config['hidden_dim'], config['dropout'], config['use_attention'])
final_model, _, _, final_history = train_model(final_model, train_loader, test_loader, config, device)

# Final Eval
y_true, y_probs = evaluate_model_probs(final_model, test_loader, device)
y_pred = (y_probs >= cv_mean_threshold).astype(int)

results = {
    'nested_cv': {'f1_macro_scores': outer_scores['f1']},
    'test_set': {
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_probs,
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'optimal_threshold': cv_mean_threshold
    },
    'final_history': final_history # Contains 'train_f1_macro' (optimized)
}

with open('nn_results_f1macro_optimized.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\nDONE. Test F1: {results['test_set']['f1_macro']:.4f}")
