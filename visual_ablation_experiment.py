"""
Visual-Only Ablation Experiment

This experiment tests whether concept prediction still works when we REMOVE
all cross-attention (text conditioning) layers and use ONLY visual self-attention layers.

Background (from Stable Diffusion v1.5 U-Net architecture):
- attn1 = Self-Attention layers - image features attend to themselves
- attn2 = Cross-Attention layers - image features attend to text embeddings (CLIP)

Experiment Design:
1. Baseline: Use ALL LoRA layers (attn1 + attn2)
   - Uses both visual self-attention AND text cross-attention
   
2. Visual-Only: Use ONLY attn1 (self-attention) layers, REMOVE attn2 (cross-attention)
   - Uses ONLY visual features, NO text conditioning
   
3. Test all representation methods on both configurations

4. Compare: Does prediction work without text conditioning layers?
   - If visual-only works well → concepts encoded in visual features
   - If visual-only fails → concepts require text conditioning

Author: Claude  
Date: 2025-10-27
Updated: 2025-10-28 (verified layer filtering with SD v1.5 architecture)
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import json
from typing import List, Dict, Callable
from tqdm import tqdm
from scipy.fft import fft2
import time
import hashlib
import pickle
from pathlib import Path
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Feature Caching System
# ============================================================================

def get_feature_cache_key(dataset_name, layer_names, representation_name):
    """Generate unique cache key for features"""
    key_str = f"{dataset_name}_{representation_name}_{'_'.join(sorted(layer_names))}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_features_to_cache(cache_dir, cache_key, X_train, X_val, X_test):
    """Save computed features to disk"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_path / f"{cache_key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'X_train': X_train.cpu(),
            'X_val': X_val.cpu(),
            'X_test': X_test.cpu()
        }, f)
    print(f"  💾 Saved features to cache: {cache_file.name}")

def load_features_from_cache(cache_dir, cache_key):
    """Load computed features from disk"""
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✅ Loaded features from cache: {cache_file.name}")
        return data['X_train'].to(device), data['X_val'].to(device), data['X_test'].to(device)
    return None, None, None

# ============================================================================
# Checkpoint/Resume System
# ============================================================================

def save_checkpoint(output_dir, config_name, rep_name, results):
    """Save checkpoint for a completed representation"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f"{config_name}_{rep_name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Checkpoint saved: {checkpoint_file.name}")

def load_checkpoints(output_dir, config_name):
    """Load all existing checkpoints for a configuration"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return {}
    
    pattern = f"{config_name}_*.json"
    checkpoint_files = glob.glob(str(checkpoint_dir / pattern))
    
    checkpoints = {}
    for filepath in checkpoint_files:
        filename = Path(filepath).stem
        rep_name = filename.replace(f"{config_name}_", "")
        
        with open(filepath, 'r') as f:
            checkpoints[rep_name] = json.load(f)
    
    if checkpoints:
        print(f"\n🔄 Found {len(checkpoints)} existing checkpoints for {config_name}")
        for rep_name in checkpoints.keys():
            print(f"  ✅ {rep_name}")
    
    return checkpoints

def get_completed_representations(output_dir):
    """Get list of all completed representations from checkpoints"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return set()
    
    all_layers_completed = set()
    visual_only_completed = set()
    
    for filepath in glob.glob(str(checkpoint_dir / "all_layers_*.json")):
        filename = Path(filepath).stem
        rep_name = filename.replace("all_layers_", "")
        all_layers_completed.add(rep_name)
    
    for filepath in glob.glob(str(checkpoint_dir / "visual_only_*.json")):
        filename = Path(filepath).stem
        rep_name = filename.replace("visual_only_", "")
        visual_only_completed.add(rep_name)
    
    return all_layers_completed, visual_only_completed

# ============================================================================
# Model Architecture
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for classification"""
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.5):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Dataset Utilities
# ============================================================================

def load_and_split_dataset(dataset_name='1k'):
    """Load dataset from HuggingFace and split into train/val/test"""
    dataset_map = {
        '1k': 'jacekduszenko/lora-ws-1k',
        '10k': 'jacekduszenko/lora-ws-10k',
        '50k': 'jacekduszenko/lora-ws-50k',
        '100k': 'jacekduszenko/lora-ws-100k',
        'small': 'jacekduszenko/weights-dataset-small',
        'sanity-check': 'jacekduszenko/lora-ws-sanity-check'
    }

    with tqdm(total=3, desc="Loading dataset", unit="step") as pbar:
        if os.path.exists(dataset_name):
            pbar.set_postfix_str(f"Loading from local directory: {dataset_name}")
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_name)
        else:
            pbar.set_postfix_str(f"Downloading/loading {dataset_name} from HuggingFace")
            dataset = load_dataset(dataset_map[dataset_name])['train']
        
        pbar.set_postfix_str("Converting to torch format on GPU")
        dataset = dataset.with_format("torch", device=device)
        pbar.update(1)

        if len(dataset) <= 10:
            pbar.set_postfix_str(f"Small dataset ({len(dataset)} samples) - using all for train/val/test")
            training_set = dataset
            validation_set = dataset
            testing_set = dataset
            pbar.update(2)
        else:
            pbar.set_postfix_str("Splitting into train/temp (70%/30%)")
            initial_split = dataset.train_test_split(test_size=0.3, seed=42)
            pbar.update(1)
            
            pbar.set_postfix_str("Splitting temp into val/test (10%/20%)")
            validation_test_split = initial_split["test"].train_test_split(test_size=0.67, seed=42)
            pbar.update(1)

            training_set = initial_split["train"]
            validation_set = validation_test_split["train"]
            testing_set = validation_test_split["test"]

    return training_set, validation_set, testing_set


def get_lora_layers(dataset, layer_type='all'):
    """
    Extract list of LoRA layer keys from dataset

    Args:
        dataset: HuggingFace dataset
        layer_type: 'all', 'visual' (attn1 only), or 'text' (attn2 only)
    
    In Stable Diffusion U-Net:
    - attn1 = Self-Attention layers (visual/image processing) 
    - attn2 = Cross-Attention layers (text conditioning from CLIP)
    """
    all_layers = [key for key in dataset.features.keys()
                  if 'lora.down.weight' in key or 'lora.up.weight' in key]

    if layer_type == 'visual':
        visual_layers = sorted([l for l in all_layers if '.attn1.' in l])
        print(f"  → Filtering for VISUAL layers (self-attention): {len(visual_layers)} layers with '.attn1.'")
        return visual_layers
    elif layer_type == 'text':
        text_layers = sorted([l for l in all_layers if '.attn2.' in l])
        print(f"  → Filtering for TEXT layers (cross-attention): {len(text_layers)} layers with '.attn2.'")
        return text_layers
    else:
        print(f"  → Using ALL layers: {len(all_layers)} total")
        return sorted(all_layers)


# ============================================================================
# Representation Methods
# ============================================================================

def flat_vec_representation(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Flatten weight matrices into 1D vectors"""
    return layer_tensor.reshape(layer_tensor.size(0), -1)


def stats_representation(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute statistical features: mean, std, median, min, max, skewness, kurtosis
    """
    batch_size = layer_tensor.size(0)
    layer_flat = layer_tensor.reshape(batch_size, -1)
    
    means = layer_flat.mean(dim=1, keepdim=True)
    stds = layer_flat.std(dim=1, keepdim=True)
    medians = layer_flat.median(dim=1, keepdim=True).values
    mins = layer_flat.min(dim=1, keepdim=True).values
    maxs = layer_flat.max(dim=1, keepdim=True).values
    
    normalized = (layer_flat - means) / (stds + 1e-8)
    skewness = (normalized ** 3).mean(dim=1, keepdim=True)
    kurtosis = (normalized ** 4).mean(dim=1, keepdim=True)
    
    return torch.cat([means, stds, medians, mins, maxs, skewness, kurtosis], dim=1)


def spectral_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Extract spectral features using SVD"""
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i]

        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

            # Top-10 singular values
            top_k = 10
            if len(S) < top_k:
                S_padded = torch.zeros(top_k, device=matrix.device)
                S_padded[:len(S)] = S
                S = S_padded
            else:
                S = S[:top_k]

            # Normalized singular values
            S_norm = S / (S.sum() + 1e-8)

            # Effective rank
            entropy = -(S_norm * torch.log(S_norm + 1e-8)).sum()

            # Condition number
            cond_num = S[0] / (S[-1] + 1e-8)

            # Spectral gaps
            spectral_gaps = (S[:-1] - S[1:]) / (S[:-1] + 1e-8)

            # Nuclear norm
            nuclear_norm = S.sum()

            features = torch.cat([
                S, S_norm, spectral_gaps,
                torch.tensor([entropy, cond_num, nuclear_norm], device=matrix.device)
            ])

        except Exception as e:
            features = torch.zeros(10 + 10 + 9 + 3, device=matrix.device)

        features_list.append(features)

    return torch.stack(features_list)


def matrix_norm_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Extract various matrix norm features"""
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i]

        frob_norm = torch.linalg.norm(matrix, ord='fro')
        nuclear_norm = torch.linalg.norm(matrix, ord='nuc')
        spectral_norm = torch.linalg.norm(matrix, ord=2)
        l1_norm = torch.max(torch.sum(torch.abs(matrix), dim=0))
        linf_norm = torch.max(torch.sum(torch.abs(matrix), dim=1))
        nuclear_frob_ratio = nuclear_norm / (frob_norm + 1e-8)
        spectral_frob_ratio = spectral_norm / (frob_norm + 1e-8)

        features = torch.stack([
            frob_norm, nuclear_norm, spectral_norm,
            l1_norm, linf_norm,
            nuclear_frob_ratio, spectral_frob_ratio
        ])

        features_list.append(features)

    return torch.stack(features_list)


def distribution_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Extract higher-order distribution features - batch optimized"""
    batch_size = layer_tensor.size(0)
    layer_flat = layer_tensor.reshape(batch_size, -1)
    
    quantile_vals = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=layer_tensor.device)
    quantiles = torch.quantile(layer_flat, quantile_vals, dim=1).T
    
    iqr = (quantiles[:, 4] - quantiles[:, 2]).unsqueeze(1)
    mad = torch.abs(layer_flat - layer_flat.mean(dim=1, keepdim=True)).mean(dim=1, keepdim=True)
    means = layer_flat.mean(dim=1, keepdim=True)
    cv = layer_flat.std(dim=1, keepdim=True) / (means.abs() + 1e-8)
    
    stds = layer_flat.std(dim=1, keepdim=True)
    normalized = (layer_flat - means) / (stds + 1e-8)
    skewness = (normalized ** 3).mean(dim=1, keepdim=True)
    kurtosis = (normalized ** 4).mean(dim=1, keepdim=True)
    
    return torch.cat([quantiles, iqr, mad, cv, skewness, kurtosis], dim=1)


def frequency_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Extract frequency domain features using 2D FFT"""
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i].cpu().numpy()

        fft_matrix = fft2(matrix)
        magnitude = np.abs(fft_matrix)

        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        mag_max = np.max(magnitude)

        magnitude_flat = magnitude.flatten()
        magnitude_sorted = np.sort(magnitude_flat)[::-1]
        total_energy = np.sum(magnitude_flat ** 2)
        top_k_energy = np.sum(magnitude_sorted[:10] ** 2)
        energy_concentration = top_k_energy / (total_energy + 1e-8)

        top_k_freqs = magnitude_sorted[:10]

        features = torch.tensor(
            np.concatenate([
                top_k_freqs,
                [mag_mean, mag_std, mag_max, energy_concentration]
            ]),
            device=layer_tensor.device,
            dtype=torch.float32
        )

        features_list.append(features)

    return torch.stack(features_list)


def information_theoretic_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Extract information-theoretic features"""
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i].flatten()

        hist, _ = torch.histogram(matrix, bins=50)
        hist = hist.float()
        hist = hist / (hist.sum() + 1e-8)

        entropy = -(hist * torch.log(hist + 1e-8)).sum()
        normalized_entropy = entropy / np.log(50)

        features = torch.tensor([entropy, normalized_entropy], device=matrix.device)
        features_list.append(features)

    return torch.stack(features_list)


def simple_stats_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Simpler, faster statistical features - batch optimized"""
    batch_size = layer_tensor.size(0)
    layer_flat = layer_tensor.reshape(batch_size, -1)
    
    frob_norms = torch.linalg.norm(layer_flat, dim=1, keepdim=True)
    means = layer_flat.mean(dim=1, keepdim=True)
    stds = layer_flat.std(dim=1, keepdim=True)
    maxs = layer_flat.max(dim=1, keepdim=True).values
    mins = layer_flat.min(dim=1, keepdim=True).values
    
    return torch.cat([frob_norms, means, stds, maxs, mins], dim=1)


def rank_based_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Rank-based features - robust to outliers - batch optimized"""
    batch_size = layer_tensor.size(0)
    layer_flat = layer_tensor.reshape(batch_size, -1)
    
    percentile_vals = torch.tensor([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=layer_tensor.device)
    percentiles = torch.quantile(layer_flat, percentile_vals, dim=1).T
    
    iqr = (percentiles[:, 5] - percentiles[:, 3]).unsqueeze(1)
    range_val = (percentiles[:, 8] - percentiles[:, 0]).unsqueeze(1)
    
    return torch.cat([percentiles, iqr, range_val], dim=1)


def ensemble_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Combine multiple representation methods"""
    target_device = layer_tensor.device
    
    spectral_feats = spectral_features(layer_tensor).to(target_device)
    norm_feats = matrix_norm_features(layer_tensor).to(target_device)
    dist_feats = distribution_features(layer_tensor).to(target_device)
    freq_feats = frequency_features(layer_tensor).to(target_device)
    info_feats = information_theoretic_features(layer_tensor).to(target_device)

    features = torch.cat([
        spectral_feats, norm_feats, dist_feats,
        freq_feats, info_feats
    ], dim=1)

    return features


# ============================================================================
# Experiment Evaluation
# ============================================================================

def evaluate_representation(
    train_data, val_data, test_data,
    train_labels, val_labels, test_labels,
    lora_layers: List[str],
    representation_fn: Callable,
    num_classes: int,
    representation_name: str,
    num_runs: int = 10,
    cache_dir: str = None,
    dataset_name: str = None,
    dropout: float = 0.5,
    weight_decay: float = 1e-4,
    hidden_dim: int = 512,
    batch_size: int = 64
) -> Dict[str, any]:
    """
    Evaluate a representation method with feature caching and configurable regularization
    """
    print(f"\nEvaluating {representation_name} ({len(lora_layers)} layers)...")

    val_accs = []
    test_accs = []
    train_accs = []
    val_f1s = []
    test_f1s = []
    val_precisions = []
    test_precisions = []
    val_recalls = []
    test_recalls = []
    training_times = []
    best_epochs = []
    feature_dims = []
    val_confusion_matrices = []
    test_confusion_matrices = []
    per_run_results = []
    learning_curves = []

    # Try to load cached features
    X_train_cached, X_val_cached, X_test_cached = None, None, None
    if cache_dir and dataset_name:
        cache_key = get_feature_cache_key(dataset_name, lora_layers, representation_name)
        X_train_cached, X_val_cached, X_test_cached = load_features_from_cache(cache_dir, cache_key)
    
    if X_train_cached is not None:
        print(f"  🚀 Using cached features (skipping computation)")
        X_train_final = X_train_cached
        X_val_final = X_val_cached
        X_test_final = X_test_cached
        feature_dims.append(X_train_final.size(1))
    else:
        # Compute features once for all runs (directly from dataset, no intermediate caching)
        print(f"Computing features from {len(lora_layers)} layers...")
        train_features_list = []
        val_features_list = []
        test_features_list = []

        for layer_name in tqdm(lora_layers, desc=f"Computing features ({representation_name})", leave=False):
            train_features_list.append(representation_fn(train_data[layer_name][:]))
            val_features_list.append(representation_fn(val_data[layer_name][:]))
            test_features_list.append(representation_fn(test_data[layer_name][:]))

        # Concatenate features from all layers
        X_train_final = torch.cat(train_features_list, dim=1)
        X_val_final = torch.cat(val_features_list, dim=1)
        X_test_final = torch.cat(test_features_list, dim=1)
        
        feature_dims.append(X_train_final.size(1))
        
        # Save to cache if enabled
        if cache_dir and dataset_name and X_train_cached is None:
            save_features_to_cache(cache_dir, cache_key, X_train_final, X_val_final, X_test_final)

    # Now run multiple training runs with the same features
    for run in tqdm(range(num_runs), desc=f"Training runs ({representation_name})", leave=False):
        run_start_time = time.time()
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # Normalize (each run gets independent normalization)
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_final.cpu().numpy()), device=device, dtype=torch.float32)
        X_val = torch.tensor(scaler.transform(X_val_final.cpu().numpy()), device=device, dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test_final.cpu().numpy()), device=device, dtype=torch.float32)

        # Create data loaders
        train_dataset = TensorDataset(X_train, train_labels)
        val_dataset = TensorDataset(X_val, val_labels)
        test_dataset = TensorDataset(X_test, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create model with configurable parameters
        model = MLP(X_train.size(1), num_classes, hidden_dim=hidden_dim, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # Training loop (optimized for faster experiments)
        best_val_acc = 0
        patience_counter = 0
        epochs = 300
        patience = 15
        val_acc_history = []
        train_loss_history = []
        best_epoch = 0

        epoch_pbar = tqdm(range(epochs), desc=f"Epochs (run {run+1}/{num_runs})", leave=False)
        for epoch in epoch_pbar:
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            train_loss_history.append(avg_train_loss)

            # Validation
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())

            val_acc = accuracy_score(val_true, val_preds)
            val_acc_history.append(val_acc)
            scheduler.step(val_acc)
            
            epoch_pbar.set_postfix({'val_acc': f'{val_acc:.4f}', 'best': f'{best_val_acc:.4f}'})

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_pbar.close()
                    break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()
        
        run_end_time = time.time()
        training_time = run_end_time - run_start_time

        # Train metrics (for overfitting analysis)
        train_preds = []
        train_true = []
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(y_batch.cpu().numpy())
        
        train_acc = accuracy_score(train_true, train_preds)

        # Validation metrics
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        val_precision = precision_score(val_true, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_true, val_preds, average='macro', zero_division=0)
        val_conf_matrix = confusion_matrix(val_true, val_preds)

        # Test metrics
        test_preds = []
        test_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(y_batch.cpu().numpy())

        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
        test_precision = precision_score(test_true, test_preds, average='macro', zero_division=0)
        test_recall = recall_score(test_true, test_preds, average='macro', zero_division=0)
        test_conf_matrix = confusion_matrix(test_true, test_preds)

        # Store all metrics
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)
        val_precisions.append(val_precision)
        test_precisions.append(test_precision)
        val_recalls.append(val_recall)
        test_recalls.append(test_recall)
        training_times.append(training_time)
        best_epochs.append(best_epoch)
        val_confusion_matrices.append(val_conf_matrix.tolist())
        test_confusion_matrices.append(test_conf_matrix.tolist())
        
        learning_curves.append({
            'val_acc_history': val_acc_history,
            'train_loss_history': train_loss_history,
            'best_epoch': best_epoch
        })
        
        per_run_results.append({
            'run': run + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'val_precision': val_precision,
            'test_precision': test_precision,
            'val_recall': val_recall,
            'test_recall': test_recall,
            'training_time': training_time,
            'best_epoch': best_epoch,
            'feature_dim': X_train.size(1)
        })

        print(f"  Run {run+1}/{num_runs}: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}, Time={training_time:.2f}s")

    results = {
        'representation': representation_name,
        'num_layers': len(lora_layers),
        
        'train_acc_mean': np.mean(train_accs),
        'train_acc_std': np.std(train_accs),
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs),
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        
        'val_f1_mean': np.mean(val_f1s),
        'val_f1_std': np.std(val_f1s),
        'test_f1_mean': np.mean(test_f1s),
        'test_f1_std': np.std(test_f1s),
        
        'val_precision_mean': np.mean(val_precisions),
        'val_precision_std': np.std(val_precisions),
        'test_precision_mean': np.mean(test_precisions),
        'test_precision_std': np.std(test_precisions),
        
        'val_recall_mean': np.mean(val_recalls),
        'val_recall_std': np.std(val_recalls),
        'test_recall_mean': np.mean(test_recalls),
        'test_recall_std': np.std(test_recalls),
        
        'training_time_mean': np.mean(training_times),
        'training_time_std': np.std(training_times),
        'training_time_total': np.sum(training_times),
        
        'best_epoch_mean': np.mean(best_epochs),
        'best_epoch_std': np.std(best_epochs),
        
        'feature_dim': feature_dims[0] if feature_dims else 0,
        
        'per_run_results': per_run_results,
        'learning_curves': learning_curves,
        'val_confusion_matrices': val_confusion_matrices,
        'test_confusion_matrices': test_confusion_matrices,
        
        'train_acc_all_runs': train_accs,
        'val_acc_all_runs': val_accs,
        'test_acc_all_runs': test_accs,
    }

    print(f"Results: Train={results['train_acc_mean']:.4f}±{results['train_acc_std']:.4f}, "
          f"Val={results['val_acc_mean']:.4f}±{results['val_acc_std']:.4f}, "
          f"Test={results['test_acc_mean']:.4f}±{results['test_acc_std']:.4f}, "
          f"Time={results['training_time_mean']:.2f}±{results['training_time_std']:.2f}s")

    return results


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visual-only ablation experiment for LoRA representations')
    parser.add_argument('--dataset', type=str, default='1k',
                       help='Dataset to use: 1k, 10k, 50k, 100k, small, sanity-check, or a local path')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of training runs for averaging')
    parser.add_argument('--output-dir', type=str, default='visual-ablation-results',
                       help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer runs')
    parser.add_argument('--cache-features', action='store_true',
                       help='Cache computed features to disk for faster re-runs (10x speedup!)')
    parser.add_argument('--cache-dir', type=str, default='.feature_cache',
                       help='Directory to store cached features')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate for regularization (default: 0.5, higher = more regularization)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization) (default: 1e-4)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden layer dimension (default: 512, lower = less overfitting)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64, higher = better generalization)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'visual_ablation_{args.dataset}')
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_data, val_data, test_data = load_and_split_dataset(args.dataset)

    print("Extracting labels...")
    
    train_labels_raw = train_data['category_label'][:]
    val_labels_raw = val_data['category_label'][:]
    test_labels_raw = test_data['category_label'][:]
    
    if isinstance(train_labels_raw[0], str) or (hasattr(train_labels_raw[0], 'item') and isinstance(train_labels_raw[0].item(), str)):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        
        if torch.is_tensor(train_labels_raw):
            train_labels_list = [str(x.item() if hasattr(x, 'item') else x) for x in train_labels_raw.cpu()]
            val_labels_list = [str(x.item() if hasattr(x, 'item') else x) for x in val_labels_raw.cpu()]
            test_labels_list = [str(x.item() if hasattr(x, 'item') else x) for x in test_labels_raw.cpu()]
        else:
            train_labels_list = [str(x) for x in train_labels_raw]
            val_labels_list = [str(x) for x in val_labels_raw]
            test_labels_list = [str(x) for x in test_labels_raw]
        
        all_labels = train_labels_list + val_labels_list + test_labels_list
        label_encoder.fit(all_labels)
        
        train_labels = torch.tensor(label_encoder.transform(train_labels_list), device=device)
        val_labels = torch.tensor(label_encoder.transform(val_labels_list), device=device)
        test_labels = torch.tensor(label_encoder.transform(test_labels_list), device=device)
        
        print(f"Encoded string labels to integers")
        print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        if torch.is_tensor(train_labels_raw):
            train_labels = train_labels_raw.to(device)
            val_labels = val_labels_raw.to(device)
            test_labels = test_labels_raw.to(device)
        else:
            train_labels = torch.tensor(train_labels_raw, device=device)
            val_labels = torch.tensor(val_labels_raw, device=device)
            test_labels = torch.tensor(test_labels_raw, device=device)

    num_classes = len(set(train_labels.cpu().numpy()))
    print(f"Number of classes: {num_classes}")

    print("\n" + "="*80)
    print("IDENTIFYING LoRA LAYERS")
    print("="*80)
    all_layers = get_lora_layers(train_data, layer_type='all')
    visual_layers = get_lora_layers(train_data, layer_type='visual')
    text_layers = get_lora_layers(train_data, layer_type='text')

    print(f"\n📊 Layer Statistics:")
    print(f"  Total LoRA layers: {len(all_layers)}")
    print(f"  Visual layers (attn1 - self-attention): {len(visual_layers)}")
    print(f"  Text layers (attn2 - cross-attention): {len(text_layers)}")
    
    print(f"\n✅ Verification: {len(visual_layers)} + {len(text_layers)} = {len(visual_layers) + len(text_layers)} "
          f"(expected: {len(all_layers)})")
    
    if len(visual_layers) + len(text_layers) != len(all_layers):
        print("⚠️  WARNING: Layer counts don't match! Some layers may not be categorized correctly.")
    
    print(f"\n🔍 Sample visual layers (first 5):")
    for layer in visual_layers[:5]:
        print(f"  - {layer}")
    
    print(f"\n🔍 Sample text layers (first 5):")
    for layer in text_layers[:5]:
        print(f"  - {layer}")

    # Define representation methods to test
    representations = {
        'simple_stats': simple_stats_features,
        'rank_based': rank_based_features,
        'stats': stats_representation,
        'spectral': spectral_features,
        'matrix_norms': matrix_norm_features,
        'distribution': distribution_features,
        'frequency': frequency_features,
        'info_theoretic': information_theoretic_features,
        'ensemble': ensemble_features,
    }

    num_runs = 3 if args.quick else args.num_runs
    print(f"\nStarting experiments with {num_runs} runs per configuration...")

    # Load existing checkpoints (resume functionality)
    all_layers_checkpoints = load_checkpoints(output_path, "all_layers")
    visual_only_checkpoints = load_checkpoints(output_path, "visual_only")
    
    # Show progress summary
    total_reps = len(representations)
    completed_all_layers = len(all_layers_checkpoints)
    completed_visual_only = len(visual_only_checkpoints)
    remaining_all_layers = total_reps - completed_all_layers
    remaining_visual_only = total_reps - completed_visual_only
    
    print("\n" + "="*80)
    print("PROGRESS SUMMARY")
    print("="*80)
    print(f"All Layers Config:    {completed_all_layers}/{total_reps} completed, {remaining_all_layers} remaining")
    print(f"Visual-Only Config:   {completed_visual_only}/{total_reps} completed, {remaining_visual_only} remaining")
    print(f"Total Progress:       {completed_all_layers + completed_visual_only}/{total_reps * 2} ({(completed_all_layers + completed_visual_only) / (total_reps * 2) * 100:.1f}%)")
    
    if remaining_all_layers == 0 and remaining_visual_only == 0:
        print("\n✅ All experiments already completed! Loading results from checkpoints...")
    elif remaining_all_layers > 0 or remaining_visual_only > 0:
        print(f"\n🔄 Resuming from checkpoints. Will run {remaining_all_layers + remaining_visual_only} remaining representations...")

    # Experiment 1: ALL LAYERS (baseline)
    print("\n" + "="*80)
    print("EXPERIMENT 1: ALL LAYERS (attn1 + attn2)")
    print("="*80)

    all_layers_results = []
    for rep_name, rep_fn in tqdm(representations.items(), desc="Testing representations (all layers)"):
        full_name = f"{rep_name}_all_layers"
        
        if rep_name in all_layers_checkpoints:
            print(f"\n⏭️  Skipping {full_name} (already completed, loading from checkpoint)")
            all_layers_results.append(all_layers_checkpoints[rep_name])
            continue
        
        try:
            results = evaluate_representation(
                train_data, val_data, test_data,
                train_labels, val_labels, test_labels,
                all_layers,
                rep_fn,
                num_classes,
                full_name,
                num_runs=num_runs,
                cache_dir=args.cache_dir if args.cache_features else None,
                dataset_name=args.dataset,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size
            )
            all_layers_results.append(results)
            
            save_checkpoint(output_path, "all_layers", rep_name, results)
            
        except Exception as e:
            print(f"❌ Error evaluating {rep_name}: {e}")
            print(f"   Saving partial results and continuing...")
            import traceback
            traceback.print_exc()

    # Experiment 2: VISUAL-ONLY LAYERS
    print("\n" + "="*80)
    print("EXPERIMENT 2: VISUAL-ONLY LAYERS (attn1 only - NO TEXT CONDITIONING)")
    print("="*80)

    visual_only_results = []
    for rep_name, rep_fn in tqdm(representations.items(), desc="Testing representations (visual-only)"):
        full_name = f"{rep_name}_visual_only"
        
        if rep_name in visual_only_checkpoints:
            print(f"\n⏭️  Skipping {full_name} (already completed, loading from checkpoint)")
            visual_only_results.append(visual_only_checkpoints[rep_name])
            continue
        
        try:
            results = evaluate_representation(
                train_data, val_data, test_data,
                train_labels, val_labels, test_labels,
                visual_layers,
                rep_fn,
                num_classes,
                full_name,
                num_runs=num_runs,
                cache_dir=args.cache_dir if args.cache_features else None,
                dataset_name=args.dataset,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size
            )
            visual_only_results.append(results)
            
            save_checkpoint(output_path, "visual_only", rep_name, results)
            
        except Exception as e:
            print(f"❌ Error evaluating {rep_name}: {e}")
            print(f"   Saving partial results and continuing...")
            import traceback
            traceback.print_exc()

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    all_results_dict = {
        'dataset': args.dataset,
        'num_runs': num_runs,
        'num_classes': num_classes,
        'num_all_layers': len(all_layers),
        'num_visual_layers': len(visual_layers),
        'num_text_layers': len(text_layers),
        'all_layers_results': all_layers_results,
        'visual_only_results': visual_only_results
    }
    
    if isinstance(train_labels_raw[0], str):
        all_results_dict['label_mapping'] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    with open(os.path.join(output_path, 'results.json'), 'w') as f:
        json.dump(all_results_dict, f, indent=2)

    # Create comparison table
    with open(os.path.join(output_path, 'results.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Visual-Only Ablation Experiment Results - Dataset: {args.dataset}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Total layers: {len(all_layers)}\n")
        f.write(f"  Visual layers (attn1): {len(visual_layers)}\n")
        f.write(f"  Text layers (attn2): {len(text_layers)}\n")
        f.write(f"  Number of classes: {num_classes}\n")
        f.write(f"  Runs per experiment: {num_runs}\n")
        f.write("\n")

        f.write("-"*80 + "\n")
        f.write("Results with ALL LAYERS (attn1 + attn2):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Representation':<20} {'Val Acc':<20} {'Test Acc':<20} {'Test F1':<20}\n")
        for res in all_layers_results:
            f.write(f"{res['representation']:<20} "
                   f"{res['val_acc_mean']:.4f}±{res['val_acc_std']:.4f}     "
                   f"{res['test_acc_mean']:.4f}±{res['test_acc_std']:.4f}     "
                   f"{res['test_f1_mean']:.4f}±{res['test_f1_std']:.4f}\n")

        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("Results with VISUAL-ONLY LAYERS (attn1 only - NO TEXT):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Representation':<20} {'Val Acc':<20} {'Test Acc':<20} {'Test F1':<20}\n")
        for res in visual_only_results:
            f.write(f"{res['representation']:<20} "
                   f"{res['val_acc_mean']:.4f}±{res['val_acc_std']:.4f}     "
                   f"{res['test_acc_mean']:.4f}±{res['test_acc_std']:.4f}     "
                   f"{res['test_f1_mean']:.4f}±{res['test_f1_std']:.4f}\n")

        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("COMPARISON: Performance Drop (All Layers → Visual-Only)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Representation':<20} {'Test Acc Drop':<20} {'Relative Drop %':<20}\n")
        for all_res, vis_res in zip(all_layers_results, visual_only_results):
            drop = all_res['test_acc_mean'] - vis_res['test_acc_mean']
            rel_drop = (drop / all_res['test_acc_mean']) * 100 if all_res['test_acc_mean'] > 0 else 0
            rep_name = all_res['representation'].replace('_all_layers', '')
            f.write(f"{rep_name:<20} {drop:+.4f}              {rel_drop:+.2f}%\n")

    print(f"\nResults saved to {output_path}/")
    print("Experiment complete!")


if __name__ == '__main__':
    main()
