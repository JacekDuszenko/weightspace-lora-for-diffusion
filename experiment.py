import os
import sys
import argparse
from datasets import load_dataset, load_from_disk
from collections import Counter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, roc_auc_score

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['1k', '10k', '100k', 'small', '50k'], default='10k',
                   help='Dataset size to use (1k, 10k, 100k samples, or small subset)')
parser.add_argument('--flat-one', action='store_true', help='Run flat-one-layer experiment')
parser.add_argument('--flat-vec', action='store_true', help='Run flat-vec experiment')
parser.add_argument('--pca-flat-vec', action='store_true', help='Run PCA flat-vec experiment')
parser.add_argument('--stats-flat-vec', action='store_true', help='Run stats flat-vec experiment')
parser.add_argument('--pca-one', action='store_true', help='Run PCA one-layer experiment')
parser.add_argument('--stats-one', action='store_true', help='Run stats one-layer experiment')
parser.add_argument('--stats-concat', action='store_true', help='Run stats concat experiment')
parser.add_argument('--dense', action='store_true', help='Run dense experiment')
parser.add_argument('--num-experiments', type=int, default=10, help='Number of experiments to run')
parser.add_argument('--output-directory', type=str, default='paper-results', help='Output directory')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = '10k' if args.dataset == 'small' else args.dataset
try:
    print(f"Loading dataset from disk: ../datasets/ws-{dataset_name}")
    dataset = load_from_disk(f"../datasets/ws-{dataset_name}").with_format("torch", device=device)
    print("Loaded dataset from disk")
except:
    print('Load from disk failed, loading from HuggingFace - dataset name: jacekduszenko/lora-ws-{dataset_name}')
    dataset = load_dataset(f"jacekduszenko/lora-ws-{dataset_name}", trust_remote_code=True)
    dataset = dataset['train'].with_format("torch", device=device)


if args.dataset == 'small':
    categories = dataset['category_label']
    indices_by_category = {}
    for i, cat in enumerate(categories):
        if cat not in indices_by_category:
            indices_by_category[cat] = []
        indices_by_category[cat].append(i)
    
    selected_indices = []
    for cat_indices in indices_by_category.values():
        selected_indices.extend(cat_indices[:10])
    
    dataset = dataset.select(selected_indices)

initial_split = dataset.train_test_split(
    test_size=0.3, 
    seed=42
)
validation_test_split = initial_split["test"].train_test_split(
    test_size=0.67, 
    seed=42
)

training_set = initial_split["train"]
validation_set = validation_test_split["train"] 
testing_set = validation_test_split["test"]

print(f"Train size: {len(training_set)}")
print(f"Validation size: {len(validation_set)}")
print(f"Test size: {len(testing_set)}")
print("\nLabel distribution:")
for split_name, split_data in [("Train", training_set), ("Val", validation_set), ("Test", testing_set)]:
    labels = split_data["category_label"]
    counter = Counter(labels)
    print(f"\n{split_name} split distribution:")
    for label, count in counter.most_common():
        print(f"Label '{label}': {count} ({count/len(split_data):.2%})")


import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def get_lora_layers(dataset):
    lora_layers = [key for key in dataset.features.keys() 
                  if 'lora.down.weight' in key or 'lora.up.weight' in key]
    return lora_layers


def flat_vec_per_layer_representation(layer_tensor: torch.Tensor) -> torch.Tensor:
    return layer_tensor.reshape(layer_tensor.size(0), -1).to(device)


def stats_representation_fn(layer_tensor: torch.Tensor) -> torch.Tensor:
    layer_tensor = layer_tensor.reshape(layer_tensor.size(0), -1)
    stats_mean = torch.mean(layer_tensor, dim=1)
    stats_std = torch.std(layer_tensor, dim=1)
    stats_median = torch.median(layer_tensor, dim=1).values
    stats_min = torch.min(layer_tensor, dim=1).values
    stats_max = torch.max(layer_tensor, dim=1).values
    stats_skew = torch.mean((layer_tensor - stats_mean.unsqueeze(-1))**3, dim=1) / (stats_std**3)
    stats_kurt = torch.mean((layer_tensor - stats_mean.unsqueeze(-1))**4, dim=1) / (stats_std**4)
    
    return torch.cat([
        stats_mean.unsqueeze(1), 
        stats_std.unsqueeze(1),
        stats_median.unsqueeze(1),
        stats_min.unsqueeze(1), 
        stats_max.unsqueeze(1),
        stats_skew.unsqueeze(1),
        stats_kurt.unsqueeze(1)
    ], dim=1)
    

def eval_layer_by_layer(training_set, validation_set, testing_set, representation_fn, results_file, output_dir='paper-results', num_runs=10, with_pca=False, num_components=50):
    all_layers = get_lora_layers(dataset)
    unique_labels = sorted(list(set(training_set['category_label'])))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    all_val_accuracies = {layer: [] for layer in all_layers}
    all_test_accuracies = {layer: [] for layer in all_layers}
    best_models = []
    best_layers = []
    best_val_accuracies = []

    y_train = torch.tensor([label_to_int[label] for label in training_set['category_label']]).to(device)
    y_val = torch.tensor([label_to_int[label] for label in validation_set['category_label']]).to(device)
    y_test = torch.tensor([label_to_int[label] for label in testing_set['category_label']]).to(device)
    
    for run in range(num_runs):
        best_val_accuracy = 0
        best_layer_idx = 0
        best_classifier = None
        
        for layer_idx in tqdm(range(len(all_layers)), desc=f"Run {run + 1} - Evaluating layers"):
            layer = all_layers[layer_idx]
            
            if with_pca:
                from torch_pca import PCA
                pca = PCA(n_components=num_components)
                X_train = pca.fit_transform(training_set[layer].reshape(training_set[layer].shape[0], -1))
                X_val = pca.transform(validation_set[layer].reshape(validation_set[layer].shape[0], -1))
                X_test = pca.transform(testing_set[layer].reshape(testing_set[layer].shape[0], -1))
            else:
                X_train = representation_fn(training_set[layer])
                X_val =representation_fn(validation_set[layer])
                X_test = representation_fn(testing_set[layer]) 
                
            mean = X_train.mean(dim=0)
            std = X_train.std(dim=0)
            X_train_scaled = (X_train - mean) / std
            X_val_scaled = (X_val - mean) / std 
            X_test_scaled = (X_test - mean) / std
            
            model = MLP(X_train_scaled.shape[1], len(unique_labels)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            criterion = torch.nn.CrossEntropyLoss()
            
            X_train_tensor = X_train_scaled.to(device)
            X_val_tensor = X_val_scaled.to(device)
            y_train_tensor = y_train.to(device)
            y_val_tensor = y_val.to(device)
            X_test_tensor = X_test_scaled.to(device)
            y_test_tensor = y_test.to(device)
            
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            best_val_loss = float('inf')
            patience = 20
            patience_counter = 0
            num_epochs = 2000
            
            for epoch in range(num_epochs):
                model.train()
                epoch_train_loss = 0
                epoch_train_acc = 0
                num_batches = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == batch_y).float().mean().item()
                    
                    epoch_train_loss += loss.item()
                    epoch_train_acc += acc
                    num_batches += 1
                
                epoch_train_loss /= num_batches
                epoch_train_acc /= num_batches
                
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = (val_preds == y_val_tensor).float().mean().item()
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    print(f'Layer {layer_idx}, Epoch {epoch}: Train Loss = {epoch_train_loss:.4f}, '
                          f'Train Acc = {epoch_train_acc:.4f}, '
                          f'Val Loss = {val_loss:.4f}, '
                          f'Val Acc = {val_acc:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            all_val_accuracies[layer].append(val_acc)
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_preds = torch.argmax(test_outputs, dim=1)
                test_acc = (test_preds == y_test_tensor).float().mean().item()
            all_test_accuracies[layer].append(test_acc)
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_layer_idx = layer_idx
                best_classifier = model
        
        best_layer = all_layers[best_layer_idx]
        best_layers.append(best_layer)
        best_models.append(best_classifier)
        best_val_accuracies.append(best_val_accuracy)
    
    layer_stats = {}
    for layer in all_layers:
        val_accs = all_val_accuracies[layer]
        test_accs = all_test_accuracies[layer]
        layer_stats[layer] = {
            'val_mean': np.mean(val_accs) if len(val_accs) > 0 else 0,
            'val_std': np.std(val_accs) if len(val_accs) > 0 else 0,
            'test_mean': np.mean(test_accs) if len(test_accs) > 0 else 0,
            'test_std': np.std(test_accs) if len(test_accs) > 0 else 0
        }
    
    overall_best_layer = max(layer_stats.keys(), key=lambda x: layer_stats[x]['val_mean'])
    
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, results_file)
    with open(results_file + '.txt', 'w') as f:
        f.write(f"Results averaged over {num_runs} runs:\n\n")
        
        f.write("Layer-wise statistics:\n")
        for layer in all_layers:
            stats = layer_stats[layer]
            f.write(f"\n{layer}:\n")
            f.write(f"Validation: {stats['val_mean']:.4f} ± {stats['val_std']:.4f}\n")
            f.write(f"Test: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}\n")
        
        f.write(f"\nOverall Best Layer: {overall_best_layer}\n")
        f.write(f"Best Layer Statistics:\n")
        f.write(f"Validation: {layer_stats[overall_best_layer]['val_mean']:.4f} ± {layer_stats[overall_best_layer]['val_std']:.4f}\n")
        f.write(f"Test: {layer_stats[overall_best_layer]['test_mean']:.4f} ± {layer_stats[overall_best_layer]['test_std']:.4f}\n")
        
        f.write("\nBest layers per run:\n")
        for run_idx, (layer, val_acc) in enumerate(zip(best_layers, best_val_accuracies)):
            f.write(f"Run {run_idx + 1}: {layer} (val_acc: {val_acc:.4f})\n")
    
    return best_models, layer_stats, overall_best_layer


def create_tensor_split(split, split_name):
    all_layers = get_lora_layers(dataset)
    num_samples = len(split[all_layers[0]])
    features = None
    
    layers_dir = f"{args.dataset}_progress_layers_{split_name}"
    os.makedirs(layers_dir, exist_ok=True)
    
    processed_layers = {f.replace('.safetensors','') for f in os.listdir(layers_dir) if f.endswith('.safetensors')}
    if processed_layers:
        print(f"Found {len(processed_layers)} already processed layers for {split_name}")
        
        from safetensors.torch import load_file
        for layer in all_layers:
            if layer not in processed_layers:
                continue
            layer_path = os.path.join(layers_dir, f"{layer}.safetensors")
            layer_tensors = load_file(layer_path)
            layer_features = layer_tensors['features'].to(device)
            print('Loaded already processed intermediate layer', layer)
            if features is None:
                features = layer_features
            else:
                features = torch.cat([features, layer_features], dim=1)
    
    remaining_layers = [l for l in all_layers if l not in processed_layers]
    print('Processing remaining', len(remaining_layers), 'layers')
    for layer in tqdm(remaining_layers, desc=f"Processing layers for {split_name}"):
        try:
            layer_data = split[layer].to(device)
            layer_features = layer_data.reshape(num_samples, -1)
            
            from safetensors.torch import save_file
            layer_path = os.path.join(layers_dir, f"{layer}.safetensors")
            save_file({'features': layer_features}, layer_path)
            
            if features is None:
                features = layer_features
            else:
                features = torch.cat([features, layer_features], dim=1)
                
        except Exception as e:
            print(f"Error processing layer {layer}: {str(e)}")
            return features
            
    return features

def flat_vec(training_set, validation_set, testing_set, experiment_name, output_dir='paper-results', num_runs=10):
    os.makedirs(output_dir, exist_ok=True)
    all_layers = get_lora_layers(dataset)
    
    unique_labels = sorted(list(set(training_set['category_label'])))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    y_train = np.array([label_to_int[label] for label in training_set['category_label']])
    y_val = np.array([label_to_int[label] for label in validation_set['category_label']])
    y_test = np.array([label_to_int[label] for label in testing_set['category_label']])
    
    safetensors_path = f"flattened_weights_{args.dataset}.safetensors"
    
    try:
        from safetensors.torch import load_file
        tensors = load_file(safetensors_path)
        X_train_tensor = tensors["train"].to(device)
        X_val_tensor = tensors["val"].to(device)
        X_test_tensor = tensors["test"].to(device)
        print("Loaded tensors from safetensors file")
        print(f"X_train_tensor size: {X_train_tensor.element_size() * X_train_tensor.nelement() / (1024 ** 3):.2f} GB")
        print(f"X_val_tensor size: {X_val_tensor.element_size() * X_val_tensor.nelement() / (1024 ** 3):.2f} GB")
        print(f"X_test_tensor size: {X_test_tensor.element_size() * X_test_tensor.nelement() / (1024 ** 3):.2f} GB")
        
    except:
        print("Creating tensors from scratch...")
        from safetensors.torch import save_file
        X_train_tensor = create_tensor_split(training_set, "train").to(device)
        X_val_tensor = create_tensor_split(validation_set, "val").to(device)
        X_test_tensor = create_tensor_split(testing_set, "test").to(device)
        tensors_dict = {
            "train": X_train_tensor,
            "val": X_val_tensor,    
            "test": X_test_tensor
        }
        save_file(tensors_dict, safetensors_path)
    
    all_train_accs = []
    all_val_accs = []
    all_test_accs = []
    all_macro_precision = []
    all_macro_recall = []
    all_macro_f1 = []
    all_roc_auc = []

    for run in range(num_runs):
        mean = X_train_tensor.mean(dim=0, keepdim=True)
        std = X_train_tensor.std(dim=0, keepdim=True)
        
        X_train_tensor.sub_(mean).div_(std)
        X_val_tensor.sub_(mean).div_(std)
        X_test_tensor.sub_(mean).div_(std)

        y_train_tensor = torch.tensor(y_train, device=device)
        y_val_tensor = torch.tensor(y_val, device=device)
        y_test_tensor = torch.tensor(y_test, device=device)
        
        num_features = X_train_tensor.shape[1]
        num_classes = len(unique_labels)
        model = MLP(num_features, num_classes).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = 2000
        batch_size = 32
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in tqdm(range(num_epochs), desc='Training epochs'):
            model.train()
            epoch_train_loss = 0
            epoch_train_acc = 0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == batch_y).float().mean().item()
                
                epoch_train_loss += loss.item()
                epoch_train_acc += acc
                num_batches += 1
            
            epoch_train_loss /= num_batches
            epoch_train_acc /= num_batches
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            scheduler.step(val_loss)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss.item())
            train_accs.append(epoch_train_acc)
            val_accs.append(val_acc)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {epoch_train_loss:.4f}, '
                      f'Train Acc = {epoch_train_acc:.4f}, '
                      f'Val Loss = {val_loss:.4f}, '
                      f'Val Acc = {val_acc:.4f}')
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training and Validation Loss Over Time (Run {run + 1})')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(train_accs, label='Training Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Training and Validation Accuracy Over Time (Run {run + 1})')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'loss-acc-plot-{experiment_name}-run{run+1}.png')
        plt.savefig(plot_path)
        plt.close()
        
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
            train_acc = accuracy_score(y_train, train_preds)
            
            val_outputs = model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
            
            test_outputs = model(X_test_tensor)
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            test_probas = torch.softmax(test_outputs, dim=1).cpu().numpy()
            roc_auc = roc_auc_score(y_test, test_probas, multi_class='ovr')
            report = classification_report(y_test, test_preds, output_dict=True)
            test_acc = accuracy_score(y_test, test_preds)
            
            all_macro_precision.append(report['macro avg']['precision'])
            all_macro_recall.append(report['macro avg']['recall'])
            all_macro_f1.append(report['macro avg']['f1-score'])
            all_roc_auc.append(roc_auc)
            
            all_train_accs.append(train_acc)
            all_val_accs.append(val_acc)
            all_test_accs.append(test_acc)
    
    train_mean = np.mean(all_train_accs)
    train_std = np.std(all_train_accs)
    val_mean = np.mean(all_val_accs)
    val_std = np.std(all_val_accs)
    test_mean = np.mean(all_test_accs)
    test_std = np.std(all_test_accs)
    
    macro_precision_mean = np.mean(all_macro_precision)
    macro_precision_std = np.std(all_macro_precision)
    macro_recall_mean = np.mean(all_macro_recall)
    macro_recall_std = np.std(all_macro_recall)
    macro_f1_mean = np.mean(all_macro_f1)
    macro_f1_std = np.std(all_macro_f1)

    roc_auc_mean = np.mean(all_roc_auc)
    roc_auc_std = np.std(all_roc_auc)

    results_file = os.path.join(output_dir, f"{experiment_name}.txt")
    with open(results_file, 'a') as f:
        f.write("\nResults averaged over 10 runs:\n")
        f.write(f"Train accuracy: {train_mean:.4f} ± {train_std:.4f}\n")
        f.write(f"Validation accuracy: {val_mean:.4f} ± {val_std:.4f}\n")
        f.write(f"Test accuracy: {test_mean:.4f} ± {test_std:.4f}\n")
        f.write(f"Macro Precision: {macro_precision_mean:.4f} ± {macro_precision_std:.4f}\n")
        f.write(f"Macro Recall: {macro_recall_mean:.4f} ± {macro_recall_std:.4f}\n")
        f.write(f"Macro F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f}\n")
        f.write(f"ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}\n")
        
        f.write("\nIndividual run metrics:\n")
        for run in range(num_runs):
            f.write(f"\nRun {run + 1}:\n")
            f.write(f"Train accuracy: {all_train_accs[run]:.4f}\n")
            f.write(f"Validation accuracy: {all_val_accs[run]:.4f}\n")
            f.write(f"Test accuracy: {all_test_accs[run]:.4f}\n")
            f.write(f"Macro Precision: {all_macro_precision[run]:.4f}\n")
            f.write(f"Macro Recall: {all_macro_recall[run]:.4f}\n")
            f.write(f"Macro F1: {all_macro_f1[run]:.4f}\n")
            f.write(f"ROC AUC: {all_roc_auc[run]:.4f}\n")
    
    return model


def stats_concat(training_set, validation_set, testing_set, results_file, output_dir='paper-results', num_runs=10):
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = sorted(list(set(training_set['category_label'])))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    start_time = time.time()
    ts = training_set[0]
    lora_keys = [k for k in ts if 'lora.down.weight' in k or 'lora.up.weight' in k]
    layer_sizes = [ts[key].size(0) * ts[key].size(1) for key in lora_keys]
    print(f"Time to load first tensor: {time.time() - start_time:.2f}s")
    
    def process_dataset(flat_tensor: torch.Tensor):
        start_idx = 0
        """
        flat_tensor shape - (num_samples, num_features) - features are flattened weights of all layers
        Split flat_tensor into matrices based on layer_sizes and compute stats per matrix
        """
        start_idx = 0
        all_features = []
        
        for size in tqdm(layer_sizes, desc="Processing layers"):
            end_idx = start_idx + size
            layer_matrix = flat_tensor[:, start_idx:end_idx]
            means = torch.mean(layer_matrix, dim=1)
            stds = torch.std(layer_matrix, dim=1) 
            medians = torch.median(layer_matrix, dim=1).values
            mins = torch.min(layer_matrix, dim=1).values
            maxs = torch.max(layer_matrix, dim=1).values
            
            layer_stats = torch.stack([means, stds, medians, mins, maxs], dim=1)
            all_features.append(layer_stats)
            
            start_idx = end_idx
            
        return torch.cat(all_features, dim=1)
        
    try:
        from safetensors.torch import load_file
        tensors = load_file(f"flattened_weights_{args.dataset}.safetensors")
        print('Loaded tensors from safetensors file')
    except Exception as e:
        print(f"Failed to load tensors from flattened_weights_{args.dataset}.safetensors file: {e}")
        sys.exit(1)

    X_train_tensor = process_dataset(tensors["train"].to(device))
    X_val_tensor = process_dataset(tensors["val"].to(device))
    X_test_tensor = process_dataset(tensors["test"].to(device))
    
    mean = X_train_tensor.mean(dim=0, keepdim=True)
    std = X_train_tensor.std(dim=0, keepdim=True)
        
    X_train_tensor.sub_(mean).div_(std)
    X_val_tensor.sub_(mean).div_(std)
    X_test_tensor.sub_(mean).div_(std)
    
    y_train = training_set['category_label']
    y_val = validation_set['category_label']
    y_test = testing_set['category_label']
        
    mean_train_acc, mean_val_acc, mean_test_acc, var_train_acc, var_val_acc, var_test_acc, roc_auc_mean, roc_auc_std, macro_precision_mean, macro_precision_std, macro_recall_mean, macro_recall_std, macro_f1_mean, macro_f1_std = run_lr_all_layers(X_train_tensor, X_val_tensor, X_test_tensor, y_train, y_val, y_test, device, results_file, output_dir=output_dir, num_runs=num_runs)
    results_path = os.path.join(output_dir, results_file)
    with open(results_path, 'a') as f:
        f.write(f"\nStats concat Results:\n")
        f.write(f"Training accuracy: {mean_train_acc:.4f} ± {np.sqrt(var_train_acc):.4f}\n")
        f.write(f"Validation accuracy: {mean_val_acc:.4f} ± {np.sqrt(var_val_acc):.4f}\n") 
        f.write(f"Testing accuracy: {mean_test_acc:.4f} ± {np.sqrt(var_test_acc):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}\n")
        f.write(f"Macro Precision: {macro_precision_mean:.4f} ± {macro_precision_std:.4f}\n")
        f.write(f"Macro Recall: {macro_recall_mean:.4f} ± {macro_recall_std:.4f}\n")
        f.write(f"Macro F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f}\n")
    
            
def evaluate_pca_flat_vec(training_set, validation_set, testing_set, results_file, output_dir='paper-results', n_components=100, num_runs=10):
    results_file = f'pca-{n_components}' + results_file
    output_dir = output_dir + f'pca-{n_components}' 
    from safetensors.torch import load_file
    from pca_gpu import IncrementalPCAonGPU

    
    unique_labels = sorted(list(set(training_set['category_label'])))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    pca = IncrementalPCAonGPU(n_components=n_components, batch_size=1024)

    tensors = load_file(f"flattened_weights_{args.dataset}.safetensors")
    X_train_tensor = tensors["train"].to(device)
    X_val_tensor = tensors["val"].to(device)
    X_test_tensor = tensors["test"].to(device)

    y_train = training_set['category_label']
    y_val = validation_set['category_label']
    y_test = testing_set['category_label']
    
    mean = X_train_tensor.mean(dim=0, keepdim=True)
    std = X_train_tensor.std(dim=0, keepdim=True)

    X_train_tensor.sub_(mean).div_(std)
    X_val_tensor.sub_(mean).div_(std)
    X_test_tensor.sub_(mean).div_(std)
    
    X_train_pca = pca.fit_transform(X_train_tensor)
    X_val_pca = pca.transform(X_val_tensor)
    X_test_pca = pca.transform(X_test_tensor)
    
    mean_train_acc, mean_val_acc, mean_test_acc, var_train_acc, var_val_acc, var_test_acc, roc_auc_mean, roc_auc_std, macro_precision_mean, macro_precision_std, macro_recall_mean, macro_recall_std, macro_f1_mean, macro_f1_std = run_lr_all_layers(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, device, results_file, output_dir=output_dir, num_runs=num_runs)
    
    results_path = os.path.join(output_dir, results_file)
    with open(results_path, 'a') as f:
        f.write(f"\nPCA {n_components} Results:\n")
        f.write(f"Training accuracy: {mean_train_acc:.4f} ± {np.sqrt(var_train_acc):.4f}\n")
        f.write(f"Validation accuracy: {mean_val_acc:.4f} ± {np.sqrt(var_val_acc):.4f}\n") 
        f.write(f"Testing accuracy: {mean_test_acc:.4f} ± {np.sqrt(var_test_acc):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}\n")
        f.write(f"Macro Precision: {macro_precision_mean:.4f} ± {macro_precision_std:.4f}\n")
        f.write(f"Macro Recall: {macro_recall_mean:.4f} ± {macro_recall_std:.4f}\n")
        f.write(f"Macro F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f}\n")
    
    
def stats_flat_vec(training_set, validation_set, testing_set, results_file, output_dir='paper-results', num_runs=10):
    from safetensors.torch import load_file
    
    unique_labels = sorted(list(set(training_set['category_label'])))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    tensors = load_file(f"flattened_weights_{args.dataset}.safetensors")
    print('Loaded tensors from safetensors file')
    X_train_tensor = tensors["train"].to(device)
    X_val_tensor = tensors["val"].to(device)
    X_test_tensor = tensors["test"].to(device)
    
    def compute_stats(tensor):
        stats_mean = torch.mean(tensor, dim=1)
        stats_std = torch.std(tensor, dim=1)
        stats_median = torch.median(tensor, dim=1).values
        stats_min = torch.min(tensor, dim=1).values
        stats_max = torch.max(tensor, dim=1).values
        
        return torch.cat([
            stats_mean.unsqueeze(1), 
            stats_std.unsqueeze(1),
            stats_median.unsqueeze(1),
            stats_min.unsqueeze(1), 
            stats_max.unsqueeze(1),
        ], dim=1)
    
    mean = X_train_tensor.mean(dim=0, keepdim=True)
    std = X_train_tensor.std(dim=0, keepdim=True)
        
    X_train_tensor.sub_(mean).div_(std)
    X_val_tensor.sub_(mean).div_(std)
    X_test_tensor.sub_(mean).div_(std)
    
    y_train = training_set['category_label']
    y_val = validation_set['category_label']
    y_test = testing_set['category_label']        
    
    mean_train_acc, mean_val_acc, mean_test_acc, var_train_acc, var_val_acc, var_test_acc, roc_auc_mean, roc_auc_std, macro_precision_mean, macro_precision_std, macro_recall_mean, macro_recall_std, macro_f1_mean, macro_f1_std = run_lr_all_layers(X_train_tensor, X_val_tensor, X_test_tensor, y_train, y_val, y_test, device, results_file, output_dir=output_dir, num_runs=num_runs)
    
    results_path = os.path.join(output_dir, results_file)
    with open(results_path, 'a') as f:
        f.write(f"\nStats Results:\n")
        f.write(f"Training accuracy: {mean_train_acc:.4f} ± {np.sqrt(var_train_acc):.4f}\n")
        f.write(f"Validation accuracy: {mean_val_acc:.4f} ± {np.sqrt(var_val_acc):.4f}\n") 
        f.write(f"Testing accuracy: {mean_test_acc:.4f} ± {np.sqrt(var_test_acc):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}\n")
        f.write(f"Macro Precision: {macro_precision_mean:.4f} ± {macro_precision_std:.4f}\n")
        f.write(f"Macro Recall: {macro_recall_mean:.4f} ± {macro_recall_std:.4f}\n")
        f.write(f"Macro F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f}\n")
        
        
def dense_stats(training_set, validation_set, testing_set, results_file, output_dir='paper-results', num_runs=10):
    from safetensors.torch import load_file
    unique_labels = sorted(list(set(training_set['category_label'])))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    ts = training_set[0]
    lora_keys = [k for k in ts if 'lora.down.weight' in k or 'lora.up.weight' in k]
    layer_sizes = [ts[key].size(0) * ts[key].size(1) for key in lora_keys]
    
    try:
        tensors = load_file(f"flattened_weights_{args.dataset}.safetensors")
        print('Loaded tensors from safetensors file')
    except Exception as e:
        print(f'Failed to load flattened weights. exiting... {e}')
        sys.exit(1)
        
    def make_dense_stats(tensor):
        all_features = []
        start_idx = 0
        batch_size = 1024
        
        for size_down_idx in tqdm(range(0, len(layer_sizes), 2), desc="Creating dense stats from layers"):
            size_down = layer_sizes[size_down_idx]
            size_up = layer_sizes[size_down_idx + 1]
            
            batch_features = []
            for batch_start in range(0, tensor.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, tensor.shape[0])
                batch_tensor = tensor[batch_start:batch_end]
                
                layer_matrix_down = batch_tensor[:, start_idx:start_idx+size_down]
                layer_matrix_up = batch_tensor[:, start_idx+size_down:start_idx+size_down+size_up]
                
                dense_flat = torch.einsum('bf,bk->bfk', layer_matrix_down, layer_matrix_up).flatten(start_dim=1)
                stats_mean = torch.mean(dense_flat, dim=1)
                stats_std = torch.std(dense_flat, dim=1)
                stats_median = torch.median(dense_flat, dim=1).values
                stats_min = torch.min(dense_flat, dim=1).values
                stats_max = torch.max(dense_flat, dim=1).values
                
                batch_stats = torch.cat([
                    stats_mean.unsqueeze(1),
                    stats_std.unsqueeze(1),
                    stats_median.unsqueeze(1),
                    stats_min.unsqueeze(1),
                    stats_max.unsqueeze(1),
                ], dim=1)
                
                batch_features.append(batch_stats)
                
                del dense_flat, layer_matrix_down, layer_matrix_up, batch_stats
                torch.cuda.empty_cache()
            
            layer_features = torch.cat(batch_features, dim=0)
            all_features.append(layer_features)
            start_idx += size_down + size_up
        
        return torch.cat(all_features, dim=1)
        
    X_train_tensor = make_dense_stats(tensors["train"].to(device)[:5000])
    X_val_tensor = make_dense_stats(tensors["val"].to(device)[:5000])
    X_test_tensor = make_dense_stats(tensors["test"].to(device)[:5000])
        
    y_train = training_set['category_label'][:5000]
    y_val = validation_set['category_label'][:5000]
    y_test = testing_set['category_label'][:5000]    
    
        
    mean = X_train_tensor.mean(dim=0, keepdim=True)
    std = X_train_tensor.std(dim=0, keepdim=True)
        
    X_train_tensor.sub_(mean).div_(std)
    X_val_tensor.sub_(mean).div_(std)
    X_test_tensor.sub_(mean).div_(std)
    
    mean_train_acc, mean_val_acc, mean_test_acc, var_train_acc, var_val_acc, var_test_acc, roc_auc_mean, roc_auc_std, macro_precision_mean, macro_precision_std, macro_recall_mean, macro_recall_std, macro_f1_mean, macro_f1_std = run_lr_all_layers(X_train_tensor, X_val_tensor, X_test_tensor, y_train, y_val, y_test, device, results_file, output_dir=output_dir, num_runs=num_runs)
    
    results_path = os.path.join(output_dir, results_file)
    with open(results_path, 'a') as f:
        f.write(f"\nDense Stats Results:\n")
        f.write(f"Training accuracy: {mean_train_acc:.4f} ± {np.sqrt(var_train_acc):.4f}\n")
        f.write(f"Validation accuracy: {mean_val_acc:.4f} ± {np.sqrt(var_val_acc):.4f}\n") 
        f.write(f"Testing accuracy: {mean_test_acc:.4f} ± {np.sqrt(var_test_acc):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}\n")
        f.write(f"Macro Precision: {macro_precision_mean:.4f} ± {macro_precision_std:.4f}\n")
        f.write(f"Macro Recall: {macro_recall_mean:.4f} ± {macro_recall_std:.4f}\n")
        f.write(f"Macro F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f}\n")
    
    
def run_lr_all_layers(X_train, X_val, X_test, y_train, y_val, y_test, device, experiment_name, output_dir='paper-results', num_runs=10):
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = sorted(list(set(y_train)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    num_features = X_train.shape[1]
    num_classes = len(unique_labels)
    
    y_train = torch.tensor([label_to_int[label] for label in y_train]).to(device)
    y_val = torch.tensor([label_to_int[label] for label in y_val]).to(device)
    y_test = torch.tensor([label_to_int[label] for label in y_test]).to(device)
    
    all_train_accs = []
    all_val_accs = []
    all_test_accs = []
    all_macro_precision = []
    all_macro_recall = []
    all_macro_f1 = []
    all_roc_auc = []
    
    for run in tqdm(range(num_runs), desc="Running LR all layers"):
        model = MLP(num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = 1000
        batch_size = 32
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            epoch_train_acc = 0
            num_batches = 0
            
            debug_log_path = os.path.join(output_dir, f'debug_log_run_{run}.txt')
            with open(debug_log_path, 'a') as debug_f:
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == batch_y).float().mean().item()
                    
                    epoch_train_loss += loss.item()
                    epoch_train_acc += acc
                    num_batches += 1

                    debug_f.write(f"\nEpoch {epoch} Batch {batch_idx}:\n")
                    debug_f.write(f"batch_X shape: {batch_X.shape}, values: {batch_X[:5]}\n")
                    debug_f.write(f"batch_y: {batch_y}\n")
                    debug_f.write(f"loss: {loss.item()}\n")
                    debug_f.write(f"preds: {preds}\n")
                    debug_f.write(f"batch accuracy: {acc}\n")
                
                epoch_train_loss /= num_batches
                epoch_train_acc /= num_batches

                debug_f.write(f"\nEpoch {epoch} Training Summary:\n")
                debug_f.write(f"epoch_train_loss: {epoch_train_loss}\n")
                debug_f.write(f"epoch_train_acc: {epoch_train_acc}\n")
                
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = (val_preds == y_val).float().mean().item()

                    debug_f.write(f"\nEpoch {epoch} Validation:\n")
                    debug_f.write(f"val_outputs shape: {val_outputs.shape}, values: {val_outputs[:5]}\n")
                    debug_f.write(f"val_loss: {val_loss.item()}\n")
                    debug_f.write(f"y_val: {y_val}\n")
                    debug_f.write(f"val_preds: {val_preds}\n")
                    debug_f.write(f"val_acc: {val_acc}\n")
                    debug_f.write("-"*50 + "\n")
            
            scheduler.step(val_loss)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss.item())
            train_accs.append(epoch_train_acc)
            val_accs.append(val_acc)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {epoch_train_loss:.4f}, '
                      f'Train Acc = {epoch_train_acc:.4f}, '
                      f'Val Loss = {val_loss:.4f}, '
                      f'Val Acc = {val_acc:.4f}')
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training and Validation Loss Over Time (Run {run + 1})')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(train_accs, label='Training Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Training and Validation Accuracy Over Time (Run {run + 1})')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'loss-acc-plot-{experiment_name}-run{run+1}.png')
        plt.savefig(plot_path)
        plt.close()
        
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
            train_acc = accuracy_score(y_train.cpu(), train_preds)
            
            val_outputs = model(X_val)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu(), val_preds)
            
            test_outputs = model(X_test)
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            test_probas = torch.softmax(test_outputs, dim=1).cpu().numpy()
            roc_auc = roc_auc_score(y_test.cpu().numpy(), test_probas, multi_class='ovr')
            report = classification_report(y_test.cpu().numpy(), test_preds, output_dict=True)
            test_acc = accuracy_score(y_test.cpu().numpy(), test_preds)
            
            all_macro_precision.append(report['macro avg']['precision'])
            all_macro_recall.append(report['macro avg']['recall'])
            all_macro_f1.append(report['macro avg']['f1-score'])
            all_roc_auc.append(roc_auc)
            
            all_train_accs.append(train_acc)
            all_val_accs.append(val_acc)
            all_test_accs.append(test_acc)
    
    mean_train_acc = np.mean(all_train_accs)
    mean_val_acc = np.mean(all_val_accs)
    mean_test_acc = np.mean(all_test_accs)
    
    var_train_acc = np.var(all_train_accs)
    var_val_acc = np.var(all_val_accs)
    var_test_acc = np.var(all_test_accs)
    
    roc_auc_mean = np.mean(all_roc_auc)
    roc_auc_std = np.std(all_roc_auc)
    macro_precision_mean = np.mean(all_macro_precision)
    macro_precision_std = np.std(all_macro_precision)
    macro_recall_mean = np.mean(all_macro_recall)
    macro_recall_std = np.std(all_macro_recall)
    macro_f1_mean = np.mean(all_macro_f1)
    macro_f1_std = np.std(all_macro_f1)

    return mean_train_acc, mean_val_acc, mean_test_acc, var_train_acc, var_val_acc, var_test_acc, roc_auc_mean, roc_auc_std, macro_precision_mean, macro_precision_std, macro_recall_mean, macro_recall_std, macro_f1_mean, macro_f1_std


# ONLY ON 1k DATASET
if args.flat_one:
    eval_layer_by_layer(training_set, validation_set, testing_set, 
                       flat_vec_per_layer_representation,
                       f"flat-one-layer-{args.dataset}", 
                       output_dir=f"{args.output_directory}/{args.dataset}/flat-one-layer-{args.dataset}", 
                       num_runs=args.num_experiments)

# DONE
if args.flat_vec:
    flat_vec(training_set, validation_set, testing_set,
             f"flat-vec-{args.dataset}",
             output_dir=f"{args.output_directory}/{args.dataset}/flat-vec-{args.dataset}",
             num_runs=args.num_experiments)

# DONE
if args.pca_flat_vec:
    for components in tqdm([1000], desc="PCA components"):
        evaluate_pca_flat_vec(training_set, validation_set, testing_set,
                            f"flat-vec-{args.dataset}",
                            output_dir=f"{args.output_directory}/{args.dataset}/flat-vec-{args.dataset}-pca-{components}",
                            n_components=components,
                            num_runs=args.num_experiments)

# DONE
if args.stats_flat_vec:
    stats_flat_vec(training_set, validation_set, testing_set,
                  f"stats-flat-vec-{args.dataset}",
                  output_dir=f"{args.output_directory}/{args.dataset}/stats-flat-vec-{args.dataset}",
                  num_runs=args.num_experiments)
    
# DONE
if args.stats_concat:
    stats_concat(training_set, validation_set, testing_set,
                f"stats-concat-{args.dataset}",
                output_dir=f"{args.output_directory}/{args.dataset}/stats-concat-{args.dataset}",
                num_runs=args.num_experiments)
    
# DONE
if args.dense:
    dense_stats(training_set, validation_set, testing_set,
          f"dense-stats-{args.dataset}",
          output_dir=f"{args.output_directory}/{args.dataset}/dense-stats-{args.dataset}",
          num_runs=args.num_experiments)

# NOT IN EXPERIMENT
if args.pca_one:
    eval_layer_by_layer(training_set, validation_set, testing_set,
                       None,
                       f"pca-{100}-one-layer-{args.dataset}",
                       with_pca=True,
                       num_components=100,
                       output_dir=f"{args.output_directory}/{args.dataset}/pca-{100}-one-layer-{args.dataset}",
                       num_runs=args.num_experiments)
# NOT IN EXPERIMENT
if args.stats_one:
    eval_layer_by_layer(training_set, validation_set, testing_set,
                       stats_representation_fn,
                       f"stats-one-layer-{args.dataset}",
                       output_dir=f"{args.output_directory}/{args.dataset}/stats-one-layer-{args.dataset}",
                       num_runs=args.num_experiments)
    
