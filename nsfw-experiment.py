from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
    
    def forward(self, x):
        return self.network(x)

def train_and_evaluate(model, X_train, y_train, X_val, y_val, device, num_runs=10):
    all_metrics = []
    
    for run in range(num_runs):
        shuffle_idx = torch.randperm(len(X_train))
        X_train_shuffled = X_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]
        
        model = MLP(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = torch.utils.data.TensorDataset(X_train_shuffled, y_train_shuffled)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(2000):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_shuffled)
            train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
            train_probs = train_outputs[:, 1].cpu().numpy()
            
            val_outputs = model(X_val)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_probs = val_outputs[:, 1].cpu().numpy()
            
        metrics = {
            'train_accuracy': accuracy_score(y_train_shuffled.cpu(), train_preds),
            'train_f1': f1_score(y_train_shuffled.cpu(), train_preds),
            'train_roc_auc': roc_auc_score(y_train_shuffled.cpu(), train_probs),
            'val_accuracy': accuracy_score(y_val.cpu(), val_preds),
            'val_f1': f1_score(y_val.cpu(), val_preds),
            'val_roc_auc': roc_auc_score(y_val.cpu(), val_probs)
        }
        all_metrics.append(metrics)
        print(f"Run {run+1} completed")
    
    return all_metrics

print("Loading dataset...")
dataset = load_dataset("jacekduszenko/weights-dataset-small")

X_train = np.array(dataset['train']['weights'])
X_val = np.array(dataset['test']['weights'])

label_map = {'neutral': 0, 'hentai': 1}
y_train_orig = np.array([label_map[label] for label in dataset['train']['label']])
y_val = torch.tensor([label_map[label] for label in dataset['test']['label']])

split_list = [
    320, 320, 320, 320, 320, 320, 768,  
    320, 320, 320, 320, 320, 320, 768, 
    320, 640, 640, 640, 640, 640, 640, 768,  
    640, 640, 640, 640, 640, 640, 768, 
    640, 1280, 1280, 1280, 1280, 1280, 1280, 768, 
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 768, 
    1280, 1280, 1280, 1280, 1280, 1280, 768,  
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 768,  
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 768, 
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 768,  
    1280, 640, 640, 640, 640, 640, 640, 768,  
    640, 640, 640, 640, 640, 640, 640, 768,  
    640, 640, 640, 640, 640, 640, 640, 768,  
    640, 320, 320, 320, 320, 320, 320, 768,  
    320, 320, 320, 320, 320, 320, 320, 768,  
    320, 320, 320, 320, 320, 320, 320, 768, 
    320
]

start = 0
for i, size in enumerate(split_list[:80]):
    start += size
end = start + split_list[80]

print(f"\nExtracting layer 80 (boundaries: {start}:{end})...")

# Extract layer 80 data - best layer so far.
X_train_layer = torch.tensor(X_train[:, start:end], device=device).float()
X_val_layer = torch.tensor(X_val[:, start:end], device=device).float()

scaler = StandardScaler()
X_train_proc = torch.tensor(scaler.fit_transform(X_train_layer.cpu()), device=device).float()
X_val_proc = torch.tensor(scaler.transform(X_val_layer.cpu()), device=device).float()

os.makedirs("results", exist_ok=True)

experiments = ['flat_vector', 'dense_stats', 'stats_concat', 'flat_vector_stats', 'pca_200']

for exp_name in experiments:
    print(f"\nRunning {exp_name} experiment on layer 80...")
    
    if exp_name == 'flat_vector':
        X_train_proc = torch.tensor(X_train_layer, device=device).float()
        X_val_proc = torch.tensor(X_val_layer, device=device).float()
    
    elif exp_name == 'dense_stats':
        def compute_stats(X):
            X = torch.tensor(X, device=device).float()
            stats_mean = torch.mean(X, dim=1)
            stats_std = torch.std(X, dim=1)
            stats_median = torch.median(X, dim=1).values
            stats_min = torch.min(X, dim=1).values
            stats_max = torch.max(X, dim=1).values
            return torch.stack([stats_mean, stats_std, stats_median, stats_min, stats_max], dim=1)
            
        X_train_proc = compute_stats(X_train_layer)
        X_val_proc = compute_stats(X_val_layer)
    
    elif exp_name == 'stats_concat':
        def compute_layer_stats(X):
            X = torch.tensor(X, device=device).float()
            stats = torch.cat([
                torch.mean(X, dim=1, keepdim=True),
                torch.std(X, dim=1, keepdim=True),
                torch.median(X, dim=1, keepdim=True).values,
                torch.min(X, dim=1, keepdim=True).values,
                torch.max(X, dim=1, keepdim=True).values
            ], dim=1)
            return stats
            
        X_train_proc = compute_layer_stats(X_train_layer)
        X_val_proc = compute_layer_stats(X_val_layer)
    
    elif exp_name == 'flat_vector_stats':
        scaler = StandardScaler()
        X_train_proc = torch.tensor(scaler.fit_transform(X_train_layer.cpu()), device=device).float()
        X_val_proc = torch.tensor(scaler.transform(X_val_layer.cpu()), device=device).float()
    
    elif exp_name == 'pca_200':
        pca = PCA(n_components=20)
        X_train_proc = torch.tensor(pca.fit_transform(X_train_layer.cpu()), device=device).float()
        X_val_proc = torch.tensor(pca.transform(X_val_layer.cpu()), device=device).float()

    scaler = StandardScaler()
    X_train_proc = torch.tensor(scaler.fit_transform(X_train_proc.cpu()), device=device).float()
    X_val_proc = torch.tensor(scaler.transform(X_val_proc.cpu()), device=device).float()

    metrics = train_and_evaluate(
        MLP(X_train_proc.shape[1]), 
        X_train_proc,
        torch.tensor(y_train_orig, device=device),
        X_val_proc, 
        y_val.to(device),
        device,
        num_runs=10
    )

    mean_metrics = {}
    std_metrics = {}
    for key in metrics[0].keys():
        values = [m[key] for m in metrics]
        mean_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    print(f"\n{exp_name} Results Summary (mean ± std over 10 runs)")
    print(f"Train Accuracy: {mean_metrics['train_accuracy']:.4f} ± {std_metrics['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {mean_metrics['val_accuracy']:.4f} ± {std_metrics['val_accuracy']:.4f}")
    print(f"ROC AUC: {mean_metrics['val_roc_auc']:.4f} ± {std_metrics['val_roc_auc']:.4f}")
    print(f"Val F1 Score: {mean_metrics['val_f1']:.4f} ± {std_metrics['val_f1']:.4f}")
