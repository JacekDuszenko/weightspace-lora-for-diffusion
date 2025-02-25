import os
import argparse
import json
import glob
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from safetensors import safe_open
from tqdm import tqdm
import tempfile
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--num-per-category', type=int, default=1000,
                   help='Number of examples to take from each category')
parser.add_argument('--batch-size', type=int, default=128,
                   help='Number of files to process in each batch')
args = parser.parse_args()

categories = ['airplane', 'bird', 'cat', 'car', 'dog', 'fish', 'fruit', 'ship', 'snake', 'vegetable']


def load_metadata(path):
    with open(path) as f:
        return json.load(f)

def process_category_batch(category_folders, batch_start, batch_size):
    batch_data = {
        'folder': [],
        'category_label': [],
        'category_name': [],
        'num_images': [],
        'leaf_id': [],
        'num_batches': [],
        'num_epochs': [],
        'batch_size': [],
        'gradient_accumulation_steps': [],
        'max_train_steps': [],
        'rank': [],
        'step_loss': []
    }
    
    lora_keys = None
    
    batch_folders = category_folders[batch_start:batch_start + batch_size]
    
    for folder in tqdm(batch_folders, desc=f"Processing batch {batch_start//batch_size + 1}", leave=False):
        folder_path = Path(folder)
        metadata_path = folder_path / "metadata.json"
        safetensors_path = folder_path / "pytorch_lora_weights.safetensors"
        
        if not metadata_path.exists() or not safetensors_path.exists():
            continue
            
        try:
            metadata = load_metadata(metadata_path)
            
            for key in tqdm(batch_data.keys(), desc="Processing metadata fields", leave=False):
                if key in metadata:
                    batch_data[key].append(metadata[key])
            batch_data['folder'].append(str(folder_path))
            
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                if lora_keys is None:
                    lora_keys = list(f.keys())
                    for key in tqdm(lora_keys, desc="Initializing LoRA keys", leave=False):
                        batch_data[key] = []
                
                for key in tqdm(lora_keys, desc="Processing weight tensors", leave=False):
                    tensor = f.get_tensor(key)
                    batch_data[key].append(tensor.numpy())
                    
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            continue
            
    return Dataset.from_dict(batch_data)

def create_category_dataset(category_paths, num_examples, batch_size, output_path):
    all_folders = []
    for path in category_paths:
        folders = sorted(glob.glob(str(path / "folder_*")))
        valid_folders = [f for f in folders if Path(f).joinpath("pytorch_lora_weights.safetensors").exists()]
        all_folders.extend(valid_folders)
    import sys
    all_folders = all_folders[:num_examples]
    print('all folders has length ', len(all_folders))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        intermediate_datasets = []
        
        for batch_start in tqdm(range(0, len(all_folders), batch_size), desc="Processing batches", leave=False):
            batch_dataset = process_category_batch(all_folders, batch_start, batch_size)
            
            temp_batch_path = os.path.join(temp_dir, f"batch_{batch_start}")
            batch_dataset.save_to_disk(temp_batch_path)
            intermediate_datasets.append(temp_batch_path)
            
            del batch_dataset
            
        print("Combining all batches...")
        datasets_to_combine = [Dataset.load_from_disk(path) for path in intermediate_datasets]
        combined_dataset = concatenate_datasets(datasets_to_combine)
        
        print(f"Saving category dataset to {output_path}")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        combined_dataset.save_to_disk(output_path)
        
        del datasets_to_combine
        del combined_dataset

print("Creating dataset...")
base_path = Path(os.environ["SCRATCH"]) / "ws/lora-adapters-are-good-feature-extractors/data/weightspace-images"
plg_base_path = Path("/net/pr2/projects/plgrid/plggweightspace/data")
output_path = os.path.join(os.environ["PLG_GROUPS_STORAGE"], "plggweightspace/datasets/ws-100k")

for category in tqdm(categories, desc="Processing categories"):
    category_paths = [
        base_path / category,
        plg_base_path / category
    ]
    category_output = output_path + f"_{category}"
    
    create_category_dataset(
        category_paths=category_paths,
        num_examples=args.num_per_category,
        batch_size=args.batch_size,
        output_path=category_output
    )
    
category_datasets = []
for category in tqdm(categories, desc="Loading category datasets"):
    dataset = Dataset.load_from_disk(output_path + f"_{category}")
    category_datasets.append(dataset)

print("Concatenating final dataset...")
final_dataset = concatenate_datasets(category_datasets)

if os.path.exists(output_path):
    shutil.rmtree(output_path)
final_dataset.save_to_disk(output_path)

print(f"Dataset saved to {output_path}")
print(f"Total samples: {len(final_dataset)}")
