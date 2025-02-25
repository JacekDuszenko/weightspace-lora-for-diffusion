import argparse
from datasets import load_dataset
import os
import random
from collections import defaultdict
import json

def download_and_prepare_dataset(dataset_name, token, cache_dir, num_images_per_group=5):
    """Download dataset and split images into groups by label"""
    print(f"Downloading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", token=token, cache_dir=cache_dir)
    
    label_groups = defaultdict(list)
    for idx in range(len(dataset)):
        label = dataset[idx]['label']
        label_groups[label].append(idx)
    
    groups = []
    for label, indices in label_groups.items():
        random.shuffle(indices)
        
        for i in range(0, len(indices), num_images_per_group):
            group = indices[i:i + num_images_per_group]
            if len(group) == num_images_per_group:
                groups.append((label, group))
    
    print(f"Created {len(groups)} groups across {len(label_groups)} labels")
    for label in label_groups:
        print(f"Label {label} has {len(label_groups[label])} images")
    
    return dataset, groups

def prepare_training_directories(dataset, groups, base_dir):
    """Create directories and save images for each group"""
    group_dirs = []
    
    for group_idx, (label, group) in enumerate(groups):
        group_dir = os.path.join(base_dir, f"group_{group_idx}_label_{label}")
        os.makedirs(group_dir, exist_ok=True)
        group_dirs.append(group_dir)
        
        for idx in group:
            sample = dataset[idx]
            image = sample['image']
            image_path = os.path.join(group_dir, f"image_{idx}.png")
            image.save(image_path)
            
        print(f"Prepared group {group_idx} (label {label}) with {len(group)} images in {group_dir}")
    
    return group_dirs

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA training")
    parser.add_argument("--token", type=str, required=True, help="Huggingface token")
    parser.add_argument("--num-images-per-group", type=int, default=5, help="Number of images per group")
    parser.add_argument("--base-dir", type=str, default="training_data", help="Base directory for training data")
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)

    dataset_name = "jacekduszenko/lora-adapters-are-good-feature-extractors"
    cache_dir = args.base_dir + '/nsfw-dataset'
    os.makedirs(cache_dir, exist_ok=True)
    dataset, groups = download_and_prepare_dataset(
        dataset_name, 
        args.token, 
        cache_dir, 
        args.num_images_per_group
    )

    group_dirs = prepare_training_directories(dataset, groups, args.base_dir)
    
    with open('group_dirs.json', 'w') as f:
        json.dump(group_dirs, f)
    print(f"\nSaved {len(group_dirs)} group directories to group_dirs.json")

if __name__ == "__main__":
    main() 