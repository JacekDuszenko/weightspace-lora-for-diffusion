import os
from datasets import load_from_disk

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, required=True,
                   help='Path to the dataset to check/upload')
args = parser.parse_args()
dataset_path = args.dataset_path
print(f"Loading dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)

print(f"\nDataset size: {len(dataset)}")

print("\nPushing dataset to Hugging Face Hub...")

dataset.push_to_hub(
    f"jacekduszenko/lora-ws-{args.dataset_size}",
    private=False
)
print("\nDone!")
