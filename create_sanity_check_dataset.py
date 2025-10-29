import numpy as np
from datasets import Dataset, Features, Array2D, Value
import argparse

def create_sanity_check_dataset(num_samples=5, num_classes=3):
    """
    Create a minimal sanity-check dataset for testing the training pipeline.
    
    Args:
        num_samples: Number of samples to generate (default: 5)
        num_classes: Number of unique classes (default: 3)
    """
    print(f"Creating sanity-check dataset with {num_samples} samples and {num_classes} classes...")
    
    common_lora_layers = [
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora.up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight",
    ]
    
    layer_shapes = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight": (4, 320),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight": (320, 4),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": (4, 320),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight": (320, 4),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora.down.weight": (4, 320),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora.up.weight": (320, 4),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight": (4, 768),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight": (320, 4),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora.down.weight": (4, 320),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora.up.weight": (320, 4),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora.down.weight": (4, 768),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora.up.weight": (320, 4),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight": (4, 320),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight": (320, 4),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight": (4, 768),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight": (320, 4),
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight": (4, 1280),
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight": (1280, 4),
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight": (4, 768),
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight": (1280, 4),
    }
    
    data_dict = {}
    
    for layer_name in common_lora_layers:
        shape = layer_shapes[layer_name]
        layer_data = []
        
        for sample_idx in range(num_samples):
            class_label = sample_idx % num_classes
            
            mean = class_label * 0.1
            std = 0.01
            weights = np.random.normal(mean, std, shape).astype(np.float32)
            
            layer_data.append(weights)
        
        data_dict[layer_name] = layer_data
    
    labels = [i % num_classes for i in range(num_samples)]
    data_dict['category_label'] = labels
    
    concepts = [f"concept_{i % num_classes}" for i in range(num_samples)]
    data_dict['concept'] = concepts
    
    print(f"\nDataset structure:")
    print(f"  - Samples: {num_samples}")
    print(f"  - Classes: {num_classes}")
    print(f"  - LoRA layers: {len(common_lora_layers)}")
    print(f"  - Visual layers (attn1): {len([l for l in common_lora_layers if '.attn1.' in l])}")
    print(f"  - Text layers (attn2): {len([l for l in common_lora_layers if '.attn2.' in l])}")
    
    features_dict = {}
    for layer_name in common_lora_layers:
        shape = layer_shapes[layer_name]
        features_dict[layer_name] = Array2D(shape=shape, dtype='float32')
    features_dict['category_label'] = Value('int64')
    features_dict['concept'] = Value('string')
    
    features = Features(features_dict)
    
    dataset = Dataset.from_dict(data_dict, features=features)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"\nDataset info:")
    print(dataset)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Create sanity-check dataset for testing')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples (default: 5)')
    parser.add_argument('--classes', type=int, default=3,
                       help='Number of classes (default: 3)')
    parser.add_argument('--output', type=str, default='sanity_check_dataset',
                       help='Output directory name (default: sanity_check_dataset)')
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Push to HuggingFace Hub (requires login)')
    parser.add_argument('--repo-name', type=str, default='jacekduszenko/lora-ws-sanity-check',
                       help='HuggingFace repo name (default: jacekduszenko/lora-ws-sanity-check)')
    
    args = parser.parse_args()
    
    dataset = create_sanity_check_dataset(
        num_samples=args.samples,
        num_classes=args.classes
    )
    
    print(f"\n{'='*80}")
    print("SAVING DATASET")
    print(f"{'='*80}")
    
    dataset.save_to_disk(args.output)
    print(f"✅ Dataset saved locally to: {args.output}/")
    
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.repo_name}")
        try:
            dataset.push_to_hub(args.repo_name, private=False)
            print(f"✅ Dataset pushed to: https://huggingface.co/datasets/{args.repo_name}")
        except Exception as e:
            print(f"❌ Failed to push to hub: {e}")
            print("Make sure you're logged in: huggingface-cli login")
    
    print(f"\n{'='*80}")
    print("USAGE")
    print(f"{'='*80}")
    print("\nTo use this dataset in your experiment, load it with:")
    print(f"\n  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{args.output}')")
    
    if args.push_to_hub:
        print(f"\nOr from HuggingFace Hub:")
        print(f"\n  from datasets import load_dataset")
        print(f"  dataset = load_dataset('{args.repo_name}')['train']")
    
    print(f"\n{'='*80}")
    print("SAMPLE DATA")
    print(f"{'='*80}")
    
    print(f"\nFirst sample:")
    sample = dataset[0]
    print(f"  category_label: {sample['category_label']}")
    print(f"  concept: {sample['concept']}")
    
    sample_layer = common_lora_layers[0]
    sample_array = np.array(sample[sample_layer])
    print(f"  {sample_layer}.shape: {sample_array.shape}")
    print(f"  {sample_layer}[0, :5]: {sample_array[0, :5]}")
    
    print(f"\n✅ Sanity-check dataset ready!")


if __name__ == '__main__':
    common_lora_layers = [
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora.up.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora.down.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora.up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora.down.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora.up.weight",
    ]
    main()

