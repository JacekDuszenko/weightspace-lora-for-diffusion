"""
Quick script to inspect LoRA layer names in the dataset
"""
from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset('jacekduszenko/lora-ws-1k')['train']

# Get layer names
lora_layers = [key for key in dataset.features.keys()
              if 'lora.down.weight' in key or 'lora.up.weight' in key]

print(f"\nTotal LoRA layers: {len(lora_layers)}\n")

# Group by type
cross_attn_layers = [l for l in lora_layers if 'attn2' in l or 'cross' in l.lower()]
self_attn_layers = [l for l in lora_layers if 'attn1' in l or ('attn' in l and 'attn2' not in l and 'cross' not in l.lower())]
ff_layers = [l for l in lora_layers if 'ff' in l.lower() or 'mlp' in l.lower()]
other_layers = [l for l in lora_layers if l not in cross_attn_layers + self_attn_layers + ff_layers]

print(f"Cross-attention layers (text conditioning): {len(cross_attn_layers)}")
print(f"Self-attention layers (visual): {len(self_attn_layers)}")
print(f"Feed-forward layers: {len(ff_layers)}")
print(f"Other layers: {len(other_layers)}")

print("\n=== Sample Cross-Attention Layers ===")
for layer in cross_attn_layers[:10]:
    print(f"  {layer}")

print("\n=== Sample Self-Attention Layers ===")
for layer in self_attn_layers[:10]:
    print(f"  {layer}")

print("\n=== ALL Layer Names (sorted) ===")
for layer in sorted(lora_layers):
    layer_type = ""
    if 'attn2' in layer or 'cross' in layer.lower():
        layer_type = "[CROSS-ATTN - TEXT]"
    elif 'attn1' in layer:
        layer_type = "[SELF-ATTN - VISUAL]"
    elif 'ff' in layer.lower() or 'mlp' in layer.lower():
        layer_type = "[FF - VISUAL]"
    print(f"{layer_type:<25} {layer}")

print(f"\n\n=== Summary ===")
print(f"Text conditioning layers to REMOVE: {len(cross_attn_layers)}")
print(f"Visual layers to KEEP: {len(self_attn_layers) + len(ff_layers) + len(other_layers)}")
