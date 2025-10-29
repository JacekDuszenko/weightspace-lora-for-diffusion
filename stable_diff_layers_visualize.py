import torch
from diffusers import StableDiffusionPipeline
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_stable_diffusion_model():
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return pipe

def analyze_unet_attention_layers(unet):
    attention_info = {
        'cross_attention': [],
        'self_attention': [],
        'layer_structure': []
    }
    
    def get_attention_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            class_name = child.__class__.__name__
            
            if 'CrossAttention' in class_name or 'Attention' in class_name:
                if hasattr(child, 'to_k') and hasattr(child, 'to_q') and hasattr(child, 'to_v'):
                    is_cross = hasattr(child, 'added_kv_proj_dim') or \
                               (hasattr(child, 'cross_attention_dim') and child.cross_attention_dim is not None)
                    
                    attention_info['layer_structure'].append({
                        'name': full_name,
                        'type': 'CrossAttention' if is_cross else 'SelfAttention',
                        'class': class_name
                    })
                    
                    if is_cross:
                        attention_info['cross_attention'].append(full_name)
                    else:
                        attention_info['self_attention'].append(full_name)
            
            get_attention_layers(child, full_name)
    
    get_attention_layers(unet)
    return attention_info

def analyze_unet_blocks(unet):
    block_info = defaultdict(lambda: {'self_attn': 0, 'cross_attn': 0})
    
    for name, module in unet.named_modules():
        if 'attn1' in name:
            parts = name.split('.')
            if len(parts) >= 2:
                block_type = parts[0]
                block_info[block_type]['self_attn'] += 1
        
        elif 'attn2' in name:
            parts = name.split('.')
            if len(parts) >= 2:
                block_type = parts[0]
                block_info[block_type]['cross_attn'] += 1
    
    return block_info

def visualize_attention_architecture(attention_info, block_info):
    print("=" * 80)
    print("STABLE DIFFUSION v1.5 U-NET ATTENTION LAYER ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal Cross-Attention Layers (Text Conditioning): {len(attention_info['cross_attention'])}")
    print(f"Total Self-Attention Layers (Image): {len(attention_info['self_attention'])}")
    
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER BREAKDOWN")
    print("=" * 80)
    
    current_block = None
    for layer in attention_info['layer_structure']:
        block_name = layer['name'].split('.')[0]
        
        if block_name != current_block:
            current_block = block_name
            print(f"\n{'─' * 80}")
            print(f"📦 {block_name.upper()}")
            print(f"{'─' * 80}")
        
        icon = "🔤" if layer['type'] == 'CrossAttention' else "🖼️ "
        print(f"  {icon} {layer['name']}")
        print(f"     └─ Type: {layer['type']}")
    
    print("\n" + "=" * 80)
    print("BLOCK SUMMARY")
    print("=" * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    blocks = list(block_info.keys())
    self_attn_counts = [block_info[b]['self_attn'] for b in blocks]
    cross_attn_counts = [block_info[b]['cross_attn'] for b in blocks]
    
    x = np.arange(len(blocks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, self_attn_counts, width, label='Self-Attention (Image)', color='skyblue')
    bars2 = ax1.bar(x + width/2, cross_attn_counts, width, label='Cross-Attention (Text)', color='lightcoral')
    
    ax1.set_xlabel('U-Net Block', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Attention Layers', fontsize=12, fontweight='bold')
    ax1.set_title('Attention Layer Distribution in SD v1.5 U-Net', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(blocks, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
    total_self = sum(self_attn_counts)
    total_cross = sum(cross_attn_counts)
    
    ax2.pie([total_self, total_cross], 
            labels=[f'Self-Attention\n(Image)\n{total_self} layers', 
                   f'Cross-Attention\n(Text)\n{total_cross} layers'],
            autopct='%1.1f%%',
            colors=['skyblue', 'lightcoral'],
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Overall Attention Type Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('stable_diffusion_v1_5_attention_layers.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved as 'stable_diffusion_v1_5_attention_layers.png'")
    
    print("\n" + "=" * 80)
    for block in blocks:
        print(f"\n{block.upper()}:")
        print(f"  🖼️  Self-Attention Layers: {block_info[block]['self_attn']}")
        print(f"  🔤 Cross-Attention Layers: {block_info[block]['cross_attn']}")
    
    return fig

def print_detailed_architecture_summary():
    print("\n" + "=" * 80)
    print("ARCHITECTURE EXPLANATION")
    print("=" * 80)
    print("""
The Stable Diffusion v1.5 U-Net uses a hierarchical architecture:

1. DOWN BLOCKS (Encoder):
   - Process the noisy latent image at multiple resolutions
   - Self-Attention (attn1): Image attends to itself
   - Cross-Attention (attn2): Image attends to text embeddings from CLIP

2. MID BLOCK (Bottleneck):
   - Lowest resolution processing
   - Self-Attention: Image self-attention
   - Cross-Attention: Text conditioning

3. UP BLOCKS (Decoder):
   - Reconstruct the denoised image with skip connections
   - Self-Attention (attn1): Image attends to itself
   - Cross-Attention (attn2): Image attends to text embeddings

Key Insights:
• Cross-Attention (attn2) = Text conditioning - this is where the text prompt 
  guides the image generation
• Self-Attention (attn1) = Image self-attention - this helps the model understand
  spatial relationships within the image
• Text embeddings come from CLIP ViT-L/14 encoder
• Typically: attn1 (self) comes before attn2 (cross) in each transformer block
""")

def main():
    print("Loading Stable Diffusion v1.5 model...")
    print("(This may take a few minutes on first run as it downloads ~4GB)")
    
    pipe = load_stable_diffusion_model()
    unet = pipe.unet
    
    print("\n✅ Model loaded successfully!")
    print(f"U-Net Architecture: {unet.__class__.__name__}")
    
    print("\nAnalyzing attention layers...")
    attention_info = analyze_unet_attention_layers(unet)
    block_info = analyze_unet_blocks(unet)
    
    fig = visualize_attention_architecture(attention_info, block_info)
    
    print_detailed_architecture_summary()
    
    print("\n" + "=" * 80)
    print("CROSS-ATTENTION LAYERS (Text Conditioning):")
    print("=" * 80)
    for i, layer in enumerate(attention_info['cross_attention'], 1):
        print(f"{i}. {layer}")
    
    print("\n" + "=" * 80)
    print("SELF-ATTENTION LAYERS (Image):")
    print("=" * 80)
    for i, layer in enumerate(attention_info['self_attention'], 1):
        print(f"{i}. {layer}")
    
    plt.show()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

