# Layer Filtering for Visual-Only Ablation Experiment

## Summary

The experiment has been updated to correctly filter LoRA layers based on Stable Diffusion v1.5 U-Net architecture:

- **attn1** = Self-Attention layers (visual/image processing)
- **attn2** = Cross-Attention layers (text conditioning from CLIP)

## Changes Made

### 1. Updated Layer Filtering (`get_lora_layers` function)

**Old behavior:**
- Simple string matching with 'attn1' or 'attn2'
- No verification or logging

**New behavior:**
- More precise matching with '.attn1.' and '.attn2.' to avoid false positives
- Added logging to show how many layers of each type are found
- Added documentation explaining the layer types

### 2. Enhanced Main Experiment Logging

Added detailed output showing:
- Total number of layers
- Breakdown of visual vs text layers
- Sample layer names for verification
- Validation that layer counts add up correctly

### 3. Updated Documentation

- Clarified in docstrings what attn1 and attn2 represent
- Updated experiment header with architecture background
- Added date of verification with SD v1.5 architecture

## Layer Architecture (from SD v1.5)

Based on the visualization of Stable Diffusion v1.5 model:

```
DOWN_BLOCKS (Encoder):
  Self-Attention Layers: 42
  Cross-Attention Layers: 42

UP_BLOCKS (Decoder):
  Self-Attention Layers: 63
  Cross-Attention Layers: 63

MID_BLOCK (Bottleneck):
  Self-Attention Layers: 7
  Cross-Attention Layers: 7
```

Total per attention type:
- **Self-Attention (attn1)**: 112 layers
- **Cross-Attention (attn2)**: 112 layers
- **Total**: 224 attention layers

Note: Each attention layer has multiple LoRA adapters (to_k, to_q, to_v, to_out), 
and each adapter has both up and down weights, so the actual number of LoRA layers 
in your dataset will be much higher.

## Experiment Modes

### Mode 1: ALL LAYERS (Baseline)
Uses all LoRA layers from both attn1 and attn2
- Includes visual self-attention features
- Includes text cross-attention features
- Tests if concepts are encoded in the full model

### Mode 2: VISUAL-ONLY (Ablation)
Uses ONLY attn1 layers, discards all attn2 layers
- Includes ONLY visual self-attention features
- Removes ALL text conditioning information
- Tests if concepts can be predicted without text features

## Expected Outcomes

**If visual-only performs similarly to all-layers:**
→ Concepts are primarily encoded in visual self-attention features
→ Text conditioning is less important for concept identification

**If visual-only performs much worse:**
→ Concepts require text conditioning information
→ Cross-attention with text embeddings is crucial

## Running the Experiment

```bash
cd /home/jacekduszenko/Workspace/weightspace-lora-for-diffusion

python3 visual_ablation_experiment.py --dataset 1k --num-runs 10

python3 visual_ablation_experiment.py --dataset 1k --quick
```

The experiment will automatically:
1. Load the dataset
2. Identify and separate visual (attn1) and text (attn2) layers
3. Run experiments with all layers
4. Run experiments with visual-only layers
5. Compare the results
6. Save results to `visual-ablation-results/`

## Verification

A test was run to verify the layer filtering works correctly:
- ✅ attn1 layers correctly identified as visual/self-attention
- ✅ attn2 layers correctly identified as text/cross-attention
- ✅ All layers properly categorized (no overlap or missing layers)

## References

- Stable Diffusion v1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- Architecture visualization: `stable_diff_layers_visualize.py`
- Paper: "High-Resolution Image Synthesis With Latent Diffusion Models" (Rombach et al., 2022)

