#!/usr/bin/env python3
"""
Merge supplemental experiment results with the main results.

This script combines the original 7 completed representations with the
2 newly completed ones (info_theoretic and ensemble).
"""

import json
import sys
from pathlib import Path

def merge_results(original_path, supplemental_path, output_path):
    """Merge two results.json files"""

    # Load original results (7 representations)
    with open(original_path, 'r') as f:
        original = json.load(f)

    # Load supplemental results (2 representations)
    with open(supplemental_path, 'r') as f:
        supplemental = json.load(f)

    # Merge the results
    merged = original.copy()

    # Append the new representations to all_layers_results
    merged['all_layers_results'].extend(supplemental['all_layers_results'])

    # Append the new representations to visual_only_results
    merged['visual_only_results'].extend(supplemental['visual_only_results'])

    # Update metadata
    merged['num_all_representations'] = len(merged['all_layers_results'])
    merged['num_visual_representations'] = len(merged['visual_only_results'])

    # Save merged results
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged results saved to: {output_path}")
    print(f"   Original had {len(original['all_layers_results'])} representations")
    print(f"   Supplemental had {len(supplemental['all_layers_results'])} representations")
    print(f"   Merged has {len(merged['all_layers_results'])} representations")

    # List all representations
    print(f"\n📊 Complete representation list:")
    for i, res in enumerate(merged['all_layers_results'], 1):
        rep_name = res['representation'].replace('_all_layers', '')
        print(f"   {i}. {rep_name}")

if __name__ == '__main__':
    original_path = 'visual-ablation-results-optimized/visual_ablation_10k/results.json'
    supplemental_path = 'visual-ablation-results-optimized-supplemental/visual_ablation_10k/results.json'
    output_path = 'visual-ablation-results-optimized/visual_ablation_10k/results_complete.json'

    merge_results(original_path, supplemental_path, output_path)
