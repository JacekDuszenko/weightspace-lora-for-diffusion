# Checkpoint & Resume System - Never Lose Progress! 💾

## 🎯 Problem Solved

Long experiments (10+ hours) can fail due to:
- Bugs in specific representations
- Out of memory errors  
- System crashes
- Power outages
- SSH disconnections

**Before:** Start from scratch, lose ALL progress ❌
**After:** Resume from last checkpoint, keep ALL progress ✅

## ✅ How It Works

### Automatic Checkpointing

After **each representation completes**, results are automatically saved:

```
results_50k/
├── visual_ablation_50k/
│   ├── checkpoints/
│   │   ├── all_layers_simple_stats.json    ✅ Completed
│   │   ├── all_layers_rank_based.json      ✅ Completed
│   │   ├── all_layers_stats.json           ✅ Completed
│   │   ├── visual_only_simple_stats.json   ✅ Completed
│   │   └── ...
│   └── results.json (final merged results)
```

### Automatic Resume

When you restart the experiment:
1. ✅ Loads all existing checkpoints
2. ✅ Shows progress summary (X/18 completed)
3. ✅ Skips completed representations  
4. ✅ Only runs remaining work
5. ✅ Merges everything at the end

**Zero configuration needed - it just works!** 🚀

## 📊 Example Scenarios

### Scenario 1: Crash After 3 Representations

**What happens:**
```bash
# Run 1: Start fresh
python3 visual_ablation_experiment.py --dataset 50k ...

# Output:
# simple_stats_all_layers ✅ (checkpoint saved)
# rank_based_all_layers ✅ (checkpoint saved)  
# stats_all_layers ✅ (checkpoint saved)
# spectral_all_layers ... 💥 CRASH!

# Run 2: Just restart with same command!
python3 visual_ablation_experiment.py --dataset 50k ...

# Output:
# 🔄 Found 3 existing checkpoints for all_layers
#   ✅ simple_stats_all_layers
#   ✅ rank_based_all_layers
#   ✅ stats_all_layers
# 
# Progress: 3/18 completed (16.7%)
# 🔄 Resuming from checkpoints. Will run 15 remaining...
#
# ⏭️ Skipping simple_stats_all_layers (already completed)
# ⏭️ Skipping rank_based_all_layers (already completed)
# ⏭️ Skipping stats_all_layers (already completed)
# Evaluating spectral_all_layers... (continues from here!)
```

### Scenario 2: Fix Bug and Rerun

**What happens:**
```bash
# Run 1: Bug in ensemble representation
python3 visual_ablation_experiment.py --dataset 50k ...

# Complete 8/9 all_layers, then ensemble crashes ❌

# Fix the bug in code, then restart:
python3 visual_ablation_experiment.py --dataset 50k ...

# Loads 8 checkpoints, skips them, only runs fixed ensemble!
# Saves 8 hours of recomputation! ✅
```

### Scenario 3: Experiment Interrupted

```bash
# Run overnight, accidentally kill process in morning
# Lost: Nothing! 
# Saved: Everything completed before kill

# Just restart:
python3 visual_ablation_experiment.py --dataset 50k ...

# Continues exactly where it left off! ✅
```

## 🔧 Advanced Usage

### Check Progress Without Running

```bash
# See what's been completed
ls results_50k/visual_ablation_50k/checkpoints/

# Count completed
ls results_50k/visual_ablation_50k/checkpoints/ | wc -l

# Expected: 18 files total (9 all_layers + 9 visual_only)
```

### Manually Delete Specific Checkpoint

If you want to re-run a specific representation:

```bash
# Re-run just spectral_all_layers
rm results_50k/visual_ablation_50k/checkpoints/all_layers_spectral.json

# Restart experiment - will skip others, only run spectral
python3 visual_ablation_experiment.py --dataset 50k ...
```

### Start Fresh

```bash
# Delete all checkpoints to start from scratch
rm -rf results_50k/visual_ablation_50k/checkpoints/

# Or delete entire output directory
rm -rf results_50k/
```

## 📈 Time Savings Examples

### 50k Dataset Crash Scenarios:

#### Crash after 3 representations (3 hours in):
- **Without checkpoints:** Lose 3 hours, start over
- **With checkpoints:** Resume, save 3 hours ✅

#### Crash after 12 representations (12 hours in):
- **Without checkpoints:** Lose 12 hours, start over
- **With checkpoints:** Resume, save 12 hours ✅

#### Bug in ensemble (17/18 done, 15 hours in):
- **Without checkpoints:** Fix bug, restart, lose 15 hours
- **With checkpoints:** Fix bug, restart, rerun 1 rep (1 hour) ✅

## 🔍 What Gets Checkpointed

Each checkpoint saves complete results:
```json
{
  "representation": "spectral_all_layers",
  "num_layers": 128,
  "train_acc_mean": 0.95,
  "train_acc_std": 0.01,
  "val_acc_mean": 0.68,
  "test_acc_mean": 0.65,
  "... all 30+ metrics ...": "...",
  "per_run_results": [...],
  "learning_curves": [...],
  "confusion_matrices": [...]
}
```

**Everything** is saved - no data loss!

## ⚠️ Important Notes

### Checkpoints Are Configuration-Specific

Checkpoints include:
- Dataset name
- Layer configuration (all vs visual-only)
- Representation method

If you change these, new checkpoints are created (old ones ignored).

### Feature Cache vs Checkpoints

Two separate systems:

**Feature Cache** (`.feature_cache/`):
- Saves computed features (before training)
- Speeds up feature computation (10x)
- Can reuse across different runs/configs

**Checkpoints** (`output_dir/checkpoints/`):
- Saves final results (after training)
- Enables resume from crashes
- Specific to this experiment run

**Both are independent and both are awesome!** 🎉

### When Checkpoints Are NOT Used

```bash
# Different output directory = fresh start
python3 visual_ablation_experiment.py --dataset 50k --output-dir NEW_DIR

# Different dataset = fresh start  
python3 visual_ablation_experiment.py --dataset 100k --output-dir results_50k

# But feature cache CAN be reused!
```

## 💡 Best Practices

### 1. Use Consistent Output Directory

```bash
# GOOD: Always use same output dir for a dataset
python3 visual_ablation_experiment.py --dataset 50k --output-dir results_50k
# If it crashes, restart with SAME command

# BAD: Different output dirs
python3 visual_ablation_experiment.py --dataset 50k --output-dir results_50k_v1
# crashes
python3 visual_ablation_experiment.py --dataset 50k --output-dir results_50k_v2
# Lost all checkpoints!
```

### 2. Don't Delete Checkpoints Until Done

Keep checkpoints until experiment fully completes and you've verified results!

### 3. Use With Feature Caching

```bash
# Best combo: Feature cache + Checkpoints
python3 visual_ablation_experiment.py \
    --dataset 50k \
    --output-dir results_50k \
    --cache-features \     # ← Feature cache
    --cache-dir .cache     # ← (Checkpoints automatic)
```

### 4. Monitor Checkpoints

```bash
# Watch checkpoints being created
watch -n 60 'ls -lh results_50k/visual_ablation_50k/checkpoints/'

# Each new file = 1 representation completed!
```

## 🚀 Restart Your 50k Experiment

Your current run is stuck. Kill it and restart with fixed code:

```bash
# 1. Kill the slow run
pkill -f "visual_ablation_experiment.py.*50k"

# 2. Keep existing checkpoints if any completed successfully
# (simple_stats and rank_based should be done)

# 3. Restart with fixed code
cd /home/jacekduszenko/Workspace/weightspace-lora-for-diffusion && \
source ../research/.venv/bin/activate && \
nohup python3 visual_ablation_experiment.py \
    --dataset 50k \
    --num-runs 10 \
    --output-dir results_50k_final \
    --cache-features \
    --dropout 0.5 \
    --weight-decay 1e-4 \
    --batch-size 64 \
    > results_50k_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 4. Monitor
tail -f results_50k_restart_*.log
```

The experiment will:
- Load any completed representations from checkpoints
- Skip them automatically
- Only compute what's missing
- Save checkpoints as it goes
- Can be restarted anytime without losing progress!

## 📊 Expected Behavior

### First Run (No Checkpoints):
```
PROGRESS SUMMARY
All Layers Config:    0/9 completed, 9 remaining
Visual-Only Config:   0/9 completed, 9 remaining  
Total Progress:       0/18 (0.0%)

🔄 Resuming from checkpoints. Will run 18 remaining...

Evaluating simple_stats_all_layers...
[completes] ✅
💾 Checkpoint saved

Evaluating rank_based_all_layers...
[completes] ✅
💾 Checkpoint saved

... continues ...
```

### After Crash & Restart (3 Completed):
```
🔄 Found 3 existing checkpoints for all_layers
  ✅ simple_stats_all_layers
  ✅ rank_based_all_layers
  ✅ stats_all_layers

PROGRESS SUMMARY
All Layers Config:    3/9 completed, 6 remaining
Visual-Only Config:   0/9 completed, 9 remaining
Total Progress:       3/18 (16.7%)

🔄 Resuming from checkpoints. Will run 15 remaining...

⏭️ Skipping simple_stats_all_layers (already completed)
⏭️ Skipping rank_based_all_layers (already completed)  
⏭️ Skipping stats_all_layers (already completed)

Evaluating spectral_all_layers...  ← Starts here!
[completes] ✅
💾 Checkpoint saved

... continues ...
```

## ✨ Benefits

1. **Crash-Resistant** - Never lose hours of work
2. **Bug-Friendly** - Fix and continue
3. **Interruptible** - Can stop/start anytime
4. **Progress Tracking** - Always know what's done
5. **Zero Config** - Works automatically
6. **Fast Recovery** - Resume in seconds

Your long experiments are now **production-grade robust**! 🎓📊

## 🎉 Summary

✅ **Checkpoints after each representation**
✅ **Automatic resume on restart**
✅ **Progress tracking**
✅ **Crash recovery**
✅ **Bug recovery**
✅ **Zero data loss**

**Your 50k experiment is now crash-proof!** 🚀

