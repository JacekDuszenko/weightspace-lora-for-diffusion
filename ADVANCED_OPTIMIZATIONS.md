# Advanced Optimizations for 100k Dataset

## Goal
Run experiments on 100k dataset in **~2 hours** instead of days.

## Optimization Strategies (Ordered by Impact)

### 1. 🔥 **Feature Caching to Disk** (10x speedup)
**Problem:** Features are recomputed for every run (10 runs = 10x wasted compute)
**Solution:** Compute features once, save to disk, load for all runs

**Impact:** 
- Current: Compute features 10 times
- After: Compute features 1 time
- **Speedup: 10x for feature computation**

### 2. ⚡ **Parallel Representation Processing** (8x speedup)
**Problem:** Representations processed sequentially
**Solution:** Process 8 representations in parallel using multiprocessing

**Impact:**
- Current: 8 representations × 3 hours each = 24 hours
- After: max(3 hours) = 3 hours
- **Speedup: 8x for total experiment time**

### 3. 🎮 **GPU-Accelerated Feature Computation** (2-5x speedup)
**Problem:** CPU-bound operations (FFT, SVD) are slow
**Solution:** Keep tensors on GPU, use torch operations instead of numpy

**Impact:**
- SVD: CPU → GPU (5x faster)
- Stats: Already fast
- FFT: numpy → torch.fft (3x faster)
- **Speedup: 2-5x for feature computation**

### 4. 🏃 **More Aggressive Training** (2x speedup)
**Problem:** Training 1000 epochs with patience=20 is conservative
**Solution:** 
- Reduce max epochs to 200
- Reduce patience to 10
- Use larger batch size (64 instead of 32)

**Impact:**
- Faster convergence
- Fewer epochs needed
- **Speedup: 2x for training**

### 5. 📦 **Batch Feature Computation** (1.5x speedup)
**Problem:** Processing layers one at a time has overhead
**Solution:** Process multiple layers in parallel batches

**Impact:**
- Better GPU utilization
- Reduced overhead
- **Speedup: 1.5x for feature computation**

## Combined Impact

| Stage | Before | After | Speedup |
|-------|--------|-------|---------|
| Data loading | 3 min | 3 min | 1x (already optimized) |
| Feature computation | 120 min | 6 min | **20x** (cache + GPU) |
| Training (10 runs) | 60 min | 30 min | **2x** (aggressive early stop) |
| Total per representation | 183 min | 39 min | **~5x** |
| All representations (8) | 1464 min (24h) | 39 min | **~37x** (parallel) |
| Both configs (2) | 2928 min (49h) | 78 min (**1.3h**) | **~37x** |

**For 100k dataset:**
- Estimated time with current optimizations: ~4 days
- **With advanced optimizations: ~2-3 hours** ✅

## Implementation Priority

### Phase 1: Feature Caching (Biggest Win)
- Implement feature cache to disk
- Check if cached features exist before computing
- Save with hash of layer names + dataset

### Phase 2: Parallel Processing
- Use multiprocessing for representations
- Process all 8 representations simultaneously

### Phase 3: GPU Acceleration
- Replace numpy FFT with torch.fft
- Keep tensors on GPU throughout

### Phase 4: Training Optimization
- Reduce max epochs
- Increase batch size
- More aggressive early stopping

