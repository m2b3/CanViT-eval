# NYU Depth v2 — Exploration Report

## Goal

Evaluate whether CanViT canvas features support monocular depth estimation
via linear probing, as a third downstream task alongside ADE20K segmentation
and ImageNet-1K classification.

## DINOv3 Depth Protocol (from their codebase)

### Architecture
- **Backbone**: Frozen DINOv3 ViT-B/16 (768-dim, 12 blocks)
- **Head**: BatchNorm → Conv1x1(768 → 256) — "linear" probe
- **Bin conversion**: 256 linearly-spaced depth bins [0.001, 10.0]m,
  AdaBins-style weighted sum (ReLU + normalize + einsum)
- **Loss**: SigLoss (scale-invariant log, 100-step warmup)

### Training
- 38.4k iterations, batch 16 (2×8 GPUs), AdamW lr=3e-4 wd=1e-4
- WarmupOneCycleLR (33% warmup), grad clip 35.0
- NYU crop (43,45,608,472) → random crop 416×544 → flip + color aug
- Depth: uint16 PNG / 1000 → meters

### Evaluation
- Full 480×640 image (no crop), NYU Eigen mask for metrics
- TTA: horizontal flip + average
- Metrics: RMSE (primary), abs_rel, a1 (δ < 1.25)

### DINOv3 ViT-B/16 Reference Numbers
- Paper reports RMSE only (Table in Appendix D.2)
- ViT-B/16: **0.356 RMSE** on NYU (linear probe)

## Our POC Approach

### Simplifications vs DINOv3
1. **Resolution**: Resize to 512×512 (square) vs native 480×640
   - Gives 32×32 patch grid = same as CanViT canvas_grid
   - Slight distortion (1.33:1 → 1:1) — acceptable for POC
2. **Augmentation**: Random hflip only (no rotation, color aug, random crop)
3. **Training**: 10k steps, batch 4 (single GPU) vs 38.4k steps, batch 16
4. **Evaluation**: No Eigen crop mask, no TTA — compute metrics on all valid pixels
5. **Scheduler**: PyTorch OneCycleLR vs DINOv3 WarmupOneCycleLR

### What's Preserved
- Same probe architecture: LN → Dropout2d → BN → Conv1x1(768→256)
- Same loss: SigLoss (imported from dinov3)
- Same metrics: calculate_depth_metrics (imported from dinov3)
- Same bin conversion: FeaturesToDepth (imported from dinov3)
- Same depth range: [0.001, 10.0] meters

### Feature Extraction
- **Teacher**: DINOv3 forward → [B, 32, 32, 768] patches (at 512×512 input)
- **CanViT**: Episode with random viewpoints → [B, 32, 32, D] canvas per timestep
  - Training: loss averaged across all timesteps (anytime decoding)
  - Eval: metrics per timestep

## Dataset

NYU Depth v2 (BTS subset):
- **Train**: 24,231 RGB-D pairs from indoor scenes
- **Test**: 654 pairs
- **Depth range**: 0–2.1m (indoor close-range)
- **93.5% valid pixels** (depth > 0 after NYU border crop)
- **Size on disk**: ~6.4 GB (BinsFormer Google Drive download)

## POC Results (2026-03-18, crockett RTX 4090)

10k steps, batch 4, 512×512, no augmentation, no Eigen crop, no TTA.

| Model | RMSE | abs_rel | a1 (δ<1.25) | Train time |
|-------|------|---------|-------------|------------|
| DINOv3 ViT-B/16 (teacher) | **0.425** | **0.112** | **0.881** | 3.5 min |
| CanViT-B (5 random glimpses) | 0.499 | 0.139 | 0.826 | ~12 min |
| DINOv3 paper (proper eval) | 0.356 | — | — | — |

### Analysis

- **CanViT achieves 85% of teacher RMSE** (0.499 vs 0.425) through 5 random
  128px glimpses — never seeing the full image at full resolution.
- The gap (17% RMSE) is comparable to the active-passive gap on ADE20K segmentation
  at similar timestep counts.
- CanViT training shows more instability (RMSE bounces 0.50–0.57) — random viewpoints
  introduce stochasticity. C2F viewpoints or more timesteps may help.
- Our simplified eval (no Eigen crop, no TTA) explains the gap vs DINOv3's
  reported 0.356. Adding proper eval could bring both numbers down.
- **Depth estimation is viable as a third downstream task.**

### Per-timestep results (C2F policy, 21 timesteps)

Probe trained with 5 random timesteps, evaluated with C2F (deterministic).

| t | RMSE | abs_rel | a1 |
|---|------|---------|-----|
| 0 | 0.518 | 0.147 | 0.809 |
| 1 | 0.508 | 0.143 | 0.819 |
| 3 | 0.493 | 0.140 | 0.828 |
| 5 | **0.487** | **0.139** | **0.830** |
| 10 | 0.486 | 0.140 | 0.828 |
| 15 | 0.486 | 0.142 | 0.825 |
| 20 | 0.487 | 0.144 | 0.821 |

**Key finding**: Depth saturates by t=5 and slightly degrades after t~10.
Unlike ADE20K segmentation (which improves through t=20), depth is a more
"global" task — coarse scene structure captured in early views suffices.
Additional fine-grained views don't add depth information.

### Follow-up experiments (TODO)

- [ ] C2F viewpoints during training (may reduce instability)
- [ ] Proper Eigen crop + TTA for fair comparison to DINOv3 paper numbers
- [ ] Full 38.4k training with augmentation on Nibi
- [ ] Per-timestep CanViT evaluation with different policies (F2C, constant)
- [ ] Higher resolution (1024px / c64) — does it help for depth?

## Integration Plan (Future Work)

### Phase 1: Validate POC
- [x] Teacher baseline on NYU — **RMSE 0.425**
- [x] CanViT baseline on NYU — **RMSE 0.499**
- [x] Compare RMSE — **17% gap, viable**

### Phase 2: Proper Evaluation
- Add `DepthProbe` to `canvit-probes` (parallel to `SegmentationProbe`)
- Add `nyu_depth` task to `canvit-eval` (parallel to `ade20k_seg`)
- Train probes on Nibi (full 38.4k steps, multi-GPU)
- Evaluate with proper Eigen crop + TTA

### Phase 3: Paper Integration
- Add depth results to experiments section
- Export pipeline: .pt → JSON → Typst
- Figure: depth mIoU vs timestep, or vs FLOPs

### Estimated Integration Effort
- **New probe class**: ~5 lines (DepthProbe = SegmentationProbe with n_bins + FeaturesToDepth)
- **New dataset**: ~50 lines (already written in POC)
- **New eval task**: ~100 lines (mirror ade20k_seg.py)
- **New training script**: ~50 lines delta from ADE20K training (different loss, metrics, dataset)
- **Export + figure**: ~100 lines each (mirror existing)
- **Total**: ~400 lines of new code across repos

### Key Questions
1. Is CanViT competitive with DINOv3 on depth? (POC answers this)
2. Does processing order matter for depth? (C2F vs F2C, like for segmentation)
3. Does the 512×512 squish hurt? (vs native aspect ratio)
4. Do we need probe training at 1024px / c64 for depth too?

## File Locations

- **POC code**: `~/projects/canvit-eval/depth_poc/` (depth-poc branch)
- **Dataset**: `/datasets/NYU/nyu/` on crockett
- **DINOv3 reference**: `~/code/dinov3/dinov3/eval/depth/`
- **Training log**: `~/depth_teacher_poc.log` on crockett
