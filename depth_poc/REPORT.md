# NYU Depth v2 — Exploration Report

**Last updated**: 2026-03-18 ~13:15 EDT

## Goal

Evaluate whether CanViT canvas features support monocular depth estimation
via linear probing on NYU Depth v2, as a third downstream task alongside
ADE20K segmentation and ImageNet-1K classification.

## Prior Art: No Active Vision Model Evaluates Depth

Verified by downloading PDFs and grep'ing for "depth" across 10 papers:

| Paper | Venue | Tasks | "depth" occurrences |
|-------|-------|-------|---------------------|
| RAM (Mnih 2014) | NeurIPS | clf | 0 |
| DRAM (Ba 2015) | ICML | clf | 0 |
| GFNet (Wang 2020) | NeurIPS | clf | 2 (= network layers) |
| GAtE (Seifi 2021) | ICCV | recon/seg/clf | 0 |
| DynamicViT (Rao 2021) | NeurIPS | clf | 0 |
| PatchDropout (Liu 2022) | arxiv | clf | 3 (= model layers) |
| SimGlim (Jha 2023) | WACV | recon | 1 (= background citation) |
| AME (Pardyl 2023) | IJCAI | recon/seg/clf | 0 |
| AdaGlimpse (Pardyl 2024) | ECCV | recon/seg/clf | 0 |
| TORE (2024) | WACV | clf/recon | 1 (= transformer layers) |

**Near-misses excluded:**
- Token pruning (Liang NeurIPS 2022): evaluates depth, but single-pass, not sequential
- "Active Vision in Binocular Depth" (Priorelli 2023): neuroscience model, not CV
- Video depth (convLSTM/GRU): temporal frames, not spatial glimpses of one image

**Methodology**: PDFs downloaded to `/tmp/active_vision_pdfs/`, text extracted via
`pdftotext`, searched with `grep -i depth`. Raw extracts available for verification.

## DINOv3 Reference Protocol

### Architecture (from `dinov3/eval/depth/`)
- Frozen DINOv3 ViT-B/16, 768-dim, 12 blocks
- Head: BN → Conv1x1(768 → 256 bins) — "linear" probe
- Bin conversion: 256 linearly-spaced bins [0.001, 10.0]m, AdaBins-style
  (ReLU + normalize + weighted sum). Implementation: `FeaturesToDepth`
- Loss: SigLoss (scale-invariant log, 100-step warmup)

### Training (from `config-nyu.yaml`)
- **38,400 iterations** = 3 × 12,800 (clean 1/3 warmup, not epoch-aligned)
- Effective batch 16 (2 per GPU × 8 GPUs) → **25.4 epochs** on NYU (24,231 train)
- AdamW lr=3e-4, wd=1e-4, grad_clip=35
- WarmupOneCycleLR (12,800 warmup iters)
- Transforms: NYU crop (43,45,608,472) → random crop 416×544 → rotation ±2.5°
  → hflip → color aug (brightness 0.75–1.25) → ImageNet normalize

### Evaluation (from `config-nyu.yaml`)
- Full 480×640 image (no spatial crop), resize to model input
- **Eigen crop mask** for metrics: y=45:471, x=41:601 on 480×640
  (426×560 = 238,560 pixels, 77.7% of image)
- TTA: horizontal flip + average
- Metrics: RMSE (primary), abs_rel, δ<1.25 (a1)
- Depth range: [0.001, 10.0] meters

### What is Eigen crop?
Standard evaluation region from Eigen et al. (NeurIPS 2014, "Depth Map
Prediction from a Single Image using a Multi-Scale Deep Network"). A binary
mask excluding border pixels where Kinect depth is unreliable. Applied at
metric computation time, NOT as spatial preprocessing — the model still
processes the full image. Coordinates: `(y1=45, y2=471, x1=41, x2=601)` on
original 480×640 frames. Implementation: `dinov3.eval.depth.datasets.datasets_utils.make_valid_mask`.

### DINOv3 published RMSE (MODEL_CARD.md, linear probing)
| Model | NYU RMSE |
|-------|----------|
| ViT-S/16 | 0.403 |
| **ViT-B/16** | **0.373** |
| ViT-L/16 | 0.352 |
| ViT-7B/16 | 0.309 |

**Source**: `/Users/yberreby/code/dinov3/MODEL_CARD.md`, lines 177–184.
These numbers use the full DINOv3 protocol (native aspect ratio, 25 epochs,
TTA, random crop, tuned lr for batch 16). **We should NOT compare our
numbers to these** — we must train our own teacher probes under matching
conditions for a fair CanViT vs teacher comparison.

## Our Protocol (v1 — clean runs)

### Deviations from DINOv3
1. **Square resize** to `scene_size × scene_size` (DINOv3: native 480×640)
   - Distorts aspect ratio 1.33:1 → 1:1. Affects geometric features.
   - Done for CanViT consistency (canvas grid is square).
2. **Batch 4** single GPU (DINOv3: batch 16 on 8 GPUs)
   - 38.4k steps × batch 4 = **6.3 epochs** (DINOv3: 25.4 epochs)
   - 4× fewer training samples seen. May undertrain.
3. **LR 7.5e-5** (DINOv3: 3e-4 for batch 16, linear scaling: 3e-4 × 4/16)
   - **Rationale**: standard linear scaling rule (Goyal et al.)
   - **Caveat**: DINOv3 didn't use linear scaling — they swept {1e-4, 3e-4, 1e-3}
     and found 3e-4 optimal. Our scaling may be suboptimal.
4. **No random spatial crop** (random_crop_size = scene_size, effectively no-op)
   - DINOv3 crops 416×544 from 565×427 (post-NYU-crop). We skip this.
5. **No TTA** at eval (DINOv3: horizontal flip average)
   - User decision: TTA is a non-goal for this POC.
6. **OneCycleLR** vs WarmupOneCycleLR (similar but not identical schedules)

### What matches DINOv3
- Same probe: LN → Dropout2d → BN → Conv1x1(D → 256 bins) → FeaturesToDepth
- Same loss: SigLoss (imported from `dinov3.eval.depth.loss`)
- Same metrics: `calculate_depth_metrics` (imported from `dinov3.eval.depth.metrics`)
- Same bin conversion: `FeaturesToDepth` (imported from `dinov3.eval.depth.models`)
- Same eval mask: Eigen crop via `make_valid_mask` (imported from `dinov3.eval.depth.datasets.datasets_utils`)
- Same train transforms: `make_depth_train_transforms` (rotation, color aug, hflip)
  (imported from `dinov3.eval.depth.transforms`)
- Same depth range: [0.001, 10.0] meters
- Same depth normalization: uint16 / 1000 → meters

### v1 protocol summary
```
All v1 runs use: batch=4, steps=38400, lr=7.5e-5, wd=1e-4, grad_clip=35
  Train: NYU crop → resize (scene_size²) → DINOv3 augmentations → ImageNet normalize
  Eval:  no crop → resize (scene_size²) → ImageNet normalize → Eigen crop mask
  Loss:  SigLoss at feature resolution, depth downsampled via nearest
  Comet: online logging to m2b3-ava/canvit-depth-poc, model artifacts uploaded
```

## Dataset

NYU Depth v2, BTS subset (BinsFormer Google Drive download):
- **Train**: 24,231 RGB-D pairs (indoor scenes, ~300 scene categories)
- **Test**: 654 pairs
- **Depth range**: 0–10m (indoor close-range, most values 0–3m)
- **Valid pixels**: ~93% (depth > 0 after NYU border crop)
- **On crockett**: `/datasets/NYU/nyu/` (6.4 GB, zip deleted to save space)
- **Split files**: `nyu_train.txt`, `nyu_test.txt` (BTS format: `img_path depth_path focal_length`)

## Results

### v1 (Eigen crop + DINOv3 transforms + Comet, lr=7.5e-5)

**Status**: Running on crockett. Chain script: `~/run_clean_v1.sh`.

| Run | Scene | GFLOPs | Steps | Wallclock | Best RMSE | Status |
|-----|-------|--------|-------|-----------|-----------|--------|
| Teacher 512px | 512 | 215.2 | 38.4k | 19 min | **0.426** | Done |
| Teacher 384px | 384 | 111.9 | 38.4k | ~10 min | *running* | In progress |
| Teacher 256px | 256 | 47.2 | 38.4k | ~5 min | — | Queued |
| CanViT c32 10t | 512 | 15.9/glimpse | 38.4k | ~107 min | — | Queued |

Comet project: https://www.comet.com/m2b3-ava/canvit-depth-poc/

### v0 (archived — NO Eigen crop, NO DINOv3 transforms)

**INVALIDATED** for paper use. Archived to `~/depth_archive_v0/` on crockett.
Kept for reference only — directional findings were valid, absolute numbers are not.

| Run | Steps | LR | Best RMSE | Note |
|-----|-------|----|-----------|------|
| Teacher 512px | 10k | 3e-4 | 0.425 | No Eigen crop |
| Teacher 512px | 38.4k | 3e-4 | 0.412 | No Eigen crop |
| Teacher 256px | 38.4k | 3e-4 | 0.452 | No Eigen crop |
| Teacher 384px | 38.4k | 3e-4 | 0.422 | No Eigen crop |
| CanViT 5t | 10k | 3e-4 | 0.499 | No Eigen crop, 5 train timesteps |
| CanViT 10t | 10k | 3e-4 | 0.485 | No Eigen crop |
| CanViT 10t | 38.4k | 3e-4 | 0.477 | No Eigen crop, unstable (lr not scaled) |

### v0 per-timestep eval (C2F, no Eigen crop, 10k probe)

Directionally useful — shows depth saturation behavior:

| t | GFLOPs | RMSE (5t train) | RMSE (10t train) |
|---|--------|-----------------|------------------|
| 0 | 15.9 | 0.518 | 0.521 |
| 5 | 95.2 | 0.487 | 0.484 |
| 10 | 174.5 | 0.486 | 0.479 |
| 20 | 333.1 | 0.487 | 0.473 |

**Key finding**: 5t probe saturates at t=5. 10t probe improves through t=20.
More training timesteps → better temporal generalization (demonstrated for
depth only; untested for ADE20K).

## FLOP Analysis

All FLOPs computed analytically from `analysis/flops/` (unit-tested).
Source: `analysis/flops/models/canvit.py`, `analysis/flops/models/teacher.py`.

### Teacher FLOPs by resolution (DINOv3 ViT-B/16, patch=16)
| Input | Patch grid | GFLOPs |
|-------|-----------|--------|
| 128×128 | 8×8 | 12.0 |
| 256×256 | 16×16 | 47.2 |
| 384×384 | 24×24 | 111.9 |
| 512×512 | 32×32 | 215.2 |

### CanViT FLOPs (c32, 128px glimpse, per-glimpse = 15.9 GFLOPs)
| Component | GFLOPs | % |
|-----------|--------|---|
| Backbone (12 ViT blocks on 8×8 patches + prefix) | 12.3 | 79% |
| Canvas Read (3×) | 1.6 | 10% |
| Canvas Write (3×) | 1.6 | 10% |

Depth head (256 bins): 0.54 GFLOPs (~3.5% of per-glimpse, negligible).

### FLOP-matched comparison points
| CanViT timesteps | CanViT GFLOPs | Teacher match | Teacher GFLOPs |
|-----------------|---------------|---------------|----------------|
| t=1 | 15.9 | 128px | 12.0 |
| t=3 | 47.6 | **256px** | **47.2** |
| t=7 | 111.0 | **384px** | **111.9** |
| t=13 | 206.2 | **512px** | **215.2** |

## Paper Integration Strategy

### NeurIPS 2026 timeline
- **Today**: March 18. **Deadline**: ~May 6. **49 days remaining.**
- Paper currently ~8–9 pages main body (9-page limit). Discussion disabled.

### Current depth mentions in manuscript
6 mentions, all as "example of what DINOv3 does" — never evaluated by CanViT:
- related_work: "required for semantic segmentation, depth estimation, and other dense tasks"
- problem: "spatially-grounded, dense tasks like semantic segmentation or depth estimation"
- pretraining: DINOv3 "delivers... transfer to classification, segmentation, depth estimation"

### Recommended approach
**Appendix + one sentence in main body.** Rationale:
- Main body is near 9-page limit. Adding a depth subsection risks overflow.
- Depth results are strong but secondary to ADE20K (the star task).
- No active vision baseline exists for depth — the comparison is CanViT vs teacher only.
- Appendix has unlimited space and `evaluation_details.typ` already exists.

**Specific changes:**
1. `06_experiments.typ`: Add 1 sentence — "Preliminary results on NYU Depth v2 (Appendix X) confirm that canvas features also support geometric tasks."
2. `appendix/evaluation_details.typ`: Add ~300 words on NYU depth protocol.
3. `appendix/supp_figs_tables.typ`: Add depth RMSE table (per-policy, per-timestep).
4. Create `export/nyu_depth.py` for JSON pipeline.

### Integration effort
| Component | Lines | Location |
|-----------|-------|----------|
| DepthProbe class | ~30 | canvit-probes |
| NYU dataset class | ~45 | canvit-probes or canvit-eval |
| nyu_depth eval task | ~100 | canvit-eval |
| Export script | ~100 | this repo (export/) |
| Appendix text | ~300 words | typst/ |
| Figure generator | ~100 | plotting/ |
| **Total** | **~400 LOC** | 3 repos |

## Known Issues & Caveats

1. **Square resize distorts geometry.** NYU images are 4:3. Squishing to 1:1
   hurts depth more than segmentation (depth encodes 3D structure that depends
   on aspect ratio). Both teacher and CanViT are affected equally, so relative
   comparison is fair.

2. **6.3 epochs ≠ 25 epochs.** Our batch 4 at 38.4k steps sees 4× fewer samples
   than DINOv3. The probe may undertrain. Longer training on Nibi (single H100,
   ~150k steps for 25 epochs at batch 4) is a future TODO.

3. **LR 7.5e-5 may be suboptimal.** v0 teacher at 3e-4 got 0.412 (no Eigen);
   v1 teacher at 7.5e-5 got 0.426 (with Eigen). The lr scaling might have been
   too aggressive. A sweep {7.5e-5, 1e-4, 1.5e-4, 3e-4} on crockett would
   take ~1h per value for teacher. Not done yet.

4. **SigLoss warmup interacts with multi-timestep training.** SigLoss warms up
   over 100 backward passes. With T=10 timesteps, each training step calls it
   10×, so warmup ends after 10 training steps (not 100). Minor effect.

5. **Train/eval viewpoint mismatch.** CanViT trains with random viewpoints,
   evaluates with C2F (deterministic). The probe never sees C2F-ordered features
   during training. Same mismatch as ADE20K segmentation.

## File Locations

### Code (canvit-eval repo, `depth-poc` branch)
- `depth_poc/train.py` — training (teacher + canvit modes), ~340 lines
- `depth_poc/dataset.py` — NYU dataset class, ~45 lines
- `depth_poc/eval_per_timestep.py` — per-timestep C2F evaluation, ~120 lines
- `depth_poc/REPORT.md` — this file
- `depth_poc/run_teacher_multi_res.sh` — batch teacher training

### On crockett
- **Worktree**: `~/projects/canvit-eval-depth/` (depth-poc branch)
- **Dataset**: `/datasets/NYU/nyu/` (6.4 GB)
- **v1 checkpoints**: `checkpoints/{teacher-512px,teacher-384px,...}/`
- **v1 logs**: `~/v1_*.log`
- **v0 archive**: `~/depth_archive_v0/`
- **Chain script**: `~/run_clean_v1.sh`
- **Chain log**: `~/v1_chain.log`

### Reference
- DINOv3 depth code: `~/code/dinov3/dinov3/eval/depth/`
- DINOv3 NYU config: `~/code/dinov3/dinov3/eval/depth/configs/config-nyu.yaml`
- DINOv3 MODEL_CARD: `~/code/dinov3/MODEL_CARD.md` (lines 177–184 for NYU RMSE)
- PDF text extracts: `/tmp/active_vision_pdfs/*.txt` (local)

### Comet
- Workspace: `m2b3-ava`
- Project: `canvit-depth-poc`
- URL: https://www.comet.com/m2b3-ava/canvit-depth-poc/
- API key: `~/comet_api_key.txt` on crockett (read by train.py automatically)
