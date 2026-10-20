# Real-Time RGBT Semantic Segmentation: CSFNet-1 & CSFNet-2
Live Demo - https://rgbt-segmentation.streamlit.app/

PyTorch implementation and benchmark evaluation of **CSFNet** (Cosine Similarity Fusion Network) for real-time multispectral RGB-Thermal (RGBT) semantic segmentation on the MFNet dataset (`ir_seg_dataset/`).

This repository provides modular implementations of both **CSFNet-1** (STDC1 backbone, 11.30M parameters) and **CSFNet-2** (STDC2 backbone, 19.36M parameters), dual-stream ImageNet weight transfer, training routines with Automatic Mixed Precision (AMP), and evaluation protocols across daytime and nighttime test splits.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
  - [CSFNet-1 vs. CSFNet-2 Structural Differences](#csfnet-1-vs-csfnet-2-structural-differences)
  - [Dual-to-Single Branch Encoder](#dual-to-single-branch-encoder)
  - [Cosine Similarity Attentional Feature Fusion (CS-AFM)](#cosine-similarity-attentional-feature-fusion-cs-afm)
  - [Context Module (CM)](#context-module-cm)
  - [Lightweight Progressive Bilateral Decoder](#lightweight-progressive-bilateral-decoder)
- [Dataset &amp; Preprocessing](#dataset--preprocessing)
  - [Class Definitions &amp; Imbalance](#class-definitions--imbalance)
  - [Normalization Statistics](#normalization-statistics)
- [Benchmark Results](#benchmark-results)
  - [Head-to-Head Model Comparison](#head-to-head-model-comparison)
  - [Per-Class IoU Breakdown](#per-class-iou-breakdown)
  - [Daytime vs. Nighttime Robustness](#daytime-vs-nighttime-robustness)
- [Environment &amp; Installation](#environment--installation)
- [Pretrained Weights](#pretrained-weights)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference &amp; Visualization](#inference--visualization)
- [Repository Structure](#repository-structure)

---

## Architecture Overview

CSFNet addresses thermal-visible semantic segmentation by combining lightweight Short-Term Dense Concatenate (STDC) network backbones with an asymmetric dual-to-single branch fusion mechanism.

```
Visible (RGB)  [3, H, W]  ───► STDC Stage 1, 2, 3 ──┐
                                                    ├──► CS-AFM Fusion ──► Single Stream (Stage 4, 5) ──► Context Module ──┐
Thermal (LWIR) [1, H, W]  ───► STDC Stage 1, 2, 3 ──┘                                                                     │
                                  │      │      │                                                                          │
                                  ▼      ▼      ▼                                                                          ▼
                           Skip connections (f1, f2, f3) ────────────────────────────────────────────────────────► Progressive Decoder
                                                                                                                           │
                                                                                                                   Final Logits [9, H, W]
```

### CSFNet-1 vs. CSFNet-2 Structural Differences

Both variants share the same channel dimensions across stages, allowing identical fusion modules and decoders. The difference lies in the backbone depth within Stages 3, 4, and 5:

| Component                       |   CSFNet-1 (STDC1)   |   CSFNet-2 (STDC2)   | Output Stride | Output Channels |
| :------------------------------ | :------------------: | :------------------: | :-----------: | :-------------: |
| **Stage 1 (`x2`)**      |  1x `ConvX` block  |  1x `ConvX` block  |      2x      |       32       |
| **Stage 2 (`x4`)**      |  1x `ConvX` block  |  1x `ConvX` block  |      4x      |       64       |
| **Stage 3 (`x8`)**      | 2x `CatBottleneck` | 4x `CatBottleneck` |      8x      |       256       |
| **Stage 4 (`x16`)**     | 2x `CatBottleneck` | 5x `CatBottleneck` |      16x      |       512       |
| **Stage 5 (`x32`)**     | 2x `CatBottleneck` | 3x `CatBottleneck` |      32x      |      1024      |
| **Total Backbone Blocks** |       8 blocks       |      14 blocks      |      —      |       —       |
| **Parameters**            |   **11.30M**   |   **19.36M**   |      —      |       —       |
| **Inference Speed**       | **106.3 FPS** |  **72.7 FPS**  |      —      |       —       |
| **FLOPs (480x640)**       |     ~21.4 GFLOPs     |     ~34.8 GFLOPs     |      —      |       —       |

* **CatBottleneck Block:** Employs hierarchical multi-scale feature aggregation with multi-branch convolutions (`1x1` projection followed by sequential `3x3` convolutions), concatenating output representations along the channel dimension. In downsampling layers (stride = 2), an Average Pooling (`skip`) and an Averaged Downsampling Layer (`avd_layer`) preserve high-frequency details.

### Dual-to-Single Branch Encoder

1. **Dual Encoders (Stages 1–3):** The model processes 3-channel RGB and 1-channel thermal inputs through parallel STDC paths. Stage 1 and Stage 2 intermediate representations undergo cross-modal interaction via CS-AFM blocks (`f1` at stride 2, `f2` at stride 4).
2. **Merging Point (Stage 3):** At stride 8, the RGB and Thermal representations pass through the final cross-modal fusion block (`f3`), fusing into a single unified feature representation.
3. **Single Stream (Stages 4–5):** Stages 4 and 5 process only the merged feature map. This reduces compute and memory by eliminating parallel streams at high-channel stages.

### Cosine Similarity Attentional Feature Fusion (CS-AFM)

Rather than naive concatenation or addition, CS-AFM computes pairwise cosine similarity between modal feature vectors to evaluate complementary spatial correlations:

$$
\mathbf{S}_{i,j} = \frac{\mathbf{f}_{\text{rgb}}(i, j) \cdot \mathbf{f}_{\text{th}}(i, j)}{\|\mathbf{f}_{\text{rgb}}(i, j)\|_2 \|\mathbf{f}_{\text{th}}(i, j)\|_2 + \epsilon}
$$

1. The spatial features are passed through adaptive average pooling to calibrate global context.
2. Channel-wise attention maps are computed using $1 \times 1$ convolutions followed by sigmoid normalization.
3. Feature maps are recalibrated and fused with a residual skip connection:

$$
\mathbf{F}_{\text{fused}} = \mathbf{F}_{\text{rgb}} + \mathbf{F}_{\text{th}} + \text{Conv}_{1 \times 1}\big(\text{Attn}(\mathbf{F}_{\text{rgb}}, \mathbf{F}_{\text{th}}) \odot (\mathbf{F}_{\text{rgb}} \oplus \mathbf{F}_{\text{th}})\big)
$$

### Context Module (CM)

Applied to Stage 5 feature maps ($1024 \times 15 \times 20$), the Context Module captures horizontal and vertical driving scene context:

- Reduces channels from 1024 to 256 using an adaptive pooling layer and a $1 \times 1$ conv.
- Processes features with asymmetric strip convolutions: horizontal ($1 \times 4$) and vertical ($4 \times 1$) kernels.
- Sums the directional contexts, upsamples back to spatial resolution, and projects to 32 decoder channels.

### Lightweight Progressive Bilateral Decoder

The decoder reconstructs fine-grained resolution through cascaded bilateral upsampling stages:

- **Upsampling 1 ($2\times$):** Fuses Context Module output ($32 \text{ ch}$) with Stage 4 features ($512 \to 32 \text{ ch}$ via agent conv).
- **Upsampling 2 ($2\times$):** Fuses with Stage 3 features ($256 \to 64 \text{ ch}$ via agent conv).
- **Upsampling 3 ($4\times$):** Fuses with Stage 1 features ($32 \to 32 \text{ ch}$ via agent conv).
- **Final Projection ($2\times$):** Upsamples to full resolution ($480 \times 640$) and projects to 9 output logits.

---

## Dataset & Preprocessing

The experiments use the MFNet benchmark (`ir_seg_dataset/`):

- **Pairs:** 1,569 calibrated, synchronized 4-channel image pairs (RGB + LWIR thermal)
- **Spatial Resolution:** $480 \times 640$
- **Splits:**
  - Training: 784 image pairs
  - Validation: 392 image pairs
  - Test: 393 image pairs (split into 205 Daytime and 188 Nighttime images)

### Class Definitions & Imbalance

| ID | Class          | Color (RGB)        | Frequency (%) | Loss Weight (ENet Log) |
| :-: | :------------- | :----------------- | :-----------: | :--------------------: |
| 0 | `unlabeled`  | `[40, 40, 40]`   |    91.87%    |         0.167         |
| 1 | `car`        | `[220, 20, 60]`  |     4.53%     |         1.104         |
| 2 | `person`     | `[65, 105, 225]` |     1.38%     |         1.104         |
| 3 | `bike`       | `[218, 165, 32]` |     0.43%     |         1.104         |
| 4 | `curve`      | `[138, 43, 226]` |     0.64%     |         1.104         |
| 5 | `car_stop`   | `[255, 69, 0]`   |     0.55%     |         1.104         |
| 6 | `guardrail`  | `[0, 206, 209]`  |     0.15%     |         1.104         |
| 7 | `color_cone` | `[255, 127, 80]` |     0.18%     |         1.104         |
| 8 | `bump`       | `[50, 205, 50]`  |     0.29%     |         1.104         |

### Normalization Statistics

Computed empirically over the training split:

- **RGB Mean:** `[0.2215, 0.2586, 0.2299]`, **Std:** `[0.2280, 0.2318, 0.2367]`
- **Thermal Mean:** `[0.3954]`, **Std:** `[0.1848]`

---

## Benchmark Results

All models were evaluated on the test set using standard semantic segmentation metrics: Pixel Accuracy (PA), Overall Mean Intersection over Union (mIoU across all 9 classes), and Foreground Mean Intersection over Union (mIoU over the 8 foreground classes).

### Head-to-Head Model Comparison

| Evaluation Metric            | CSFNet-1 (Scratch Baseline) | **CSFNet-2 (Pretrained STDC2)** | Official Paper (CSFNet-2) |
| :--------------------------- | :-------------------------: | :-----------------------------------: | :-----------------------: |
| **Backbone**           |   STDC1 (`STDCNet813`)   |   **STDC2 (`STDCNet1446`)**   |  STDC2 (`STDCNet1446`)  |
| **Pretrained Weights** |     None (Random Init)     |     **ImageNet Pretrained**     |    ImageNet Pretrained    |
| **Parameters**         |           11.30M           |           **19.36M**           |          19.36M          |
| **Epochs**             |             71             |             **120**             |            600            |
| **Validation mIoU**    |           39.88%           |           **50.12%**           |            —            |
| **Overall Test mIoU**  |           38.46%           |      **49.39% (+10.93%)**      |          59.98%          |
| **Foreground mIoU**    |           31.29%           |      **43.41% (+12.12%)**      |          ~54.50%          |
| **Pixel Accuracy**     |           95.55%           |       **97.16% (+1.61%)**       |          ~97.50%          |
| **Inference FPS**      |          106.3 FPS          |          **72.7 FPS**          |          ~72 FPS          |

### Per-Class IoU Breakdown

| Class          | CSFNet-1 (Scratch) | **CSFNet-2 (Pretrained)** |  Gain  |
| :------------- | :----------------: | :-----------------------------: | :-----: |
| `unlabeled`  |       95.79%       |        **97.23%**        | +1.44% |
| `car`        |       67.67%       |        **78.16%**        | +10.49% |
| `person`     |       59.51%       |        **68.28%**        | +8.77% |
| `bike`       |       35.77%       |        **56.22%**        | +20.45% |
| `curve`      |       29.06%       |        **41.37%**        | +12.31% |
| `car_stop`   |       9.28%       |        **27.04%**        | +17.76% |
| `guardrail`  |       0.00%       |         **0.00%**         |  0.00%  |
| `color_cone` |       24.02%       |        **42.92%**        | +18.90% |
| `bump`       |       25.02%       |        **33.29%**        | +8.27% |

### Daytime vs. Nighttime Robustness

Evaluation on the disjoint day (`test_day.txt`, 205 samples) and night (`test_night.txt`, 188 samples) subsets:

| Split               | Images | Pixel Accuracy |   Overall mIoU   | Foreground mIoU | Key Class Highlights                    |
| :------------------ | :----: | :------------: | :--------------: | :-------------: | :-------------------------------------- |
| **Daytime**   |  205  |     97.22%     | **49.79%** |     43.01%     | Car: 83.51%, Bike: 63.75%               |
| **Nighttime** |  188  |     97.09%     | **47.64%** |     41.45%     | Person: **70.56%**, Curve: 47.18% |

Under dark conditions, thermal infrared sensing provides strong thermal contrast for human bodies, increasing pedestrian IoU from 66.54% (day) to **70.56% (night)**.

---

## Environment & Installation

### Hardware & Software Requirements

- **OS:** Windows 10/11 or Linux
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with $\ge 4 \text{ GB}$ VRAM (tested on RTX 3050 6GB Laptop GPU)
- **PyTorch:** $\ge 2.0.0$ with CUDA support

### Setup

```powershell
# Clone or navigate to the workspace
cd "C:\#### CSE\#####Pytorch\RGBT_Segmentation"

# Activate your virtual environment
"C:\#### CSE\#####Pytorch\pytorch_env\Scripts\Activate.ps1"

# Install dependencies if not already present
pip install -r requirements.txt
```

---

## Pretrained Weights

Pretrained ImageNet backbone weights are downloaded into the `pretrained/` folder:

```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" scripts/download_weights.py
```

This retrieves:

- `pretrained/STDCNet1446_76.47.tar` (STDC2 backbone, 47.7 MB)
- `pretrained/STDCNet813M_73.91.tar` (STDC1 backbone, 32.3 MB)

### Dual-Stream Weight Transfer

When loading backbone weights:

- **RGB Stream:** Direct 1-to-1 transfer across all convolutional and batch normalization layers (318 tensor parameters).
- **Thermal Stream:** The first convolution (`3x3`, 3 in-channels) is adapted to 1 in-channel by taking the mean across the RGB input dimension:
  $$
  \mathbf{W}_{\text{thermal}} = \frac{1}{3} \sum_{c=0}^2 \mathbf{W}_{\text{rgb}}[:, c:c+1, :, :]
  $$

  All subsequent layers transfer directly.

---

## Training

Train CSFNet with Automatic Mixed Precision (AMP):

### Train CSFNet-2 (Recommended)

```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py `
  --backbone STDCNet1446 `
  --pretrained `
  --epochs 150 `
  --batch-size 4 `
  --lr 0.001 `
  --loss combined_ce_dice `
  --save-dir runs/csfnet2_pretrained
```

### Train CSFNet-1

```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py `
  --backbone STDCNet813 `
  --pretrained `
  --epochs 100 `
  --batch-size 4 `
  --save-dir runs/csfnet1_mfnet
```

### Key Arguments:

- `--backbone`: Choice of `STDCNet1446` (CSFNet-2) or `STDCNet813` (CSFNet-1).
- `--pretrained` / `--no-pretrained`: Toggles ImageNet weight loading.
- `--loss`: Options: `combined_ce_dice`, `weighted_ce`, `focal`, `ce`.
- `--weight-strategy`: Options: `enet_log`, `median_frequency`, `none`.
- `--amp`: Automatic Mixed Precision (keeps VRAM below 2 GB).
- `--patience`: Early stopping patience based on validation mIoU (default: 25).

Outputs are logged continuously to:

- `training_metrics.csv` and `training_metrics.json`
- `training_dashboard.png` (updated live every epoch)
- `best_model.pth` (saved whenever validation mIoU improves)

---

## Evaluation

The evaluation script auto-detects the backbone architecture from checkpoint keys:

```powershell
# Evaluate overall test set
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" evaluate.py `
  --checkpoint results/csfnet2_best_model.pth `
  --split test `
  --output-dir runs/eval_test

# Evaluate daytime test set
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" evaluate.py `
  --checkpoint results/csfnet2_best_model.pth `
  --split test_day `
  --output-dir runs/eval_test_day

# Evaluate nighttime test set
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" evaluate.py `
  --checkpoint results/csfnet2_best_model.pth `
  --split test_night `
  --output-dir runs/eval_test_night
```

Each run generates:

- Full confusion matrix and metrics JSON (`metrics_<split>.json`)
- 15 side-by-side visual prediction panels (`visuals_<split>/`)

---

## Inference & Visualization

To run single-sample or batch inference and generate 4-panel visual figures:

```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" infer.py `
  --checkpoint results/csfnet2_best_model.pth `
  --samples 00001D 00055D 00678N 01402D `
  --output-dir runs/inferences
```

Generated outputs for each sample:

1. `<sample>_comparison.png`: 4-panel figure `[Visible RGB | Thermal (Inferno) | Ground Truth | Prediction]` with color legend.
2. `<sample>_mask_color.png`: Color-coded segmentation mask.
3. `<sample>_mask_raw.png`: Raw 8-bit integer mask with exact class IDs (`[0..8]`).

---

## Repository Structure

```
RGBT_Segmentation/
├── README.md                          # Full documentation & architectures
├── requirements.txt                   # Dependencies
├── train.py                           # Training script
├── evaluate.py                        # Evaluation script
├── infer.py                           # Inference script
│
├── configs/
│   └── default_config.py              # Configurations
│
├── data/
│   ├── dataset.py                     # Dataset loader
│   └── transforms.py                  # Augmentations
│
├── models/
│   ├── csfnet.py                      # CSFNet-1 & CSFNet-2 architectures
│   ├── stdc.py                        # STDC1 & STDC2 backbones + weight loader
│   ├── fusion.py                      # CS-AFM cross-attention fusion module
│   └── losses.py                      # Loss functions
│
├── utils/
│   ├── metrics.py                     # Metric calculations
│   ├── checkpoint.py                  # Checkpoint manager
│   └── visualization.py               # Mask colors & plotting
│
├── scripts/
│   ├── download_weights.py            # Download backbone weights
│   └── test_csfnet2.py                # Model test script
│
└── runs/
    ├── csfnet1_mfnet/                 # CSFNet-1 Baseline Run
    │   ├── best_model.pth             # ★ CSFNet-1 Best Model (95.7 MB)
    │   ├── training_metrics.csv       # Training CSV metrics
    │   ├── training_metrics.json      # Training JSON log
    │   ├── training_dashboard.png     # Loss & mIoU curves
    │   ├── eval_test/                 # Test evaluation metrics & visuals
    │   ├── eval_test_day/             # Daytime test metrics & visuals
    │   └── eval_test_night/           # Nighttime test metrics & visuals
    │
    └── csfnet2_pretrained/            # CSFNet-2 Pretrained Run
        ├── best_model.pth             # ★ CSFNet-2 Best Model (162.6 MB, via Git LFS)
        ├── training_metrics.csv       # Training CSV metrics
        ├── training_metrics.json      # Training JSON log
        ├── training_dashboard.png     # Loss & mIoU curves
        ├── eval_test/                 # Test evaluation metrics & visuals
        ├── eval_test_day/             # Daytime test metrics & visuals
        └── eval_test_night/           # Nighttime test metrics & visuals
```
