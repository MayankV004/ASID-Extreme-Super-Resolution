# Google Colab Training & Checkpoint (.pth) Optimization Guide

Welcome to the end-to-end guide for training, evaluating, and optimizing the **Ultra-Lightweight Attention-Sharing Transformer for Extreme Super-Resolution ($\times 8$)**.

This document covers:
1. **[Step-by-Step Google Colab Training](#1-google-colab-training-walkthrough)**: Fast setup, downloading DIV2K, generating $\times 8$ LR pairs, running training, and monitoring losses.
2. **[Checkpoint (.pth) File Management](#2-checkpoint-pth-file-management)**: Directory structure, naming formats, and transferring checkpoints between cloud and local.
3. **[Using the `.pth` File for Evaluation & Benchmarking](#3-using-the-pth-file-for-evaluation--benchmarking)**: Running `test.py`, multi-dataset comparative evaluation, and visual figure generation.
4. **[Inference & Model Optimization Techniques](#4-inference--model-optimization-techniques)**: Standalone Python inference, FP16 half-precision, PyTorch 2.x `torch.compile`, and transfer fine-tuning.
5. **[Troubleshooting & FAQ](#5-troubleshooting--faq)**: Common errors and solutions.

---

## 🏗️ Workflow Overview

```mermaid
flowchart TD
    subgraph Cloud Training: Google Colab (Tesla T4 GPU)
        A["Open ASID_Extreme_Super_Resolution.ipynb"] --> B["Download 800 DIV2K Images (1.1 GB)"]
        B --> C["Generate x8 Bicubic LR Pairs"]
        C --> D["Update env.json Paths"]
        D --> E["Train / Fine-Tune Model: python train.py"]
        E --> F["Save Checkpoint: epoch1_ASID.pth (1.5 MB)"]
    end

    subgraph Checkpoint Transfer
        F -.->|"Download to Local"| G["train_logs/ASID_PreUpsample_X8_DIV2K/checkpoints/"]
    end

    subgraph Local Deployment & Optimization (RTX 3050 / CPU)
        G --> H["1. Native Benchmark: python test.py -v ASID_PreUpsample_X8_DIV2K"]
        G --> I["2. Comparative Multi-Set: python evaluate_preupsample_x8.py"]
        G --> J["3. Visual Comparisons: python tools/generate_visual_comparisons.py"]
        G --> K["4. Standalone Single-Image Inference / FP16 Optimization"]
    end
```

---

## 1. Google Colab Training Walkthrough

You can train the full model on Google Colab using either the pre-packaged notebook [`ASID_Extreme_Super_Resolution.ipynb`](../ASID_Extreme_Super_Resolution.ipynb) or by executing the cell-by-cell commands below.

### Step 1: Select GPU Runtime in Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. In the top navigation bar, go to **Runtime ➔ Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (or A100/V100 if available) and click **Save**.
4. Verify GPU allocation:
```bash
!nvidia-smi
```

---

### Step 2: Clone Repository or Mount Google Drive

#### Option A: Clone directly from GitHub (Recommended)
```bash
!git clone https://github.com/MayankV004/ASID-Extreme-Super-Resolution.git
%cd ASID-Extreme-Super-Resolution
```

#### Option B: Mount Google Drive (if using your personal Drive storage)
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/computer-vision-project
```

---

### Step 3: Install Required Dependencies
Install the lightweight deep learning and vision dependencies:
```bash
!pip install -q thop tensorboardX einops timm opencv-python pyyaml tqdm matplotlib
```

---

### Step 4: Download the DIV2K Training Dataset (800 Images)
Download and extract the official ETH Zurich DIV2K High-Resolution dataset:
```python
import os

%cd /content
!mkdir -p /content/DIV2K
%cd /content/DIV2K

# Download High-Resolution DIV2K train images (800 images, ~1.1 GB)
if not os.path.exists('DIV2K_train_HR.zip'):
    !wget -q http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    !unzip -q DIV2K_train_HR.zip

print("DIV2K Train HR downloaded and extracted successfully!")
```

---

### Step 5: Generate Extreme $\times 8$ Low-Resolution Training Pairs
Downscale the 800 training images by factor $8$ using PIL Bicubic interpolation:
```python
import os, glob
from PIL import Image
from tqdm import tqdm

hr_dir = '/content/DIV2K/DIV2K_train_HR'
lr_out_dir = '/content/DIV2K/DIV2K_train_LR_bicubic/X8'
os.makedirs(lr_out_dir, exist_ok=True)

hr_images = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
print(f"Downsampling {len(hr_images)} training images to x8...")

for img_path in tqdm(hr_images):
    base = os.path.basename(img_path).replace('.png', '')
    lr_save = os.path.join(lr_out_dir, f"{base}x8.png")
    if not os.path.exists(lr_save):
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            crop_w = (w // 8) * 8
            crop_h = (h // 8) * 8
            img_cropped = img.crop((0, 0, crop_w, crop_h))
            lr_img = img_cropped.resize((crop_w // 8, crop_h // 8), Image.Resampling.BICUBIC)
            lr_img.save(lr_save, 'PNG')

print("x8 Low-Resolution training images ready!")
```

---

### Step 6: Configure `env/env.json` for Colab Paths
Switch back to the repository root and point the DIV2K dataset path to `/content/DIV2K`:
```python
import json

%cd /content/ASID-Extreme-Super-Resolution

env_path = 'env/env.json'
with open(env_path, 'r') as f:
    env_cfg = json.load(f)

# Point to Colab fast local storage
env_cfg['path']['dataset_paths']['DIV2K'] = '/content/DIV2K'

with open(env_path, 'w') as f:
    json.dump(env_cfg, f, indent=4)

print("env.json successfully updated for Colab!")
```

---

### Step 7: Launch Training Command

#### Option 1: Train Pre-Upsampled $\times 8$ Architecture (Our Solution)
```bash
!python train.py -v "ASID_PreUpsample_X8_DIV2K" -p train --train_yaml "train_ASID_PreUpsample_X8_DIV2K.yaml"
```

#### Option 2: Fine-Tuning with Backbone Weight Transfer (Faster Convergence)
To initialize the attention-sharing backbone weights from a pretrained $\times 4$ model (e.g., `epoch0_ASID.pth`):
```bash
!python train.py -v "ASID_PreUpsample_X8_DIV2K" -p finetune -e 0 --train_yaml "train_ASID_PreUpsample_X8_DIV2K.yaml"
```

#### Option 3: Direct $\times 8$ ASID Baseline (Comparison Model)
```bash
!python train.py -v "ASID_Direct_X8_DIV2K" -p train --train_yaml "train_ASID_X8_DIV2K.yaml"
```

---

### Step 8: Monitor Loss Curves via TensorBoard
Run TensorBoard directly inside Google Colab:
```python
%load_ext tensorboard
%tensorboard --logdir train_logs/ASID_PreUpsample_X8_DIV2K/summary
```

---

### Step 9: Download the `.pth` Checkpoint to Your Local Machine
Once training completes or saves a checkpoint, download the file:
```python
from google.colab import files

ckpt_file = 'train_logs/ASID_PreUpsample_X8_DIV2K/checkpoints/epoch1_ASID.pth'
files.download(ckpt_file)
```

---

## 2. Checkpoint (`.pth`) File Management

### Checkpoint Storage Hierarchy
All trained models are saved under the `train_logs/` directory formatted by experiment version name:

```
train_logs/
├── ASID_PreUpsample_X8_DIV2K/               # Experiment Version Name
│   ├── checkpoints/
│   │   └── epoch1_ASID.pth                 # Saved PyTorch model weights (1.5 MB)
│   ├── model_config.json                   # Snapshot of hyperparameters & paths
│   ├── samples/                            # Validation crops per epoch
│   └── summary/                            # TensorBoard event logs
└── ASID_X4_DIV2K/
    └── checkpoints/
        └── epoch0_ASID.pth                 # Pretrained x4 baseline backbone weights
```

### Checkpoint Specifications
* **File Format:** Standard PyTorch `state_dict` dictionary (`torch.save(model.state_dict(), path)`).
* **Model Parameters:** `313,104` floating-point parameters.
* **File Size:** $\approx 1.5\text{ MB}$ (ultra-lightweight footprint suitable for edge and mobile).

---

## 3. Using the `.pth` File for Evaluation & Benchmarking

Once you have your `.pth` file placed in `train_logs/ASID_PreUpsample_X8_DIV2K/checkpoints/`:

### A. Run Benchmark Evaluation with the Native Testing Engine (`test.py`)
This evaluates the trained checkpoint on benchmark datasets (`Set5`, `Set14`, `Urban100`, `B100`), calculates luminance ($Y$-channel) PSNR & SSIM, and outputs reconstructed images to `test_logs/`:

```bash
# Evaluate epoch 1 on Set5
python test.py -v "ASID_PreUpsample_X8_DIV2K" -s 1 --test_dataset_name Set5

# Evaluate on Set14
python test.py -v "ASID_PreUpsample_X8_DIV2K" -s 1 --test_dataset_name Set14

# Evaluate on Urban100
python test.py -v "ASID_PreUpsample_X8_DIV2K" -s 1 --test_dataset_name Urban100
```

**Expected Output:**
```text
Start to run test script: test_scripts.tester_Matlab
Test version: ASID_PreUpsample_X8_DIV2K
Prepare the test dataloader...
processing Set5 images...
loaded trained backbone model epoch 1...!
Elapsed [0:00:01.12], PSNR: 25.7715, SSIM: 0.7250
```

---

### B. Run Multi-Dataset Comparative Evaluation (`evaluate_preupsample_x8.py`)
Generates the comprehensive benchmark comparison table contrasting **Bicubic baseline**, **Zero-Shot Bilinear**, and **Trained Model**:

```bash
python evaluate_preupsample_x8.py --datasets Set5 Set14 Urban100
```

---

### C. Generate Side-by-Side Visual Comparison Figures
Renders zoomed high-contrast crops with bounding boxes and quantitative metric badges:

```bash
python tools/generate_visual_comparisons.py
```
*(Saved output PNGs will be placed in `SR_Results/visual_comparisons/`).*

---

## 4. Inference & Model Optimization Techniques

### Standalone Python Script: Super-Resolving Custom Images
Here is a complete, lightweight script to load the `.pth` weights and upscale an arbitrary low-resolution image by $\times 8$:

```python
import torch
import numpy as np
from PIL import Image
from components.ASID_PreUpsample import ASID_PreUpsample
from utilities.yaml_config import getConfigYaml

# 1. Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load model architecture from YAML
config = getConfigYaml('train_yamls/train_ASID_PreUpsample_X8_DIV2K.yaml')
model = ASID_PreUpsample(num_feat=config['feature_num'], **config['module_params'])

# 3. Load trained .pth checkpoint
ckpt_path = 'train_logs/ASID_PreUpsample_X8_DIV2K/checkpoints/epoch1_ASID.pth'
state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

# 4. Prepare input image (RGB normalized to [0, 1])
input_img = Image.open('path/to/your_low_res_image.png').convert('RGB')
lr_tensor = torch.from_numpy(np.array(input_img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

# 5. Run inference
with torch.no_grad():
    sr_tensor = model(lr_tensor)

# 6. Convert back to image
sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy()
output_img = Image.fromarray(sr_np)
output_img.save('super_resolved_x8_output.png')
print("Super-resolved image saved successfully!")
```

---

### Optimization Techniques for High-Performance Inference

#### 1. FP16 (Half-Precision) Acceleration
Cut VRAM usage by $50\%$ and boost inference speed on modern NVIDIA GPUs (RTX series / Tesla T4 Tensor Cores):

```python
model = model.half().to('cuda')
lr_tensor = lr_tensor.half().to('cuda')

with torch.no_grad():
    sr_tensor = model(lr_tensor)
```

#### 2. PyTorch 2.x Kernel Fusion (`torch.compile`)
Fuse multi-head self-attention kernels and layer norms to eliminate Python overhead:

```python
# Optimize execution graph for inference
compiled_model = torch.compile(model, mode="reduce-overhead")

with torch.no_grad():
    sr_tensor = compiled_model(lr_tensor)
```

#### 3. Dynamic Arbitrary Resolution Handling
Because windowed self-attention requires dimensions divisible by the window size ($8$), our model includes built-in dynamic padding:
```python
# Inside ASID_PreUpsample.forward():
x = self.check_image_size(x) # Automatically pads edge reflections
```
This guarantees that input images of any irregular resolution ($H \times W$) can be passed directly without manual cropping.

---

## 5. Troubleshooting & FAQ

### Q1: Colab throws `CUDA out of memory` during training.
* **Fix:** Open `train_yamls/train_ASID_PreUpsample_X8_DIV2K.yaml` and reduce `batch_size` from `16` to `8` or `4`, or reduce `lr_patch_size` from `48` to `32`.

### Q2: Warning: `Unexpected key(s) in state_dict: total_ops, total_params`.
* **Fix:** These keys are metadata added by FLOP profilers (`thop`). They do not affect network weights. Always load using `model.load_state_dict(state_dict, strict=False)` to cleanly ignore profiling metadata.

### Q3: `FileNotFoundError: DIV2K_train_HR` when training on Colab.
* **Fix:** Ensure step 6 was executed to update `env/env.json`. Verify by running:
```bash
python -c "import json; print(json.load(open('env/env.json'))['path']['dataset_paths']['DIV2K'])"
```
It should output `/content/DIV2K`.

---

## 📚 Related Documentation
* 📄 **[Course Project Manual](COURSE_PROJECT_MANUAL.md):** Complete theoretical derivation, hyperparameter engineering, and viva presentation Q&A defense.
* 📄 **[Final Project Report](FINAL_PROJECT_REPORT.md):** Academic report detailing quantitative metrics, zero-shot failure mechanics, and ablation analysis.
* 🚀 **[Colab One-Click Notebook](../ASID_Extreme_Super_Resolution.ipynb):** Ready-to-run training notebook.
