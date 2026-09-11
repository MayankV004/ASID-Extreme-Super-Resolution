# Ultra-Lightweight Attention-Sharing Transformer for Extreme Super-Resolution (×8)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Parameters 313K](https://img.shields.io/badge/Parameters-313K-success.svg)](#architecture--complexity)
[![Scale x8](https://img.shields.io/badge/Scale-x8_Extreme-brightgreen.svg)](#benchmark-results)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Computer Vision and Deep Learning Course Research Project**  
> Built upon **ASID (AAAI 2025)**: *Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-Resolution*.

---

## 📌 Abstract & Research Problem

While Vision Transformers (ViTs) achieve remarkable reconstruction quality in Single Image Super-Resolution (SISR), their quadratic computational complexity limits practical deployment. The **Attention-Sharing Information Distillation Transformer (ASID)** (AAAI 2025) mitigated this bottleneck by sharing self-attention maps across cascaded blocks, using only **~313K parameters**.

However, the original ASID paper validated models exclusively on moderate upscaling factors ($\times 2, \times 3, \times 4$). Under **extreme scaling ($\times 8$)**, input spatial dimensions are severely reduced ($32 \times 32 \to 256 \times 256$), causing window self-attention layers to suffer from **receptive-field starvation**.

### 💡 Our Contribution: Pre-Upsampling Intervention
We implemented and trained an end-to-end **Pre-Upsampling Attention-Sharing Architecture**:
1. Spatial resolution is doubled prior to attention distillation ($2H \times 2W$), restoring the contextual receptive field of self-attention heads.
2. Maintains **zero parameter overhead** (313,104 parameters, matching the original $\times 4$ model).
3. Achieves **$+1.37\text{ dB}$ on Set5** and **$+0.92\text{ dB}$ on Urban100** over the standard bicubic baseline.

---

## 🏆 Quantitative Benchmark Results

Evaluated on standard benchmarks using luminance ($Y$-channel) PSNR (dB) and SSIM:

| Benchmark Dataset | Bicubic $\times 8$ (Baseline) | Zero-Shot ($\times 2$ Interp + ASID $\times 4$) | **Ours: Trained Pre-Upsample $\times 8$** | **Net Gain over Baseline** |
| :--- | :---: | :---: | :---: | :---: |
| **Set5** | 24.40 dB / 0.6583 | 23.77 dB / 0.6436 | **25.77 dB / 0.7250** | **+1.37 dB / +0.067 SSIM** 🚀 |
| **Set14** | 22.96 dB / 0.5693 | 22.50 dB / 0.5563 | **23.93 dB / 0.6157** | **+0.97 dB / +0.046 SSIM** 🚀 |
| **Urban100** | 20.75 dB / 0.5170 | 20.39 dB / 0.5062 | **21.67 dB / 0.5732** | **+0.92 dB / +0.056 SSIM** 🚀 |

---

## 🖼️ Qualitative Visual Comparisons

Side-by-side zoomed-in visual crops highlighting high-frequency edge restoration and artifact suppression:

| Benchmark Image | Visual Comparison Crop (Bicubic vs. Zero-Shot vs. Ours vs. HR) |
| :--- | :--- |
| **Butterfly (Set5)** | ![Butterfly Comparison](SR_Results/visual_comparisons/butterfly_comparison_x8.png) |
| **Baby (Set5)** | ![Baby Comparison](SR_Results/visual_comparisons/baby_comparison_x8.png) |
| **Zebra (Set14)** | ![Zebra Comparison](SR_Results/visual_comparisons/zebra_comparison_x8.png) |
| **Urban100 (img_004)** | ![Urban100 Comparison](SR_Results/visual_comparisons/img_004_comparison_x8.png) |

---

## 🏗️ Architecture & Complexity

```
Low-Resolution (H x W) ──► [x2 Pre-Upsampling] ──► Intermediate (2H x 2W)
                                                         │
                     ┌───────────────────────────────────┘
                     ▼
             [Input Conv: 48ch]
                     │
                     ▼
             [IDSG_A (Self-Attention Block)] ──(Shared Maps)──┐
                     │                                        │
                     ▼                                        │
             [IDSG Block 1] ◄─────────────────────────────────┤
                     │                                        │
                     ▼                                        │
             [IDSG Block 2] ◄─────────────────────────────────┘
                     │
                     ▼
             [Output Conv: 48ch] ──► (+) Residual ──► [x4 PixelShuffle] ──► Output HR (8H x 8W)
```

| Model | Parameters | Inference Latency (RTX 3050 Laptop) | FLOPs ($48 \times 48$ patch) |
| :--- | :---: | :---: | :---: |
| **Direct ASID $\times 8$** | 375,456 | 1.1 ms | 0.88 G |
| **Pre-Upsampled ASID $\times 8$ (Ours)** | **313,104** | **1.2 ms** | **0.84 G** |

---

## ⚙️ GPU Training Approach (Two-Tier Hybrid Pipeline)

1. **Tier 1: Local Sanity Testing (RTX 3050 Laptop GPU, 4 GB VRAM)**
   * Automated mini-DIV2K test pipeline (15 images) to verify forward pass, gradient flow, and checkpoint saving in 2 minutes.
2. **Tier 2: Cloud Datacenter Training (Google Colab Tesla T4 GPU, 16 GB VRAM)**
   * End-to-end fine-tuning on full 800-image DIV2K dataset ($51,200$ patches per epoch).
   * Weight transfer initialization from pretrained $\times 4$ backbone weights enabled rapid convergence within 1 epoch.

---

## 🚀 Quick Start & Reproducibility

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/MayankV004/ASID-Extreme-Super-Resolution.git
cd ASID-Extreme-Super-Resolution

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install thop tensorboardX einops timm opencv-python pyyaml tqdm matplotlib
```

### 2. Generate x8 Benchmarks
```bash
python data_tools/generate_x8_benchmarks.py
```

### 3. Evaluate the Trained Model
```bash
# Run evaluation on Set5 using our trained checkpoint
python test.py -v "ASID_PreUpsample_X8_DIV2K" -s 1 --test_dataset_name Set5

# Run comparative metric script
python evaluate_preupsample_x8.py --datasets Set5 Set14 Urban100
```

### 4. Cloud Training via Google Colab
Upload [ASID_Extreme_Super_Resolution.ipynb](file:///home/streamliner/computer-vision-project/ASID_Extreme_Super_Resolution.ipynb) to Google Colab and run all cells with a free T4 GPU.

---

## 📚 Complete Project Documentation
* 📄 **[Course Project Manual](cv-project%20docs/COURSE_PROJECT_MANUAL.md):** Theoretical derivations, hyperparameter engineering, and viva presentation defense guide.
* 📄 **[Final Project Report](cv-project%20docs/FINAL_PROJECT_REPORT.md):** Full academic report matching course proposal specifications.
* 📄 **[Original Course Proposal](cv-project%20docs/CV_project.pdf):** Initial proposal scoping extreme SISR research gaps.

---

## Citation
```bibtex
@inproceedings{park2025efficient,
  title={Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-Resolution},
  author={Park, Karam and Soh, Jae Woong and Cho, Nam Ik},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
