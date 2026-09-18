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

## 🚀 Quick Start & Reproducibility Guide

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/MayankV004/ASID-Extreme-Super-Resolution.git
cd ASID-Extreme-Super-Resolution

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Paths
Verify that `env/env.json` points to your active workspace path:
```json
{
    "work_dir": "/path/to/ASID-Extreme-Super-Resolution",
    "dataset_dir": "/path/to/ASID-Extreme-Super-Resolution/dataset",
    "train_dir": "/path/to/ASID-Extreme-Super-Resolution/train_logs"
}
```

### 3. Generate Extreme $\times 8$ Benchmarks
```bash
python data_tools/generate_x8_benchmarks.py
```
*(Prepares modcrop-8 HR targets and bicubic downscaled LR pairs for Set5, Set14, and Urban100).*

### 4. Evaluate Trained Pre-Upsampled Model
```bash
# Run multi-benchmark comparative evaluation (Bicubic vs. Zero-Shot vs. Ours)
python evaluate_preupsample_x8.py --datasets Set5 Set14 Urban100

# Or run Set5 evaluation via the native engine
python test.py -v "ASID_PreUpsample_X8_DIV2K" -s 1 --test_dataset_name Set5
```

### 5. Local Sanity Training (RTX 3050 Laptop GPU / 2 Minutes)
```bash
python data_tools/setup_mini_div2k.py
python train.py -opt train_yamls/train_ASID_mini_test.yaml
```

### 6. Cloud Training via Google Colab (Tesla T4 GPU)
Open [`ASID_Extreme_Super_Resolution.ipynb`](file:///home/streamliner/computer-vision-project/ASID_Extreme_Super_Resolution.ipynb) in Google Colab, set runtime to **T4 GPU**, and execute all cells to train on the complete 800-image DIV2K dataset ($51,200$ patches per epoch).

---

## 📂 Repository Architecture & Directory Layout

```
├── components/          # Neural network architectures (ASID, ASID_PreUpsample, ASIDd8)
├── ops/                 # Low-level operators, self-attention blocks, ESA, LayerNorm
├── train_yamls/         # Experiment configuration files and training hyperparameters
├── data_tools/          # Dataloaders (DIV2K, DF2K) and benchmark preparation scripts
├── docs/                # Project manual, final research report, and course proposal
├── tools/               # Visualization generators, Colab generator, MATLAB evaluators
├── train_scripts/       # Deep learning training engine and metric logging loops
├── test_scripts/        # Benchmark validation and inference routines
├── utilities/           # PSNR/SSIM metrics, LR schedulers, checkpoint managers
├── benchmark/           # Standard evaluation datasets (Set5, Set14, Urban100, B100)
├── train.py             # Main CLI entry point for training
├── test.py              # Main CLI entry point for testing
├── evaluate_preupsample_x8.py # Multi-benchmark evaluation & comparative table generator
└── ASID_Extreme_Super_Resolution.ipynb # One-click cloud training notebook for Google Colab
```

---

## 📚 Complete Project Documentation
* 🚀 **[Colab Training & .pth Optimization Guide](docs/README.md):** Step-by-step cloud training instructions, checkpoint management, and inference optimization techniques.
* 📄 **[Course Project Manual](docs/COURSE_PROJECT_MANUAL.md):** Architectural diagrams, hyperparameter engineering, viva presentation Q&A defense guide, and setup manual.
* 📄 **[Final Project Report](docs/FINAL_PROJECT_REPORT.md):** Academic report detailing quantitative metrics, zero-shot failure mechanics, and ablation analysis.
* 📄 **[Original Course Proposal](docs/CV_project.pdf):** Initial course proposal scoping extreme SISR research gaps.

---

## 🙏 Credits & Literature Attribution

This project is built upon the breakthrough research, network design, and official implementation of:

### Primary Paper
* **Paper Title:** *Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-Resolution*
* **Authors:** Karam Park, Jae Woong Soh, and Nam Ik Cho
* **Affiliation:** Department of Electrical and Computer Engineering, INMC, Seoul National University, Seoul, Korea
* **Venue:** Accepted for presentation at the **AAAI Conference on Artificial Intelligence (AAAI 2025)**

### Foundational Frameworks & Acknowledgments
* **Omni-SR Base Framework:** Built upon the open-source code and training foundations of *Omni-SR: Omni Aggregation Networks for Lightweight Image Super-Resolution* (CVPR 2023) by Richard Wang et al.
* **DIV2K Benchmark:** High-resolution dataset provided by the Computer Vision Lab, ETH Zurich (*NTIRE 2017 Challenge on Single Image Super-Resolution* by Timofte et al.).
* **Standard Test Sets:** Set5 (Bevilacqua et al.), Set14 (Zeyde et al.), Urban100 (Huang et al.), and B100 (Martin et al.).

### 🔗 Official External Links

| Resource | Description | Link |
| :--- | :--- | :--- |
| **ASID Paper (arXiv)** | Official AAAI 2025 Paper Page | [arXiv:2501.15774](https://arxiv.org/abs/2501.15774) |
| **ASID Paper (PDF)** | Full Text & Supplementary PDF | [arXiv PDF](https://arxiv.org/pdf/2501.15774) |
| **ASID GitHub** | Official PyTorch Code Repository by SNU | [saturnian77/ASID](https://github.com/saturnian77/ASID) |
| **Omni-SR GitHub** | Foundational Base Framework | [Francis0625/Omni-SR](https://github.com/Francis0625/Omni-SR/) |
| **DIV2K Dataset** | ETH Zurich CVL Dataset Portal | [DIV2K Official Page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| **Course Project Repo** | Extreme x8 Extension & Pre-Upsampling Optimization | [MayankV004/ASID-Extreme-Super-Resolution](https://github.com/MayankV004/ASID-Extreme-Super-Resolution) |

---

## 📜 Citations

```bibtex
@inproceedings{park2025efficient,
  title={Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-Resolution},
  author={Park, Karam and Soh, Jae Woong and Cho, Nam Ik},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

@inproceedings{wang2023omni,
  title={Omni Aggregation Networks for Lightweight Image Super-Resolution},
  author={Wang, Richard and Dong, Zheng and Gao, Chang and Zhang, Meng and Wang, Zhiyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={22378--22387},
  year={2023}
}

@inproceedings{timofte2017ntire,
  title={NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results},
  author={Timofte, Radu and Agustsson, Eirikur and Van Gool, Luc and Yang, Ming-Hsuan and Zhang, Lei and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={114--125},
  year={2017}
}
```

