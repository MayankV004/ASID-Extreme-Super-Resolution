#!/usr/bin/env python3
"""
Generate publication-quality visual comparison figures for Extreme x8 Super-Resolution.
Renders side-by-side cropped patches with bounding boxes and PSNR/SSIM annotations.
"""

import os
import sys

# Ensure repository root is in sys.path and is the working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from utilities.utilities import calculate_psnr, calculate_ssim

OUT_DIR = os.path.join(REPO_ROOT, "SR_Results/visual_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

# Define test cases: (dataset, base_name, crop_box: (x, y, w, h))
CASES = [
    {
        "name": "Butterfly (Set5)",
        "dataset": "Set5",
        "basename": "butterfly",
        "hr_path": "benchmark/HR/Set5/x8/butterfly_HR_x8.png",
        "lr_path": "benchmark/LR/LRBI/Set5/x8/butterfly_LRBI_x8.png",
        "zero_shot_path": "SR_Results/x8_experiments/Set5/butterfly_bilinear_asidx4.png",
        "ours_path": "SR/BI/ASID_PreUpsample_X8_DIV2K/Set5/x8/butterfly_ASID_PreUpsample_X8_DIV2K_x8.png",
        "crop": (120, 90, 60, 60) # x, y, width, height on HR
    },
    {
        "name": "Baby (Set5)",
        "dataset": "Set5",
        "basename": "baby",
        "hr_path": "benchmark/HR/Set5/x8/baby_HR_x8.png",
        "lr_path": "benchmark/LR/LRBI/Set5/x8/baby_LRBI_x8.png",
        "zero_shot_path": "SR_Results/x8_experiments/Set5/baby_bilinear_asidx4.png",
        "ours_path": "SR/BI/ASID_PreUpsample_X8_DIV2K/Set5/x8/baby_ASID_PreUpsample_X8_DIV2K_x8.png",
        "crop": (200, 200, 100, 100)
    },
    {
        "name": "Zebra (Set14)",
        "dataset": "Set14",
        "basename": "zebra",
        "hr_path": "benchmark/HR/Set14/x8/zebra_HR_x8.png",
        "lr_path": "benchmark/LR/LRBI/Set14/x8/zebra_LRBI_x8.png",
        "zero_shot_path": "SR_Results/x8_experiments/Set14/zebra_bilinear_asidx4.png",
        "ours_path": "SR/BI/ASID_PreUpsample_X8_DIV2K/Set14/x8/zebra_ASID_PreUpsample_X8_DIV2K_x8.png",
        "crop": (180, 130, 90, 90)
    },
    {
        "name": "Urban100 img_004",
        "dataset": "Urban100",
        "basename": "img_004",
        "hr_path": "benchmark/HR/Urban100/x8/img_004_HR_x8.png",
        "lr_path": "benchmark/LR/LRBI/Urban100/x8/img_004_LRBI_x8.png",
        "zero_shot_path": "SR_Results/x8_experiments/Urban100/img_004_bilinear_asidx4.png",
        "ours_path": "SR/BI/ASID_PreUpsample_X8_DIV2K/Urban100/x8/img_004_ASID_PreUpsample_X8_DIV2K_x8.png",
        "crop": (250, 200, 120, 120)
    }
]

def render_comparison(case):
    print(f"Generating visual comparison for {case['name']}...")
    
    # 1. Load Ground Truth
    hr_img = Image.open(case["hr_path"]).convert("RGB")
    w_hr, h_hr = hr_img.size
    hr_np = np.array(hr_img)
    
    # 2. Load LR and create Bicubic x8
    lr_img = Image.open(case["lr_path"]).convert("RGB")
    bicubic_img = lr_img.resize((w_hr, h_hr), Image.Resampling.BICUBIC)
    bicubic_np = np.array(bicubic_img)
    
    # 3. Load Zero-Shot (Bilinear x2 + ASID x4)
    if os.path.exists(case["zero_shot_path"]):
        zero_shot_img = Image.open(case["zero_shot_path"]).convert("RGB")
        zero_shot_np = np.array(zero_shot_img)[:h_hr, :w_hr, :]
    else:
        zero_shot_np = bicubic_np
        
    # 4. Load Ours (Trained Pre-Upsample x8)
    ours_img = Image.open(case["ours_path"]).convert("RGB")
    ours_np = np.array(ours_img)[:h_hr, :w_hr, :]
    
    # Calculate Metrics for this specific image
    psnr_bic = calculate_psnr(bicubic_np, hr_np)
    ssim_bic = calculate_ssim(bicubic_np, hr_np)
    
    psnr_zs = calculate_psnr(zero_shot_np, hr_np)
    ssim_zs = calculate_ssim(zero_shot_np, hr_np)
    
    psnr_ours = calculate_psnr(ours_np, hr_np)
    ssim_ours = calculate_ssim(ours_np, hr_np)
    
    x, y, cw, ch = case["crop"]
    # Ensure crop is within bounds
    x = max(0, min(x, w_hr - cw))
    y = max(0, min(y, h_hr - ch))
    
    crop_hr = hr_np[y:y+ch, x:x+cw]
    crop_bic = bicubic_np[y:y+ch, x:x+cw]
    crop_zs = zero_shot_np[y:y+ch, x:x+cw]
    crop_ours = ours_np[y:y+ch, x:x+cw]
    
    # Create Figure: 1 big overview on the left, 4 crop panels on the right
    fig = plt.figure(figsize=(16, 8), dpi=200)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1, 1, 1], height_ratios=[1, 1], wspace=0.12, hspace=0.25)
    
    # Left: Full Ground Truth with Bounding Box
    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(hr_np)
    rect = patches.Rectangle((x, y), cw, ch, linewidth=2.5, edgecolor='#00FF66', facecolor='none', linestyle='-')
    ax_full.add_patch(rect)
    ax_full.set_title(f"{case['name']}\n(Scale: $\\times 8$)", fontsize=13, fontweight='bold', pad=10)
    ax_full.axis('off')
    
    # Panel 1: Bicubic
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(crop_bic)
    ax1.set_title(f"Bicubic $\\times 8$\n{psnr_bic:.2f} dB / {ssim_bic:.4f}", fontsize=11, fontweight='semibold')
    ax1.axis('off')
    for spine in ax1.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)
        
    # Panel 2: Zero-Shot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(crop_zs)
    ax2.set_title(f"Zero-Shot Bilinear+ASID\n{psnr_zs:.2f} dB / {ssim_zs:.4f}", fontsize=11, fontweight='semibold')
    ax2.axis('off')
    
    # Panel 3: Ours (Trained)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(crop_ours)
    ax3.set_title(f"Ours (Trained Model)\n{psnr_ours:.2f} dB / {ssim_ours:.4f}", fontsize=11, fontweight='bold', color='#009933')
    ax3.axis('off')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#00CC44')
        spine.set_linewidth(2.5)
        
    # Panel 4: Ground Truth
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(crop_hr)
    ax4.set_title("Ground Truth (HR)\nPSNR / SSIM", fontsize=11, fontweight='semibold')
    ax4.axis('off')
    
    # Save figure
    save_path = os.path.join(OUT_DIR, f"{case['basename']}_comparison_x8.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    for case in CASES:
        render_comparison(case)
    print(f"\nAll visual comparison figures successfully saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
