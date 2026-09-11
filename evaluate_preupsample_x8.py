#!/usr/bin/env python3
"""
Evaluation script for Extreme x8 Super-Resolution:
Evaluates:
1. Bicubic x8 Baseline
2. Bilinear x2 + ASID x4 (Proposed in project proposal)
3. Bicubic x2 + ASID x4
across standard benchmarks (Set5, Set14, Urban100).
"""

import os
import glob
import time
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import cv2

from components.ASID import ASID
from utilities.utilities import calculate_psnr, calculate_ssim, tensor2img
from utilities.json_config import readConfig

def load_asid_x4_model(device="cuda"):
    model_config_path = "train_logs/ASID_X4_DIV2K/model_config.json"
    cfg = readConfig(model_config_path)
    
    model = ASID(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=cfg["feature_num"],
        **cfg["module_params"]
    )
    
    ckpt_path = "train_logs/ASID_X4_DIV2K/checkpoints/epoch0_ASID.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print("Loaded ASID x4 model successfully.")
    return model, cfg

def evaluate_dataset(model, dataset_name, device="cuda", save_dir=None):
    hr_dir = os.path.join("benchmark", "HR", dataset_name, "x8")
    lr_dir = os.path.join("benchmark", "LR", "LRBI", dataset_name, "x8")
    
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    if not hr_files:
        print(f"No files found in {hr_dir}")
        return None
    
    results = {
        "bicubic_x8": {"psnr": [], "ssim": []},
        "bilinear_x2_asid_x4": {"psnr": [], "ssim": []},
        "bicubic_x2_asid_x4": {"psnr": [], "ssim": []}
    }
    
    img_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    if save_dir:
        os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
    
    with torch.no_grad():
        for hr_path in tqdm(hr_files, desc=dataset_name):
            base_name = os.path.basename(hr_path).replace("_HR_x8.png", "")
            lr_path = os.path.join(lr_dir, f"{base_name}_LRBI_x8.png")
            
            hr_pil = Image.open(hr_path).convert("RGB")
            lr_pil = Image.open(lr_path).convert("RGB")
            
            hr_np = np.array(hr_pil)
            w_hr, h_hr = hr_pil.size
            
            # 1. Standard Bicubic x8 baseline
            bicubic_x8_pil = lr_pil.resize((w_hr, h_hr), Image.Resampling.BICUBIC)
            bic_np = np.array(bicubic_x8_pil)
            results["bicubic_x8"]["psnr"].append(calculate_psnr(bic_np, hr_np))
            results["bicubic_x8"]["ssim"].append(calculate_ssim(bic_np, hr_np))
            
            # Prepare tensor for ASID
            lr_tensor = img_transform(lr_pil).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 2. Bilinear x2 pre-upsampling + ASID x4
            lr_bilinear_x2 = F.interpolate(lr_tensor, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Run patch-based inference or full inference
            # For 64x64 or smaller x2 is 128x128, which easily fits in GPU memory
            out_bilinear = model(lr_bilinear_x2)
            out_bilinear_np = tensor2img(out_bilinear.cpu())[0]  # [H, W, 3] in [0, 255]
            
            # Crop to match exact HR size if padded
            out_bilinear_np = out_bilinear_np[:h_hr, :w_hr, :]
            results["bilinear_x2_asid_x4"]["psnr"].append(calculate_psnr(out_bilinear_np, hr_np))
            results["bilinear_x2_asid_x4"]["ssim"].append(calculate_ssim(out_bilinear_np, hr_np))
            
            # 3. Bicubic x2 pre-upsampling + ASID x4
            lr_bicubic_x2 = F.interpolate(lr_tensor, scale_factor=2, mode='bicubic', align_corners=False)
            out_bicubic = model(lr_bicubic_x2)
            out_bicubic_np = tensor2img(out_bicubic.cpu())[0][:h_hr, :w_hr, :]
            results["bicubic_x2_asid_x4"]["psnr"].append(calculate_psnr(out_bicubic_np, hr_np))
            results["bicubic_x2_asid_x4"]["ssim"].append(calculate_ssim(out_bicubic_np, hr_np))
            
            # Save sample outputs
            if save_dir:
                out_bgr = cv2.cvtColor(out_bilinear_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, dataset_name, f"{base_name}_bilinear_asidx4.png"), out_bgr)
    
    summary = {}
    for method, metrics in results.items():
        summary[method] = {
            "psnr": float(np.mean(metrics["psnr"])),
            "ssim": float(np.mean(metrics["ssim"]))
        }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Set5", "Set14"], help="Datasets to evaluate")
    parser.add_argument("--save_dir", type=str, default="SR_Results/x8_experiments")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, _ = load_asid_x4_model(device=device)
    
    all_results = {}
    for ds in args.datasets:
        print(f"\n--- Evaluating {ds} at x8 ---")
        res = evaluate_dataset(model, ds, device=device, save_dir=args.save_dir)
        if res:
            all_results[ds] = res
            print(f"Results for {ds}:")
            for method, metrics in res.items():
                print(f"  {method:22s}: PSNR = {metrics['psnr']:.2f} dB, SSIM = {metrics['ssim']:.4f}")
    
    # Print formatted markdown table
    print("\n" + "="*50)
    print("### Summary Results (PSNR [dB] / SSIM)")
    print("="*50)
    header = "| Dataset | Bicubic x8 | Bilinear x2 + ASID x4 | Bicubic x2 + ASID x4 |"
    sep = "| :--- | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for ds, m in all_results.items():
        row = f"| **{ds}** | {m['bicubic_x8']['psnr']:.2f} / {m['bicubic_x8']['ssim']:.4f} | {m['bilinear_x2_asid_x4']['psnr']:.2f} / {m['bilinear_x2_asid_x4']['ssim']:.4f} | {m['bicubic_x2_asid_x4']['psnr']:.2f} / {m['bicubic_x2_asid_x4']['ssim']:.4f} |"
        print(row)

if __name__ == "__main__":
    main()
