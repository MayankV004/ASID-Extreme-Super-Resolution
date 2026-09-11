#!/usr/bin/env python3
"""
Quick setup of a mini-DIV2K dataset (15 images) for rapid local testing of the training pipeline.
"""
import os
import glob
from PIL import Image

OUT_ROOT = "datasets/mini_div2k"
HR_OUT = os.path.join(OUT_ROOT, "DIV2K_train_HR")
LR_OUT = os.path.join(OUT_ROOT, "DIV2K_train_LR_bicubic", "X8")

os.makedirs(HR_OUT, exist_ok=True)
os.makedirs(LR_OUT, exist_ok=True)

source_images = sorted(glob.glob("benchmark/HR/Urban100/x2/*.png"))[:15]
print(f"Creating mini-DIV2K from {len(source_images)} source images...")

for idx, img_path in enumerate(source_images, start=1):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        # Modcrop to multiple of 8
        w_crop = (w // 8) * 8
        h_crop = (h // 8) * 8
        hr_img = img.crop((0, 0, w_crop, h_crop))
        
        # 8x bicubic downsample
        lr_img = hr_img.resize((w_crop // 8, h_crop // 8), Image.Resampling.BICUBIC)
        
        # Save matching DIV2K format: 0001.png and 0001x8.png
        hr_path = os.path.join(HR_OUT, f"{idx:04d}.png")
        lr_path = os.path.join(LR_OUT, f"{idx:04d}x8.png")
        
        hr_img.save(hr_path, "PNG")
        lr_img.save(lr_path, "PNG")

print("Mini-DIV2K created successfully in datasets/mini_div2k/")
