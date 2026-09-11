#!/usr/bin/env python3
"""
Generate x8 benchmark test images for Set5, Set14, Urban100, and B100.
Modcrops HR images to multiples of 8 and creates corresponding LR (bicubic downsampled) images.
"""

import os
import glob
from PIL import Image
from tqdm import tqdm

DATASETS = ["Set5", "Set14", "Urban100", "B100"]
BASE_DIR = "benchmark"
SCALE = 8

def generate_x8_dataset(dataset_name):
    # Read from x2 directory as the base HR images
    hr_source_dir = os.path.join(BASE_DIR, "HR", dataset_name, "x2")
    hr_out_dir = os.path.join(BASE_DIR, "HR", dataset_name, f"x{SCALE}")
    lr_out_dir = os.path.join(BASE_DIR, "LR", "LRBI", dataset_name, f"x{SCALE}")
    
    os.makedirs(hr_out_dir, exist_ok=True)
    os.makedirs(lr_out_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(hr_source_dir, "*.png")))
    if not image_paths:
        print(f"Warning: No images found in {hr_source_dir}")
        return
    
    print(f"Processing {dataset_name} ({len(image_paths)} images)...")
    for img_path in tqdm(image_paths, desc=dataset_name):
        filename = os.path.basename(img_path)
        # Handle filenames like "baby_HR_x2.png" -> "baby"
        base_name = filename.replace("_HR_x2.png", "").replace(".png", "")
        
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            
            # Modcrop so dimensions are divisible by SCALE (8)
            crop_w = (w // SCALE) * SCALE
            crop_h = (h // SCALE) * SCALE
            hr_cropped = img.crop((0, 0, crop_w, crop_h))
            
            # Bicubic downsampling by SCALE (8)
            lr_w = crop_w // SCALE
            lr_h = crop_h // SCALE
            lr_image = hr_cropped.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
            
            # Save HR and LR pairs
            hr_save_path = os.path.join(hr_out_dir, f"{base_name}_HR_x{SCALE}.png")
            lr_save_path = os.path.join(lr_out_dir, f"{base_name}_LRBI_x{SCALE}.png")
            
            hr_cropped.save(hr_save_path, "PNG")
            lr_image.save(lr_save_path, "PNG")

def main():
    for dataset in DATASETS:
        generate_x8_dataset(dataset)
    print("\nSuccessfully generated all x8 benchmark datasets!")

if __name__ == "__main__":
    main()
