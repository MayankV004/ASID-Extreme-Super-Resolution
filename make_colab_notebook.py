import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Course Project: Ultra-Lightweight Transformers for Extreme Super-Resolution (x8)\n",
                "**Paper:** Efficient Attention-Sharing Information Distillation Transformer for Lightweight SISR (ASID, AAAI 2025)\n",
                "**Research Focus:** Evaluating and Improving Robustness under Extreme Upscaling (x8) via Pre-Upsampling and Direct Distillation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Verify GPU Allocation\n",
                "Run this cell to confirm your Colab instance has an active GPU (T4 / V100 / A100)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Mount Google Drive (Optional) and Setup Codebase\n",
                "If you have your project on Google Drive, mount it below, or clone your repository directly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Option A: Clone from GitHub (replace with your repo URL if needed)\n",
                "# !git clone https://github.com/saturnian77/ASID.git\n",
                "# %cd ASID\n",
                "\n",
                "# Option B: Mount Google Drive\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "# %cd /content/drive/MyDrive/computer-vision-project"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Install Dependencies\n",
                "Install the required lightweight packages for the ASID architecture, logging, and evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q thop tensorboardX einops timm opencv-python pyyaml tqdm matplotlib"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Download DIV2K Training Dataset\n",
                "Colab's high-speed internet downloads the official DIV2K dataset in under a minute."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "%cd /content\n",
                "!mkdir -p /content/DIV2K\n",
                "%cd /content/DIV2K\n",
                "\n",
                "# Download High-Resolution DIV2K train images (800 images)\n",
                "if not os.path.exists('DIV2K_train_HR.zip'):\n",
                "    !wget -q http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip\n",
                "    !unzip -q DIV2K_train_HR.zip\n",
                "\n",
                "print(\"DIV2K Train HR downloaded and extracted successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Prepare x8 Low-Resolution Training Images\n",
                "Generate the corresponding x8 bicubic downscaled images for training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob\n",
                "from PIL import Image\n",
                "from tqdm import tqdm\n",
                "\n",
                "hr_dir = '/content/DIV2K/DIV2K_train_HR'\n",
                "lr_out_dir = '/content/DIV2K/DIV2K_train_LR_bicubic/X8'\n",
                "os.makedirs(lr_out_dir, exist_ok=True)\n",
                "\n",
                "hr_images = sorted(glob.glob(os.path.join(hr_dir, '*.png')))\n",
                "print(f\"Downsampling {len(hr_images)} training images to x8...\")\n",
                "\n",
                "for img_path in tqdm(hr_images):\n",
                "    base = os.path.basename(img_path).replace('.png', '')\n",
                "    lr_save = os.path.join(lr_out_dir, f\"{base}x8.png\")\n",
                "    if not os.path.exists(lr_save):\n",
                "        with Image.open(img_path) as img:\n",
                "            img = img.convert('RGB')\n",
                "            w, h = img.size\n",
                "            crop_w = (w // 8) * 8\n",
                "            crop_h = (h // 8) * 8\n",
                "            img_cropped = img.crop((0, 0, crop_w, crop_h))\n",
                "            lr_img = img_cropped.resize((crop_w // 8, crop_h // 8), Image.Resampling.BICUBIC)\n",
                "            lr_img.save(lr_save, 'PNG')\n",
                "\n",
                "print(\"x8 Low-Resolution images ready!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Update env.json for Colab\n",
                "Point the dataset path to `/content/DIV2K`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "env_path = 'env/env.json'\n",
                "with open(env_path, 'r') as f:\n",
                "    env_cfg = json.load(f)\n",
                "\n",
                "env_cfg['path']['dataset_paths']['DIV2K'] = '/content/DIV2K'\n",
                "with open(env_path, 'w') as f:\n",
                "    json.dump(env_cfg, f, indent=4)\n",
                "\n",
                "print(\"env.json successfully updated for Colab:\", env_cfg['path']['dataset_paths']['DIV2K'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Train the Extreme x8 Super-Resolution Models\n",
                "Choose either:\n",
                "* **Run A:** Pre-Upsampling ASID (Bilinear x2 + ASID x4) — converges fast (~50-100 epochs).\n",
                "* **Run B:** Direct x8 ASID from scratch."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run A: Train Pre-Upsampled x8 ASID\n",
                "!python train.py -v \"ASID_PreUpsample_X8_DIV2K\" -p train --train_yaml \"train_ASID_PreUpsample_X8_DIV2K.yaml\"\n",
                "\n",
                "# Or Run B: Train Direct x8 ASID\n",
                "# !python train.py -v \"ASID_Direct_X8_DIV2K\" -p train --train_yaml \"train_ASID_X8_DIV2K.yaml\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Evaluate & Compare Final Results\n",
                "Generate the benchmark evaluation table across Set5, Set14, and Urban100."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python evaluate_preupsample_x8.py --datasets Set5 Set14 Urban100"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("ASID_Extreme_Super_Resolution.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("ASID_Extreme_Super_Resolution.ipynb generated successfully!")
