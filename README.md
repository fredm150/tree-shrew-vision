# tree-shrew-vision
University of Virginia | ENGR 1020: Engineering Foundations 2 | Spring 2026

## Project Overview
Existing 3D video is calibrated to human interpupillary distance (~60mm). 
Treeshrews have an IPD of ~12mm, causing hyper-stereopsis when viewed with 
human-calibrated 3D. This project aims to help solve the gap between generating 3D video stimuli for tree shrew vision analysis. We utilize Deep3D for 3D video generation and reconstruction and PyTorch for modeling a custom made convolutional neural network (CNN).

## Tech Stack
* **Language:** Python 3.7+
* **AI/ML:** PyTorch, Deep3D
* **Hardware:**
  * Raspberry Pi 4
  * Arducam 1/4" Multi Camera Adapter Module V2.2
  * Arducam 5MP Sensor Mini Camera Modules
  * 3.5" 480x320 TFT LCD SPI Display Panel

## Setup
```bash
pip install torch torchvision opencv-python pytorch-msssim lpips wandb librosa
```

## Usage

**Extract frames from stereo video:**
```bash
python sync_and_extract.py
```

**Train:**
```bash
python train.py --model deep3d_v1.0_640x360_cpu.pt --data ./data --epochs 25
```

**Evaluate:**
```bash
# Baseline
python evaluate.py --base_model deep3d_v1.0_640x360_cpu.pt --data ./data --baseline

# Fine-tuned
python evaluate.py --model best_model_640x360.pt --base_model deep3d_v1.0_640x360_cpu.pt --data ./data
```

**Visualize:**
```bash
python visualize.py --data ./data --n_samples 5
```

## Pre-trained Models
Download from [HypoX64/Deep3D releases](https://github.com/HypoX64/Deep3D)

## Acknowledgements
Built on [HypoX64/Deep3D](https://github.com/HypoX64/Deep3D), 
a PyTorch reimplementation of [Xie et al. 2016](https://arxiv.org/abs/1604.03650).
