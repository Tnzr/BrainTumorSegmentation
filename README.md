# BrainTumorSegmentation
Brain Tumor Segmentation System
```markdown
# 3D Brain Tumor Segmentation with Distributed Training

## Overview
This repository implements a distributed training pipeline for 3D brain tumor segmentation using MONAI and PyTorch. The system features:
- **Flexible 3D U-Net** architecture with gradient checkpointing
- **5-fold cross-validation** with distributed data parallelism
- **Multi-metric evaluation** (Dice Score, HD95)
- **Advanced visualization** of training dynamics and predictions
- **Memory-optimized** preprocessing and validation

## Features
- 🧠 BraTS dataset compatibility
- ⚡ Mixed-precision training with AMP
- 🔄 Validation every 4 epochs
- 📊 Automatic metric visualization
- 🎞️ Prediction timeline GIF generation
- 🧩 Modular components for easy customization

## Installation
```bash
conda create -n monai_seg python=3.9
conda activate monai_seg
pip install monai nibabel tensorboard torchvision imageio tqdm
```

## Dataset Preparation
1. Download BraTS dataset (Task01_BrainTumour)
2. Organize with structure:
```
Task01_BrainTumour/
├── imagesTr/
├── labelsTr/
└── dataset.json
```

## Usage

### Training
```bash
python BrainSegDistTrain.py --epochs 16 --data_path /path/to/dataset
```

Key arguments:
- `--epochs`: Total training epochs (default: 16)
- `--data_path`: Path to BraTS dataset
- `--max_samples`: Limit training samples (for debugging)

### Visualization
After training:
```bash
python visualize.py --log_dir runs --output_dir visualizations
```

Generates:
- Individual fold metrics plots
- Aggregated performance charts
- Prediction timeline GIFs
- Class probability visualizations

## Methodology

### Architecture
**Flexible 3D U-Net** with:
- 5 encoding/decoding levels
- Residual blocks (2 units per level)
- Batch normalization
- Gradient checkpointing

### Preprocessing
1. Intensity scaling (0-1 range)
2. Spatial padding/cropping (240×240×160)
3. On-the-fly augmentations:
   - Random flipping
   - 90° rotations
   - Pos/Neg crop sampling

### Training Strategy
- **Loss Function**: Dice + Cross-Entropy
- **Optimizer**: Adam (lr=1e-4)
- **Validation**: Every 4 epochs
- **Metrics**:
  - Dice Score (ET, WT, TC)
  - Hausdorff Distance 95%
  
### Distributed Training
- NCCL backend for multi-GPU coordination
- Distributed data sampling
- Cross-validation aware logging
- Memory-efficient validation with sliding window

## Results Visualization
Output directory structure:
```
visualizations/
├── metrics/
│   ├── fold_*_metrics.png
│   ├── aggregated_*.png
│   └── combined_metrics.png
└── timelines/
    └── fold_*/sample_*/
        ├── epoch_*.png
        └── timeline.gif
```

## References
1. BraTS Challenge: https://www.med.upenn.edu/sbia/brats2017.html
2. MONAI Framework: https://monai.io/
3. nnU-Net Architecture: https://arxiv.org/abs/1809.10486
https://buymeacoffee.com/bryangarcip

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/bryangarcip)
```

This README provides:
1. Clear installation instructions
2. Dataset preparation guide
3. Training/visualization commands
4. Technical methodology breakdown
5. Result interpretation guidance
6. Key references

The structure emphasizes practical usage while maintaining scientific rigor, suitable for both researchers and developers.
