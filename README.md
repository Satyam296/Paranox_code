# 🚀 Space Station Challenge: Safety Object Detection #2

**Duality AI Space Station Challenge - Complete Solution Package**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Bonus Application](#bonus-application)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository contains a complete solution for the **Duality AI Space Station Challenge #2**, which focuses on training a YOLOv8 object detection model to identify 7 critical safety equipment items in a space station environment using synthetic data from Falcon digital twin platform.

### Target Objects

1. 🔵 **Oxygen Tank**
2. 🟢 **Nitrogen Tank**
3. 🔴 **First Aid Box**
4. 🟡 **Safety Switch Panel**
5. 🔴 **Fire Extinguisher**
6. 🚨 **Fire Alarm**
7. 📞 **Emergency Phone**

### Key Objectives

✅ Train robust object detection model on synthetic data  
✅ Achieve high mAP@0.5 scores (target: 40-50% baseline, 70%+ excellent)  
✅ Evaluate across varied lighting and occlusion conditions  
✅ Document complete methodology and results  
✅ (Bonus) Create application demonstrating real-world use  

---

## ⚡ Quick Start

**Want to get started immediately? Follow these steps:**

### 1. Setup Environment
```bash
# Windows
cd ENV_SETUP
setup_env.bat

# Mac/Linux
cd ENV_SETUP
chmod +x setup_env.sh
./setup_env.sh
```

### 2. Prepare Dataset
```bash
# Extract dataset to data/ folder
# Edit configs/config.yaml to point to your dataset path
```

### 3. Train Model
```bash
conda activate EDU
cd scripts
python train.py --config ../configs/config.yaml
```

### 4. Evaluate
```bash
python predict.py --weights runs/train/exp/weights/best.pt --config ../configs/config.yaml
```

### 5. Run Web App (Bonus)
```bash
pip install -r requirements_app.txt
streamlit run app.py
```

**📖 For detailed instructions, see [RUN_APP.md](RUN_APP.md)**

---

## 📁 Project Structure

```
Space_Station_Challenge/
│
├── 📂 ENV_SETUP/                    # Environment setup
│   ├── setup_env.bat               # Windows setup script
│   ├── setup_env.sh                # Mac/Linux setup script
│   └── README.txt                  # Setup instructions
│
├── 📂 scripts/                      # Main scripts
│   ├── train.py                    # Model training
│   └── predict.py                  # Evaluation & prediction
│
├── 📂 configs/                      # Configuration files
│   └── config.yaml                 # Main config (dataset, hyperparameters)
│
├── 📂 utils/                        # Utility functions
│   ├── dataset_utils.py            # Dataset analysis tools
│   └── visualization_utils.py      # Plotting & visualization
│
├── 📂 data/                         # Dataset (not included)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── 📂 runs/                         # Training outputs
│   └── train/
│       └── exp/
│           ├── weights/            # Model checkpoints
│           │   ├── best.pt
│           │   └── last.pt
│           └── predictions/        # Evaluation results
│               ├── evaluation_report.txt
│               ├── confusion_matrix.png
│               └── visualizations/
│
├── 📂 documentation/                # Documentation
│   ├── README.md                   # Detailed project docs
│   └── HACKATHON_REPORT_TEMPLATE.md # Report template
│
├── 📄 app.py                       # Streamlit web application
├── 📄 requirements_app.txt         # App dependencies
├── 📄 RUN_APP.md                   # App instructions
└── 📄 README.md                    # This file
```

---

## ✨ Features

### Core Features

- ✅ **Complete Training Pipeline**
  - YOLOv8 integration with Ultralytics
  - Configurable hyperparameters
  - Automatic checkpointing and logging
  - Early stopping support
  - GPU/CPU automatic selection

- ✅ **Comprehensive Evaluation**
  - mAP@0.5 and mAP@0.5:0.95 calculation
  - Per-class performance metrics
  - Confusion matrix visualization
  - Failure case analysis
  - Prediction visualizations

- ✅ **Dataset Utilities**
  - Dataset integrity checking
  - Class distribution analysis
  - Sample visualization
  - Statistics reporting

- ✅ **Visualization Tools**
  - Training history plots
  - Performance dashboards
  - Radar charts for per-class metrics
  - Confusion matrices

### Bonus Features

- 🎁 **Real-time Safety Monitor Application**
  - Live object detection via webcam/video
  - Safety compliance checking
  - Alert system for missing equipment
  - Detection logging and reporting
  - Screenshot and video export capabilities

- 🎁 **Falcon Integration Strategy**
  - Detailed model update workflow
  - Continuous learning pipeline
  - Edge case handling methodology
  - Automated data generation plan

---

## 🔧 Installation

### Prerequisites

- **Operating System:** Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python:** 3.10 or higher
- **Anaconda/Miniconda:** Installed and configured
- **GPU (Recommended):** NVIDIA GPU with CUDA 11.8+ support
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10GB free space

### Step-by-Step Installation

1. **Clone or Download this Repository**
   ```bash
   git clone <your-repo-url>
   cd Space_Station_Challenge
   ```

2. **Download Dataset**
   - Visit Duality AI Falcon platform
   - Download Space Station Challenge dataset
   - Extract to `data/` folder

3. **Run Environment Setup**
   
   **Windows:**
   ```bash
   cd ENV_SETUP
   setup_env.bat
   ```
   
   **Mac/Linux:**
   ```bash
   cd ENV_SETUP
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

4. **Verify Installation**
   ```bash
   conda activate EDU
   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
   ```

### Dependencies

The setup script installs:
- PyTorch 2.0+ with CUDA support
- Ultralytics YOLOv8
- OpenCV
- Matplotlib, Seaborn
- NumPy, Pandas
- scikit-learn
- TensorBoard

---

## 🚀 Usage

### 1. Configure Dataset Path

Edit `configs/config.yaml`:
```yaml
path: ../data  # Point to your dataset location
```

### 2. Train the Model

**Basic Training:**
```bash
conda activate EDU
cd scripts
python train.py --config ../configs/config.yaml
```

**Custom Training:**
```bash
# Train with custom epochs and batch size
python train.py --config ../configs/config.yaml --epochs 150 --batch 32

# Use specific GPU
python train.py --config ../configs/config.yaml --device 0

# Use CPU
python train.py --config ../configs/config.yaml --device cpu

# Custom experiment name
python train.py --config ../configs/config.yaml --name my_experiment
```

**Training Parameters:**
- `--config`: Path to configuration file
- `--epochs`: Number of training epochs (default: from config)
- `--batch`: Batch size (default: from config)
- `--imgsz`: Image size for training (default: 640)
- `--device`: Device to use (0, 1, 2... for GPU, 'cpu' for CPU)
- `--name`: Experiment name (default: exp)

### 3. Evaluate the Model

```bash
python predict.py --weights runs/train/exp/weights/best.pt --config ../configs/config.yaml
```

**Evaluation Options:**
```bash
# Evaluate with custom test set
python predict.py --weights best.pt --config ../configs/config.yaml --source /path/to/test/images

# Skip visualization generation (faster)
python predict.py --weights best.pt --config ../configs/config.yaml --no-viz
```

### 4. Analyze Results

Check the generated files in `runs/train/exp/predictions/`:
- `evaluation_report.txt` - Detailed metrics
- `confusion_matrix.png` - Visual confusion matrix
- `metrics.json` - Machine-readable metrics
- `failure_analysis.txt` - Improvement recommendations
- `visualizations/` - Sample predictions

### 5. Run Bonus Application

```bash
cd bonus_application

# Use webcam
python safety_monitor_app.py --weights ../runs/train/exp/weights/best.pt --source 0

# Use video file
python safety_monitor_app.py --weights ../runs/train/exp/weights/best.pt --source test_video.mp4

# Save output video
python safety_monitor_app.py --weights ../runs/train/exp/weights/best.pt --source 0 --save
```

**Application Controls:**
- `q`: Quit
- `s`: Save screenshot
- `l`: Save detection log

---

## 📊 Results

### Expected Performance

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| mAP@0.5 | 40-50% | 50-70% | 70%+ |
| Precision | - | 70%+ | 85%+ |
| Recall | - | 70%+ | 85%+ |

### Benchmarking

After training and evaluation, compare your results:
- **Baseline Model (YOLOv8n, 100 epochs):** ~45% mAP@0.5
- **Optimized Model (YOLOv8m, 150 epochs):** ~65% mAP@0.5
- **Best Model (YOLOv8l, 200 epochs + tuning):** ~75% mAP@0.5

### Improving Performance

If your mAP@0.5 is below target:

1. **Train Longer**
   ```bash
   python train.py --config ../configs/config.yaml --epochs 200
   ```

2. **Use Larger Model**
   ```yaml
   # Edit config.yaml
   model: yolov8m.pt  # or yolov8l.pt, yolov8x.pt
   ```

3. **Adjust Learning Rate**
   ```yaml
   # Edit config.yaml
   lr0: 0.005  # Try lower learning rate
   ```

4. **Enhance Augmentation**
   ```yaml
   # Edit config.yaml
   mosaic: 1.0
   mixup: 0.2  # Add MixUp augmentation
   ```

---

## 🎁 Bonus Application

The **Space Station Safety Monitor** is a real-time application that demonstrates practical use of the trained model.

### Features

- 🎥 **Real-time Detection:** Process video from webcam or file
- ⚠️ **Safety Alerts:** Warn when critical equipment is missing
- 📝 **Logging:** JSON logs of all detections
- 📊 **Performance Metrics:** Live FPS and statistics
- 📸 **Screenshot Capture:** Save evidence of violations
- 💾 **Video Export:** Record monitoring sessions

### Use Cases

1. **Autonomous Monitoring:** 24/7 safety equipment verification
2. **Training Simulations:** Practice equipment identification
3. **Compliance Auditing:** Automated safety inspections
4. **Drone Integration:** Mount on inspection drones

### Falcon Integration

The application includes a comprehensive strategy for keeping the model up-to-date using Falcon:

1. **Monitor:** Detect low-confidence predictions
2. **Analyze:** Identify failure patterns
3. **Generate:** Create new training data in Falcon
4. **Retrain:** Update model with new data
5. **Deploy:** Roll out improved version

See `bonus_application/APPLICATION_PROPOSAL.md` for details.

---

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
python train.py --config ../configs/config.yaml --batch 8

# Use smaller model
# Edit config.yaml: model: yolov8n.pt

# Reduce image size
# Edit config.yaml: imgsz: 416
```

#### 2. Dataset Path Not Found

**Error:** `Dataset path does not exist`

**Solutions:**
```yaml
# Use absolute path in config.yaml
path: E:/Planet_ML/Space_Station_Challenge/data

# Verify folder structure
data/
├── train/images/
├── train/labels/
├── val/images/
└── val/labels/
```

#### 3. Low mAP Score

**Problem:** mAP@0.5 < 40%

**Solutions:**
- Train longer (150-200 epochs)
- Use larger model (yolov8m or yolov8l)
- Check dataset quality
- Adjust augmentation parameters
- Verify labels are correct

#### 4. Slow Training

**Problem:** Training is very slow

**Solutions:**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, ensure CUDA is installed properly
# Or use CPU (slower but works)
python train.py --config ../configs/config.yaml --device cpu
```

#### 5. Module Not Found

**Error:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
# Ensure EDU environment is activated
conda activate EDU

# Reinstall dependencies
pip install ultralytics opencv-python matplotlib seaborn
```

### Getting Help

- 📚 **Documentation:** Check `documentation/README.md`
- 💬 **Discord:** Join Duality AI Discord server
- 📖 **YOLOv8 Docs:** https://docs.ultralytics.com
- 🔍 **Falcon Exercises:** Review Exercise 3 on Falcon platform

---

## 📤 Submission Guidelines

### What to Submit

Create a zip file containing:

1. **✅ Code & Models**
   - All scripts (`train.py`, `predict.py`, etc.)
   - Configuration file (`config.yaml`)
   - Trained model weights (`best.pt`)
   - `runs/train/exp/` folder with results

2. **✅ Documentation**
   - Completed hackathon report (max 8 pages)
   - README with instructions
   - Methodology explanation

3. **✅ Results**
   - Evaluation report with mAP@0.5 score
   - Confusion matrix image
   - Sample predictions
   - Training curves

4. **✅ (Bonus) Application**
   - Application code
   - Demo video
   - Proposal document
   - Instructions to run

### Submission Process

1. **Create Private GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Space Station Challenge submission"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Add Collaborators**
   - Syed Muhammad Maaz: `Maazsyedm`
   - Rebekah Bogdanoff: `rebekah-bogdanoff`

3. **Submit Form**
   - Report final mAP@0.5 score
   - Provide GitHub repository link

---

## 🏆 Judging Criteria

| Criteria | Points | Description |
|----------|--------|-------------|
| **Model Performance** | 80 | mAP@0.5 score |
| **Documentation** | 20 | Report clarity and completeness |
| **Bonus Application** | +15 | Use case proposal (max 100 total) |

---

## 👥 Team

**Team Name:** [Your Team Name]

**Members:**
- [Name] - [Role]
- [Name] - [Role]
- [Name] - [Role]

---

## 📝 License

This project is created for the Duality AI Space Station Challenge and follows educational use guidelines.

---

## 🙏 Acknowledgments

- **Duality AI** for the Falcon platform and challenge
- **Ultralytics** for YOLOv8 implementation
- **PyTorch** team for the deep learning framework
- All challenge participants and community members

---

## 📧 Contact

For questions or support:
- **Email:** [your-email]
- **GitHub:** [your-username]
- **LinkedIn:** [your-profile]

---

## 🚀 Next Steps

After submission:
1. ✅ Share your work on LinkedIn (tag DualityAI)
2. ✅ Connect with participants on Discord
3. ✅ Prepare for judge discussion (selected teams)
4. ✅ Explore internship/apprenticeship opportunities

---

**Good luck with your submission! 🎉**

---

<div align="center">

**Made with ❤️ for Duality AI Space Station Challenge #2**

[Documentation](documentation/README.md) • [Quick Start](QUICK_START.md) • [Bonus App](bonus_application/APPLICATION_PROPOSAL.md)

</div>
