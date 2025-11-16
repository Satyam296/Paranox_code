#!/bin/bash
# Space Station Challenge - Environment Setup Script for Mac/Linux
# This script creates a conda environment named "EDU" with all required dependencies

echo "========================================"
echo "Space Station Challenge - Environment Setup"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "Creating conda environment 'EDU'..."
conda create -n EDU python=3.10 -y

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment"
    exit 1
fi

echo ""
echo "Activating EDU environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate EDU

echo ""
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "Installing Ultralytics YOLOv8..."
pip install ultralytics

echo ""
echo "Installing additional dependencies..."
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install pillow
pip install pyyaml
pip install tqdm
pip install scikit-learn
pip install tensorboard

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, use:"
echo "  conda activate EDU"
echo ""
echo "To start training, navigate to the scripts folder and run:"
echo "  python train.py"
echo ""
