"""
Space Station Challenge - YOLO Training Script
===============================================
This script trains a YOLOv8 model for detecting space station safety objects.

Usage:
    python train.py --config ../configs/config.yaml
    python train.py --config ../configs/config.yaml --epochs 150 --batch 32
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(project_dir, run_name):
    """Create necessary directories for training outputs."""
    run_dir = Path(project_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def validate_dataset(config):
    """Validate that dataset paths exist."""
    data_path = Path(config['path'])
    train_path = data_path / config['train']
    val_path = data_path / config['val']
    
    if not data_path.exists():
        print(f"WARNING: Dataset path does not exist: {data_path}")
        print("Please update the 'path' in config.yaml to point to your dataset")
    
    if not train_path.exists():
        print(f"WARNING: Training path does not exist: {train_path}")
    
    if not val_path.exists():
        print(f"WARNING: Validation path does not exist: {val_path}")
    
    print(f"\nDataset Configuration:")
    print(f"  Root path: {data_path.absolute()}")
    print(f"  Train: {train_path.absolute()}")
    print(f"  Val: {val_path.absolute()}")
    print(f"  Classes: {config['nc']} - {list(config['names'].values())}\n")

def print_training_info(config, args):
    """Print training configuration information."""
    print("="*70)
    print("SPACE STATION CHALLENGE - YOLO TRAINING")
    print("="*70)
    print(f"\nTraining Configuration:")
    print(f"  Model: {config.get('model', 'yolov8n.pt')}")
    print(f"  Epochs: {args.epochs if args.epochs else config.get('epochs', 100)}")
    print(f"  Batch size: {args.batch if args.batch else config.get('batch', 16)}")
    print(f"  Image size: {args.imgsz if args.imgsz else config.get('imgsz', 640)}")
    print(f"  Device: {args.device if args.device else config.get('device', 'auto')}")
    print(f"  Workers: {config.get('workers', 8)}")
    print(f"  Learning rate: {config.get('lr0', 0.01)}")
    print(f"  Optimizer: {config.get('optimizer', 'auto')}")
    print(f"\nAugmentation Settings:")
    print(f"  Mosaic: {config.get('mosaic', 1.0)}")
    print(f"  MixUp: {config.get('mixup', 0.0)}")
    print(f"  Flip LR: {config.get('fliplr', 0.5)}")
    print(f"  HSV augmentation: H={config.get('hsv_h', 0.015)}, S={config.get('hsv_s', 0.7)}, V={config.get('hsv_v', 0.4)}")
    print("="*70)
    print()

def train_model(config_path, args):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.device:
        config['device'] = args.device
    if args.name:
        config['name'] = args.name
    
    # Print training information
    print_training_info(config, args)
    
    # Validate dataset
    validate_dataset(config)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("WARNING: No GPU detected. Training will use CPU (this will be slower)\n")
    
    # Initialize YOLO model
    model_name = config.get('model', 'yolov8n.pt')
    print(f"Initializing {model_name} model...")
    model = YOLO(model_name)
    
    # Prepare training arguments
    train_args = {
        'data': config_path,
        'epochs': config.get('epochs', 100),
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'patience': config.get('patience', 50),
        'workers': config.get('workers', 8),
        'device': config.get('device', ''),
        'save': config.get('save', True),
        'save_period': config.get('save_period', -1),
        'project': config.get('project', 'runs/train'),
        'name': config.get('name', 'exp'),
        'exist_ok': config.get('exist_ok', False),
        'pretrained': config.get('pretrained', True),
        'verbose': config.get('verbose', True),
        'optimizer': config.get('optimizer', 'auto'),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
        'degrees': config.get('degrees', 0.0),
        'translate': config.get('translate', 0.1),
        'scale': config.get('scale', 0.5),
        'shear': config.get('shear', 0.0),
        'perspective': config.get('perspective', 0.0),
        'flipud': config.get('flipud', 0.0),
        'fliplr': config.get('fliplr', 0.5),
        'mosaic': config.get('mosaic', 1.0),
        'mixup': config.get('mixup', 0.0),
        'deterministic': config.get('deterministic', True),
        'single_cls': config.get('single_cls', False),
        'rect': config.get('rect', False),
        'cos_lr': config.get('cos_lr', False),
        'close_mosaic': config.get('close_mosaic', 10),
        'amp': config.get('amp', True),
    }
    
    # Start training
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training will be saved to: {train_args['project']}/{train_args['name']}\n")
    
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nResults saved to: {results.save_dir}")
        print("\nNext steps:")
        print("1. Review training metrics in the results plots")
        print("2. Run predict.py to evaluate your model on test data")
        print("3. Check the confusion matrix and performance metrics")
        print("="*70 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that your dataset paths are correct in config.yaml")
        print("2. Ensure you have enough GPU/CPU memory (try reducing batch size)")
        print("3. Verify that images and labels are properly formatted")
        print("4. Check that the EDU environment is activated")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train YOLO model for Space Station Challenge')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Image size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., 0 for GPU, cpu for CPU)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (overrides config)')
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        print("Please ensure config.yaml exists in the configs folder")
        sys.exit(1)
    
    # Train the model
    train_model(args.config, args)

if __name__ == '__main__':
    main()
