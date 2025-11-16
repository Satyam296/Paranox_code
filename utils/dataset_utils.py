"""
Utility Functions for Space Station Challenge
==============================================
Helper functions for data analysis, visualization, and performance metrics.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

def count_dataset_stats(data_path):
    """
    Count images and labels in dataset.
    
    Args:
        data_path: Path to dataset root directory
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'train': {'images': 0, 'labels': 0},
        'val': {'images': 0, 'labels': 0},
        'test': {'images': 0, 'labels': 0}
    }
    
    data_path = Path(data_path)
    
    for split in ['train', 'val', 'test']:
        img_path = data_path / split / 'images'
        lbl_path = data_path / split / 'labels'
        
        if img_path.exists():
            stats[split]['images'] = len(list(img_path.glob('*.jpg'))) + len(list(img_path.glob('*.png')))
        
        if lbl_path.exists():
            stats[split]['labels'] = len(list(lbl_path.glob('*.txt')))
    
    return stats

def visualize_dataset_distribution(config_path, output_path=None):
    """
    Visualize class distribution in dataset.
    
    Args:
        config_path: Path to config.yaml
        output_path: Where to save the plot
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = list(config['names'].values())
    data_path = Path(config['path'])
    
    # Count instances per class
    class_counts = {name: 0 for name in class_names}
    
    for split in ['train', 'val']:
        label_path = data_path / split / 'labels'
        if label_path.exists():
            for label_file in label_path.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='navy')
    plt.xlabel('Object Class', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return class_counts

def check_dataset_integrity(config_path):
    """
    Check for missing images/labels and report issues.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary with integrity check results
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = Path(config['path'])
    issues = []
    
    for split in ['train', 'val', 'test']:
        img_path = data_path / split / 'images'
        lbl_path = data_path / split / 'labels'
        
        if not img_path.exists():
            issues.append(f"Missing {split}/images directory")
            continue
        
        if not lbl_path.exists():
            issues.append(f"Missing {split}/labels directory")
            continue
        
        # Check for orphaned files
        images = {f.stem for f in img_path.glob('*') if f.suffix in ['.jpg', '.png']}
        labels = {f.stem for f in lbl_path.glob('*.txt')}
        
        orphaned_images = images - labels
        orphaned_labels = labels - images
        
        if orphaned_images:
            issues.append(f"{split}: {len(orphaned_images)} images without labels")
        
        if orphaned_labels:
            issues.append(f"{split}: {len(orphaned_labels)} labels without images")
    
    return {
        'has_issues': len(issues) > 0,
        'issues': issues
    }

def visualize_sample_images(data_path, class_names, num_samples=5, output_dir=None):
    """
    Visualize sample images with bounding boxes.
    
    Args:
        data_path: Path to dataset root
        class_names: List of class names
        num_samples: Number of samples to visualize
        output_dir: Where to save visualizations
    """
    data_path = Path(data_path)
    img_path = data_path / 'train' / 'images'
    lbl_path = data_path / 'train' / 'labels'
    
    if not img_path.exists():
        print("Train images directory not found")
        return
    
    image_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))
    
    if len(image_files) == 0:
        print("No images found")
        return
    
    # Random sample
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in samples:
        # Load image
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load labels
        label_file = lbl_path / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Draw box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display or save
        if output_dir:
            output_path = Path(output_dir) / f"sample_{img_file.stem}.jpg"
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_bgr)
        else:
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Sample: {img_file.name}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.close()

def compare_models(results_dirs, output_path=None):
    """
    Compare metrics from multiple training runs.
    
    Args:
        results_dirs: List of paths to training result directories
        output_path: Where to save comparison plot
    """
    metrics = []
    names = []
    
    for result_dir in results_dirs:
        result_path = Path(result_dir)
        names.append(result_path.name)
        
        # Try to load metrics from results
        # This is a simplified version - adjust based on actual YOLO output structure
        metric_dict = {
            'mAP50': 0,
            'Precision': 0,
            'Recall': 0
        }
        metrics.append(metric_dict)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ['mAP50', 'Precision', 'Recall']
    
    for i, metric_name in enumerate(metric_names):
        values = [m[metric_name] for m in metrics]
        axes[i].bar(names, values, color='skyblue', edgecolor='navy')
        axes[i].set_title(metric_name, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].set_ylim([0, 1])
        axes[i].grid(axis='y', alpha=0.3)
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_dataset_info(config_path):
    """Print comprehensive dataset information."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("DATASET INFORMATION")
    print("="*70)
    print(f"\nDataset Path: {config['path']}")
    print(f"Number of Classes: {config['nc']}")
    print(f"\nClasses:")
    for idx, name in config['names'].items():
        print(f"  {idx}: {name}")
    
    # Count files
    stats = count_dataset_stats(config['path'])
    print(f"\nDataset Statistics:")
    for split, counts in stats.items():
        print(f"  {split.capitalize()}:")
        print(f"    Images: {counts['images']}")
        print(f"    Labels: {counts['labels']}")
    
    # Check integrity
    integrity = check_dataset_integrity(config_path)
    print(f"\nDataset Integrity:")
    if integrity['has_issues']:
        print("  ⚠ Issues found:")
        for issue in integrity['issues']:
            print(f"    - {issue}")
    else:
        print("  ✓ No issues detected")
    
    print("="*70)

if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print_dataset_info(config_path)
    else:
        print("Usage: python utils.py <path_to_config.yaml>")
