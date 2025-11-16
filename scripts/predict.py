"""
Space Station Challenge - YOLO Prediction & Evaluation Script
==============================================================
This script evaluates a trained YOLO model and generates comprehensive metrics.

Usage:
    python predict.py --weights runs/train/exp/weights/best.pt --config ../configs/config.yaml
    python predict.py --weights runs/train/exp/weights/best.pt --source ../data/test/images
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import json

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_output_directory(weights_path):
    """Create directory for prediction outputs."""
    weights_dir = Path(weights_path).parent.parent
    output_dir = weights_dir / 'predictions'
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Space Station Object Detection', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def save_metrics_report(metrics, output_path):
    """Save detailed metrics report as text file."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SPACE STATION CHALLENGE - MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"mAP@0.5:        {metrics.get('map50', 0):.4f}\n")
        f.write(f"mAP@0.5:0.95:   {metrics.get('map', 0):.4f}\n")
        f.write(f"Precision:      {metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall:         {metrics.get('recall', 0):.4f}\n")
        f.write(f"F1-Score:       {metrics.get('f1', 0):.4f}\n\n")
        
        if 'class_metrics' in metrics:
            f.write("PER-CLASS METRICS\n")
            f.write("-"*70 + "\n")
            for class_name, class_metrics in metrics['class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {class_metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall:    {class_metrics.get('recall', 0):.4f}\n")
                f.write(f"  mAP@0.5:   {class_metrics.get('map50', 0):.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("BENCHMARK COMPARISON\n")
        f.write("="*70 + "\n")
        f.write("Target Benchmarks:\n")
        f.write("  mAP@0.5:  40-50% (Baseline), >70% (Excellent)\n")
        f.write("  Precision: >70% (Best models)\n")
        f.write("  Recall:    >70% (Best models)\n\n")
        
        map50 = metrics.get('map50', 0)
        if map50 >= 0.7:
            performance = "EXCELLENT"
        elif map50 >= 0.5:
            performance = "GOOD"
        elif map50 >= 0.4:
            performance = "BASELINE"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        f.write(f"Your Model Performance: {performance}\n")
        f.write("="*70 + "\n")
    
    print(f"Metrics report saved to: {output_path}")

def save_predictions_visualization(model, image_paths, output_dir, class_names, max_images=20):
    """Save visualization of predictions on sample images."""
    print(f"\nGenerating prediction visualizations...")
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Select random sample of images
    import random
    sample_paths = random.sample(image_paths, min(max_images, len(image_paths)))
    
    for i, img_path in enumerate(sample_paths):
        try:
            # Run prediction
            results = model.predict(img_path, conf=0.25, verbose=False)
            
            # Save annotated image
            if len(results) > 0:
                result_img = results[0].plot()
                output_path = viz_dir / f'prediction_{i:03d}.jpg'
                cv2.imwrite(str(output_path), result_img)
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")
    
    print(f"Visualizations saved to: {viz_dir}")

def analyze_failure_cases(results, output_dir, threshold=0.3):
    """Analyze and save information about failure cases."""
    print("\nAnalyzing failure cases...")
    
    failure_report = output_dir / 'failure_analysis.txt'
    
    with open(failure_report, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FAILURE CASE ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write("Common Issues to Consider:\n\n")
        f.write("1. OCCLUSION\n")
        f.write("   - Objects partially hidden by other objects\n")
        f.write("   - Solution: Add more occluded examples to training data\n\n")
        f.write("2. LIGHTING CONDITIONS\n")
        f.write("   - Poor performance in very dark or very light conditions\n")
        f.write("   - Solution: Enhance lighting augmentation, add more varied lighting\n\n")
        f.write("3. OBJECT SIZE\n")
        f.write("   - Small or distant objects are harder to detect\n")
        f.write("   - Solution: Use multi-scale training, adjust anchor boxes\n\n")
        f.write("4. CLASS CONFUSION\n")
        f.write("   - Similar-looking objects misclassified\n")
        f.write("   - Solution: Add more diverse examples, focus on distinctive features\n\n")
        f.write("5. OBJECT OVERLAP\n")
        f.write("   - Multiple objects of same class close together\n")
        f.write("   - Solution: Adjust NMS threshold, improve bounding box precision\n\n")
        f.write("="*70 + "\n")
        f.write("\nRECOMMENDATIONS FOR IMPROVEMENT:\n")
        f.write("-"*70 + "\n")
        f.write("- Increase training epochs\n")
        f.write("- Try a larger model (e.g., yolov8m or yolov8l)\n")
        f.write("- Adjust data augmentation parameters\n")
        f.write("- Collect more training data using Falcon\n")
        f.write("- Fine-tune confidence and IoU thresholds\n")
        f.write("="*70 + "\n")
    
    print(f"Failure analysis saved to: {failure_report}")

def save_metrics_json(metrics, output_path):
    """Save metrics as JSON for programmatic access."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {output_path}")

def evaluate_model(weights_path, config_path, source_path=None, save_viz=True):
    """Main evaluation function."""
    print("="*70)
    print("SPACE STATION CHALLENGE - MODEL EVALUATION")
    print("="*70)
    print(f"\nEvaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    config = load_config(config_path)
    class_names = list(config['names'].values())
    
    # Create output directory
    output_dir = create_output_directory(weights_path)
    print(f"Output directory: {output_dir}\n")
    
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}")
        print("Please ensure you have trained the model first using train.py")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("Using CPU\n")
    
    # Determine test data source
    if source_path is None:
        data_path = Path(config['path'])
        test_path = data_path / config.get('test', 'test/images')
    else:
        test_path = Path(source_path)
    
    print(f"Test data path: {test_path}\n")
    
    if not test_path.exists():
        print(f"ERROR: Test data path does not exist: {test_path}")
        print("Please update the config.yaml or provide --source argument")
        sys.exit(1)
    
    # Run validation
    print("Running model validation...\n")
    try:
        results = model.val(data=config_path, split='test')
        
        # Extract metrics
        metrics = {
            'map50': float(results.box.map50),
            'map': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
        }
        
        # Per-class metrics
        if hasattr(results.box, 'maps'):
            class_metrics = {}
            for i, class_name in enumerate(class_names):
                if i < len(results.box.maps):
                    class_metrics[class_name] = {
                        'map50': float(results.box.maps[i]),
                        'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0,
                        'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0
                    }
            metrics['class_metrics'] = class_metrics
        
        # Print results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"mAP@0.5:        {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        print(f"mAP@0.5:0.95:   {metrics['map']:.4f} ({metrics['map']*100:.2f}%)")
        print(f"Precision:      {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:         {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:       {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print("="*70 + "\n")
        
        # Save confusion matrix
        confusion_matrix_path = output_dir / 'confusion_matrix.png'
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            cm_normalized = results.confusion_matrix.matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_normalized, annot=True, fmt='.0f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {confusion_matrix_path}")
        
        # Save metrics report
        report_path = output_dir / 'evaluation_report.txt'
        save_metrics_report(metrics, report_path)
        
        # Save metrics JSON
        json_path = output_dir / 'metrics.json'
        save_metrics_json(metrics, json_path)
        
        # Analyze failure cases
        analyze_failure_cases(results, output_dir)
        
        # Generate prediction visualizations
        if save_viz:
            image_paths = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
            if image_paths:
                save_predictions_visualization(model, image_paths, output_dir, class_names)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nAll results saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - evaluation_report.txt: Detailed metrics report")
        print("  - metrics.json: Machine-readable metrics")
        print("  - confusion_matrix.png: Visual confusion matrix")
        print("  - failure_analysis.txt: Recommendations for improvement")
        print("  - visualizations/: Sample prediction images")
        print("\nNext steps:")
        print("1. Review the evaluation report and confusion matrix")
        print("2. Analyze failure cases to identify improvement areas")
        print("3. Adjust training parameters and retrain if needed")
        print("4. Document your findings in the hackathon report")
        print("="*70 + "\n")
        
        return metrics
        
    except Exception as e:
        print(f"\nERROR during evaluation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model was trained successfully")
        print("2. Check that test data path is correct in config.yaml")
        print("3. Verify that test images and labels exist")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for Space Station Challenge')
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to test images (overrides config)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip generating prediction visualizations')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run evaluation
    evaluate_model(args.weights, args.config, args.source, not args.no_viz)

if __name__ == '__main__':
    main()
