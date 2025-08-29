#!/usr/bin/env python3
"""
RDPM Demo Script

This script demonstrates the usage of RDPM models for rapid decline prediction in RCC patients.
It supports all three model architectures:
1. CT-only Model (image_only.py)
2. CT with Attention Model (image_w_attn.py)  
3. RDPM Hybrid Model (hybrid_model.py)

Usage:
    python demo.py --model hybrid --data_dir /path/to/data --json_file /path/to/metadata.json
    python demo.py --model image_only --data_dir /path/to/data --json_file /path/to/metadata.json
    python demo.py --model image_w_attn --data_dir /path/to/data --json_file /path/to/metadata.json

Requirements:
    - Model checkpoint files in checkpoints/ directory
    - Dataset with CT images and masks
    - Metadata JSON file with clinical parameters
"""

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Handle missing dependencies gracefully
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠ Warning: PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠ Warning: Pandas not available. Install with: pip install pandas")
    PANDAS_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠ Warning: Scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠ Warning: tqdm not available. Install with: pip install tqdm")
    TQDM_AVAILABLE = False
    # Fallback tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Only import model modules if torch is available
if TORCH_AVAILABLE:
    try:
        from src.models.hybrid_model import HybridModel
        from src.models.image_only import MaskedResNetClassifier
        from src.models.image_w_attn import MaskedResNetWithAttention
        from src.data.loaders import CTImageLoader
        MODELS_AVAILABLE = True
    except ImportError as e:
        print(f"⚠ Warning: Could not import model modules: {e}")
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


# Only define RDPMDemo class if dependencies are available
if TORCH_AVAILABLE and MODELS_AVAILABLE:
    class RDPMDemo:
        """
        RDPM Demonstration Class
        
        Handles model loading, data preparation, and inference for all RDPM model variants.
        """
        
        def __init__(self, model_type: str = 'hybrid', device: str = 'auto'):
            """
            Initialize the RDPM Demo
            
            Args:
                model_type: Type of model ('hybrid', 'image_only', 'image_w_attn')
                device: Device to run inference on ('auto', 'cpu', 'cuda')
            """
            self.model_type = model_type
            self.device = self._setup_device(device)
            self.model = None
            self.data_loader = None
            
            # Default paths - can be overridden
            self.checkpoint_paths = {
                'hybrid': 'checkpoints/rdpm_hybrid_best.pth',
                'image_only': 'checkpoints/rdpm_image_only_best.pth', 
                'image_w_attn': 'checkpoints/rdpm_image_w_attn_best.pth'
            }
            
            print(f"✓ RDPM Demo initialized for model type: {model_type}")
            print(f"✓ Using device: {self.device}")
        
        def _setup_device(self, device: str) -> torch.device:
            """Setup computation device"""
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                    print(f"✓ CUDA available with {torch.cuda.device_count()} GPU(s)")
                else:
                    device = 'cpu'
                    print("⚠ CUDA not available, using CPU")
            
            return torch.device(device)
    
    def load_model(self, checkpoint_path: Optional[str] = None, 
                   model_config: Optional[Dict] = None) -> None:
        """
        Load the specified model architecture and checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint file
            model_config: Configuration parameters for model initialization
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_paths[self.model_type]
        
        # Default model configurations
        default_configs = {
            'hybrid': {
                'spatial_dims': 3,
                'in_channels': 1,
                'num_classes': 2,
                'backbone': 'efficientnet-b0',
                'numerical_features_dim': 5,  # DM, maxdiameter, HTN, age, eGFR
                'fusion_method': 'multihead_cross_attention',
                'dropout_rate': 0.3,
                'use_mask_attention': True
            },
            'image_only': {
                'spatial_dims': 3,
                'in_channels': 1,
                'num_classes': 2,
                'backbone': 'resnet50',
                'dropout_rate': 0.3,
                'pretrained': False
            },
            'image_w_attn': {
                'spatial_dims': 3,
                'in_channels': 1,
                'num_classes': 2,
                'backbone': 'resnet50',
                'dropout_rate': 0.3,
                'use_mask_attention': True
            }
        }
        
        config = model_config or default_configs[self.model_type]
        
        print(f"🔄 Loading {self.model_type} model...")
        
        # Initialize model based on type
        if self.model_type == 'hybrid':
            self.model = HybridModel(**config)
        elif self.model_type == 'image_only':
            self.model = MaskedResNetClassifier(**config)
        elif self.model_type == 'image_w_attn':
            self.model = MaskedResNetWithAttention(**config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load checkpoint if available
        if os.path.exists(checkpoint_path):
            print(f"📁 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict with error handling
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print("✓ Checkpoint loaded successfully")
                
                # Print additional checkpoint info if available
                if isinstance(checkpoint, dict):
                    if 'epoch' in checkpoint:
                        print(f"   - Trained for {checkpoint['epoch']} epochs")
                    if 'best_auc' in checkpoint:
                        print(f"   - Best validation AUC: {checkpoint['best_auc']:.4f}")
                    if 'train_loss' in checkpoint:
                        print(f"   - Final training loss: {checkpoint['train_loss']:.4f}")
                        
            except RuntimeError as e:
                print(f"⚠ Warning: Could not load checkpoint strictly: {e}")
                print("   Attempting partial loading...")
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print("✓ Partial checkpoint loaded")
        else:
            print(f"⚠ Warning: Checkpoint not found at {checkpoint_path}")
            print("   Using randomly initialized weights")
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded successfully")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
    
    def setup_data(self, data_dir: str, json_file: str, 
                   batch_size: int = 4, split: str = 'test') -> None:
        """
        Setup data loader for inference
        
        Args:
            data_dir: Directory containing CT images and masks
            json_file: JSON file with metadata and labels
            batch_size: Batch size for inference
            split: Data split to use ('test', 'internal_test', 'external_test')
        """
        print(f"🔄 Setting up data loader...")
        print(f"   - Data directory: {data_dir}")
        print(f"   - Metadata file: {json_file}")
        print(f"   - Using split: {split}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON metadata file not found: {json_file}")
        
        # Initialize data loader
        self.data_loader_obj = CTImageLoader(
            data_dir=data_dir,
            json_file=json_file,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,  # Don't shuffle for demo
            roi_size=(128, 128, 32),
            cache_rate=0.0,  # No caching for demo
            use_augmentation=False  # No augmentation for inference
        )
        
        # Get data loader for specified split
        try:
            self.data_loader = self.data_loader_obj.get_data_loader(split=split)
            print(f"✓ Data loader created successfully")
            print(f"   - Number of batches: {len(self.data_loader)}")
            print(f"   - Batch size: {batch_size}")
            
            # Print dataset statistics
            self._print_dataset_stats(json_file, split)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create data loader: {e}")
    
    def _print_dataset_stats(self, json_file: str, split: str) -> None:
        """Print dataset statistics"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if split in data:
                items = data[split]
                total_samples = len(items)
                positive_samples = sum(1 for item in items if item.get('label', 0) == 1)
                negative_samples = total_samples - positive_samples
                
                print(f"   - Total samples: {total_samples}")
                print(f"   - Positive samples (rapid decline): {positive_samples}")
                print(f"   - Negative samples (stable): {negative_samples}")
                print(f"   - Class distribution: {positive_samples/total_samples:.2%} positive")
                
        except Exception as e:
            print(f"   - Could not load dataset statistics: {e}")
    
    def run_inference(self, save_results: bool = True, 
                     output_dir: str = 'demo_results') -> Dict:
        """
        Run inference on the loaded dataset
        
        Args:
            save_results: Whether to save detailed results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing inference results and metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.data_loader is None:
            raise ValueError("Data not setup. Call setup_data() first.")
        
        print(f"🔄 Running inference...")
        
        # Prepare results storage
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_filenames = []
        inference_times = []
        
        start_time = time.time()
        
        # Run inference
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Processing batches")):
                if batch is None:
                    continue
                
                batch_start_time = time.time()
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                try:
                    outputs = self.model(batch)
                    probabilities = F.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
                    
                    if 'label' in batch:
                        all_labels.extend(batch['label'].cpu().numpy())
                    
                    if 'filename' in batch:
                        batch_filenames = batch['filename']
                        if isinstance(batch_filenames, list):
                            all_filenames.extend(batch_filenames)
                        else:
                            all_filenames.extend([f"batch_{batch_idx}_sample_{i}" for i in range(len(predictions))])
                    else:
                        all_filenames.extend([f"batch_{batch_idx}_sample_{i}" for i in range(len(predictions))])
                    
                    batch_time = time.time() - batch_start_time
                    inference_times.append(batch_time)
                    
                except Exception as e:
                    print(f"⚠ Error processing batch {batch_idx}: {e}")
                    continue
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        results = {
            'model_type': self.model_type,
            'total_samples': len(all_predictions),
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'filenames': all_filenames,
            'inference_time': {
                'total_seconds': total_time,
                'average_per_batch': np.mean(inference_times),
                'average_per_sample': total_time / len(all_predictions) if all_predictions else 0
            }
        }
        
        # Add metrics if labels are available
        if all_labels:
            results['labels'] = all_labels
            results['metrics'] = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Print results summary
        self._print_results_summary(results)
        
        # Save results if requested
        if save_results:
            self._save_results(results, output_dir)
        
        return results
    
    def _calculate_metrics(self, labels: List[int], predictions: List[int], 
                          probabilities: List[float]) -> Dict:
        """Calculate performance metrics"""
        try:
            # Basic metrics
            accuracy = np.mean(np.array(predictions) == np.array(labels))
            
            # AUC if we have both classes
            auc = None
            if len(set(labels)) > 1:
                auc = roc_auc_score(labels, probabilities)
            
            # Classification report
            report = classification_report(labels, predictions, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'auc': auc,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'sensitivity': report['1']['recall'] if '1' in report else None,
                'specificity': report['0']['recall'] if '0' in report else None,
                'precision': report['1']['precision'] if '1' in report else None,
                'f1_score': report['1']['f1-score'] if '1' in report else None
            }
        except Exception as e:
            print(f"⚠ Error calculating metrics: {e}")
            return {}
    
    def _print_results_summary(self, results: Dict) -> None:
        """Print inference results summary"""
        print(f"\n📊 Inference Results Summary")
        print(f"{'='*50}")
        print(f"Model Type: {results['model_type'].upper()}")
        print(f"Total Samples: {results['total_samples']}")
        
        # Timing information
        timing = results['inference_time']
        print(f"\n⏱️ Timing Information:")
        print(f"   - Total inference time: {timing['total_seconds']:.2f} seconds")
        print(f"   - Average per batch: {timing['average_per_batch']:.3f} seconds")
        print(f"   - Average per sample: {timing['average_per_sample']:.3f} seconds")
        
        # Prediction distribution
        predictions = results['predictions']
        positive_pred = sum(predictions)
        negative_pred = len(predictions) - positive_pred
        print(f"\n🎯 Prediction Distribution:")
        print(f"   - Predicted positive (rapid decline): {positive_pred} ({positive_pred/len(predictions):.1%})")
        print(f"   - Predicted negative (stable): {negative_pred} ({negative_pred/len(predictions):.1%})")
        
        # Performance metrics (if labels available)
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n📈 Performance Metrics:")
            print(f"   - Accuracy: {metrics.get('accuracy', 0):.4f}")
            if metrics.get('auc') is not None:
                print(f"   - AUC: {metrics['auc']:.4f}")
            if metrics.get('sensitivity') is not None:
                print(f"   - Sensitivity: {metrics['sensitivity']:.4f}")
            if metrics.get('specificity') is not None:
                print(f"   - Specificity: {metrics['specificity']:.4f}")
            if metrics.get('precision') is not None:
                print(f"   - Precision: {metrics['precision']:.4f}")
            if metrics.get('f1_score') is not None:
                print(f"   - F1-Score: {metrics['f1_score']:.4f}")
    
    def _save_results(self, results: Dict, output_dir: str) -> None:
        """Save detailed results to files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save detailed predictions
            df_results = pd.DataFrame({
                'filename': results['filenames'],
                'prediction': results['predictions'],
                'probability': results['probabilities']
            })
            
            if 'labels' in results:
                df_results['true_label'] = results['labels']
                df_results['correct'] = df_results['prediction'] == df_results['true_label']
            
            predictions_file = os.path.join(output_dir, f'{self.model_type}_predictions.csv')
            df_results.to_csv(predictions_file, index=False)
            print(f"✓ Detailed predictions saved to: {predictions_file}")
            
            # Save summary metrics
            summary_file = os.path.join(output_dir, f'{self.model_type}_summary.json')
            summary_data = {
                'model_type': results['model_type'],
                'total_samples': results['total_samples'],
                'inference_time': results['inference_time']
            }
            if 'metrics' in results:
                summary_data['metrics'] = results['metrics']
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            print(f"✓ Summary metrics saved to: {summary_file}")
            
        except Exception as e:
            print(f"⚠ Error saving results: {e}")


def create_sample_data_structure():
    """Create sample data directory structure for demo"""
    print("\n📁 Sample Data Structure for RDPM Demo:")
    print("your_data/")
    print("├── images/")
    print("│   ├── patient001.nii.gz")
    print("│   ├── patient002.nii.gz")
    print("│   └── ...")
    print("├── masks/")
    print("│   ├── patient001_mask.nii.gz")
    print("│   ├── patient002_mask.nii.gz")
    print("│   └── ...")
    print("└── metadata.json")
    print()
    
    sample_metadata = {
        "internal_test": [
            {
                "filename": "patient001",
                "label": 0,
                "DM": 0,
                "maxdiameter": 3.5,
                "HTN": 1, 
                "age": 65,
                "eGFR": 75
            },
            {
                "filename": "patient002",
                "label": 1,
                "DM": 1,
                "maxdiameter": 6.2,
                "HTN": 1,
                "age": 72,
                "eGFR": 45
            }
        ]
    }
    
    print("📄 Sample metadata.json content:")
    print(json.dumps(sample_metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description='RDPM Demo Script')
    parser.add_argument('--model', choices=['hybrid', 'image_only', 'image_w_attn'], 
                       default='hybrid', help='Model type to use')
    parser.add_argument('--data_dir', type=str, required=False,
                       help='Directory containing CT images and masks')
    parser.add_argument('--json_file', type=str, required=False,
                       help='JSON file with metadata')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--split', type=str, default='internal_test',
                       choices=['test', 'internal_test', 'external_test'],
                       help='Data split to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                       help='Directory to save results')
    parser.add_argument('--show_sample_data', action='store_true',
                       help='Show sample data structure and exit')
    
    args = parser.parse_args()
    
    if args.show_sample_data:
        create_sample_data_structure()
        return
    
    # Check for required dependencies when running actual demo
    if not args.data_dir or not args.json_file:
        print("❌ Error: --data_dir and --json_file are required for running the demo")
        print("Use --show_sample_data to see the expected data format")
        return
    
    if not TORCH_AVAILABLE:
        print("❌ Error: PyTorch is required to run the demo")
        print("Install with: pip install torch")
        return
        
    if not MODELS_AVAILABLE:
        print("❌ Error: Could not import model modules")
        print("Make sure you're running from the RDPM directory and have all dependencies installed")
        return
    
    print("🚀 RDPM Demo Starting...")
    print(f"Arguments: {vars(args)}")
    
    try:
        # Initialize demo
        demo = RDPMDemo(model_type=args.model, device=args.device)
        
        # Load model
        demo.load_model(checkpoint_path=args.checkpoint)
        
        # Setup data
        demo.setup_data(
            data_dir=args.data_dir,
            json_file=args.json_file,
            batch_size=args.batch_size,
            split=args.split
        )
        
        # Run inference
        results = demo.run_inference(
            save_results=True,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
