#!/usr/bin/env python3
"""
RDPM Demo Script

Demonstrates the usage of RDPM models for rapid decline prediction in RCC patients.
Expects preprocessed data (tensor files). No image preprocessing is performed.

Supports all three model architectures:
1. CT-only Model (image_only.py)
2. CT with Attention Model (image_w_attn.py)
3. RDPM Hybrid Model (hybrid_model.py)

Usage:
    python demo.py --model hybrid --data_dir /path/to/preprocessed_data --json_file /path/to/metadata.json
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.hybrid_model import MultiModalCTClassifier as HybridModel
from src.models.image_only import MaskedResNetClassifier
from src.data.loaders import CTDataLoader


class RDPMDemo:
    """RDPM Demonstration Class for running inference on preprocessed data."""

    def __init__(self, model_type: str = 'hybrid', device: str = 'auto'):
        self.model_type = model_type
        self.device = self._setup_device(device)
        self.model = None
        self.data_loader = None

        print(f"RDPM Demo initialized: model={model_type}, device={self.device}")

    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)

    def load_model(self, weights_path: Optional[str] = None,
                   model_config: Optional[Dict] = None) -> None:
        """Load model architecture and optional pretrained weights."""

        default_configs = {
            'hybrid': {
                'spatial_dims': 3, 'in_channels': 1, 'num_classes': 2,
                'backbone': 'efficientnet-b0', 'numerical_features_dim': 5,
                'fusion_method': 'multihead_cross_attention',
                'dropout_rate': 0.3, 'use_mask_attention': True
            },
            'image_only': {
                'spatial_dims': 3, 'in_channels': 1, 'num_classes': 2,
                'backbone': 'resnet50', 'dropout_rate': 0.3, 'pretrained': False
            },
            'image_w_attn': {
                'spatial_dims': 3, 'in_channels': 1, 'num_classes': 2,
                'backbone': 'resnet50', 'dropout_rate': 0.3, 'use_mask_attention': True
            }
        }

        config = model_config or default_configs[self.model_type]

        if self.model_type == 'hybrid':
            self.model = HybridModel(**config)
        elif self.model_type == 'image_only':
            self.model = MaskedResNetClassifier(**config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights: {weights_path}")
            saved = torch.load(weights_path, map_location=self.device)

            if 'model_state_dict' in saved:
                state_dict = saved['model_state_dict']
            elif 'state_dict' in saved:
                state_dict = saved['state_dict']
            else:
                state_dict = saved

            self.model.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully")

        self.model = self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params:,} parameters")

    def setup_data(self, data_dir: str, json_file: str,
                   batch_size: int = 4, split: str = 'test') -> None:
        """Setup data loader for preprocessed data."""
        loader = CTDataLoader(
            data_dir=data_dir,
            json_file=json_file,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False
        )

        self.data_loader = loader.create_dataloader(split=split)
        print(f"Data loader ready: {len(self.data_loader)} batches")

    def run_inference(self) -> Dict:
        """Run inference and return predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.data_loader is None:
            raise ValueError("Data not setup. Call setup_data() first.")

        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_filenames = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Inference"):
                if batch is None:
                    continue

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                outputs = self.model(batch)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())

                if 'label' in batch:
                    all_labels.extend(batch['label'].cpu().numpy())

                if 'filename' in batch:
                    if isinstance(batch['filename'], list):
                        all_filenames.extend(batch['filename'])

        results = {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'filenames': all_filenames,
            'total_samples': len(all_predictions)
        }

        if all_labels:
            results['labels'] = all_labels

        # Print summary
        positive = sum(all_predictions)
        negative = len(all_predictions) - positive
        print(f"\nResults: {len(all_predictions)} samples")
        print(f"  Predicted positive (rapid decline): {positive}")
        print(f"  Predicted negative (stable): {negative}")

        return results


def main():
    parser = argparse.ArgumentParser(description='RDPM Demo (Preprocessed Data)')
    parser.add_argument('--model', choices=['hybrid', 'image_only', 'image_w_attn'],
                       default='hybrid', help='Model type')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with preprocessed tensor files (.pt or .npy)')
    parser.add_argument('--json_file', type=str, required=True,
                       help='JSON metadata file')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to pretrained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--split', type=str, default='internal_test',
                       choices=['test', 'internal_test', 'external_test'])
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])

    args = parser.parse_args()

    demo = RDPMDemo(model_type=args.model, device=args.device)
    demo.load_model(weights_path=args.weights)
    demo.setup_data(
        data_dir=args.data_dir,
        json_file=args.json_file,
        batch_size=args.batch_size,
        split=args.split
    )
    results = demo.run_inference()

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
