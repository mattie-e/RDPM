import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional


class NormalizeNumericalFeatures:
    """Z-score normalize clinical features"""
    def __init__(self, feature_stats=None):
        self.feature_stats = feature_stats or {}

    def __call__(self, data):
        for feature in ['DM', 'maxdiameter', 'HTN', 'age', 'eGFR']:
            if feature in data:
                value = data[feature].item() if torch.is_tensor(data[feature]) else data[feature]

                if feature in self.feature_stats:
                    mean, std = self.feature_stats[feature]
                    value = (value - mean) / (std + 1e-8)

                data[f'{feature}_normalized'] = torch.tensor(value, dtype=torch.float32)
            else:
                data[f'{feature}_normalized'] = torch.tensor(0.0, dtype=torch.float32)

        return data


class PreprocessedDataset(Dataset):
    """
    Dataset for loading preprocessed CT image tensors and clinical metadata.

    Expects data already preprocessed and saved as tensor files (.pt or .npy).
    No image loading, orientation, resampling, or intensity normalization is performed.

    Directory structure:
        data_dir/
        ├── patient001.pt          # Preprocessed image tensor [C, D, H, W]
        ├── patient001_mask.pt     # Preprocessed mask tensor [C, D, H, W]
        ├── patient002.pt
        ├── patient002_mask.pt
        └── ...

    Metadata JSON format:
        {
            "internal_train": [
                {
                    "filename": "patient001",
                    "label": 0,
                    "DM": 0, "maxdiameter": 3.5, "HTN": 1, "age": 65, "eGFR": 75
                },
                ...
            ]
        }
    """

    def __init__(self, data_items, data_dir, feature_stats=None):
        """
        Args:
            data_items: List of dicts with 'filename', 'label', and clinical features
            data_dir: Directory containing preprocessed .pt or .npy files
            feature_stats: Dict of {feature_name: (mean, std)} for normalization
        """
        self.data_items = data_items
        self.data_dir = data_dir
        self.normalize = NormalizeNumericalFeatures(feature_stats)

    def __len__(self):
        return len(self.data_items)

    def _load_tensor(self, path):
        """Load a tensor from .pt or .npy file"""
        if path.endswith('.pt'):
            return torch.load(path, map_location='cpu')
        elif path.endswith('.npy'):
            return torch.from_numpy(np.load(path)).float()
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _find_file(self, base_path):
        """Find file with .pt or .npy extension"""
        for ext in ['.pt', '.npy']:
            path = base_path + ext
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No preprocessed file found for: {base_path} (tried .pt, .npy)")

    def __getitem__(self, idx):
        item = self.data_items[idx]
        filename = item['filename']

        # Load preprocessed image tensor
        image_path = self._find_file(os.path.join(self.data_dir, filename))
        image = self._load_tensor(image_path)

        data = {
            'image': image,
            'filename': os.path.basename(filename),
        }

        # Load preprocessed mask tensor if available
        mask_base = os.path.join(self.data_dir, filename + "_mask")
        for ext in ['.pt', '.npy']:
            mask_path = mask_base + ext
            if os.path.exists(mask_path):
                data['mask'] = self._load_tensor(mask_path)
                data['mask_volume'] = torch.sum(data['mask']).float()
                break

        # Add label
        if 'label' in item:
            data['label'] = torch.tensor(int(item['label']))
        elif 'rapid_decline' in item:
            data['label'] = torch.tensor(int(item['rapid_decline'][0]))

        # Add clinical features
        for key in ['DM', 'maxdiameter', 'HTN', 'age', 'eGFR']:
            if key in item:
                data[key] = torch.tensor(float(item[key]))

        # Normalize clinical features
        data = self.normalize(data)

        return data


def collate_fn(batch):
    """Collate function that handles dict-based samples"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                collated[key] = values
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values

    return collated


class CTDataLoader:
    """
    Data loader for preprocessed CT data.

    Assumes all image preprocessing (loading, orientation, resampling,
    intensity normalization, resizing) has already been done.
    Only handles clinical feature normalization and DataLoader creation.
    """

    def __init__(self, data_dir, json_file, batch_size=4, num_workers=2, shuffle=True):
        """
        Args:
            data_dir: Directory containing preprocessed tensor files (.pt or .npy)
            json_file: JSON file with metadata and clinical features
            batch_size: Batch size for DataLoader
            num_workers: Number of DataLoader workers
            shuffle: Whether to shuffle training data
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        if not data_dir or not os.path.exists(data_dir):
            raise ValueError(f"data_dir must be provided and exist: {data_dir}")
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")

    def _load_data_from_json(self, split='train'):
        """Load dataset items from JSON metadata file"""
        with open(self.json_file, 'r') as f:
            split_data = json.load(f)

        available_splits = list(split_data.keys())
        print(f"Available splits in JSON: {available_splits}")

        split_mapping = {
            'train': ['internal_train'],
            'val': [],
            'test': ['internal_test', 'external_test'],
            'internal_test': ['internal_test'],
            'external_test': ['external_test']
        }

        items = None
        used_split = None

        if split in ['train', 'all_train'] and 'internal_train' in split_data:
            # For cross-validation: return all internal_train data,
            # let StratifiedKFold handle train/val splitting per fold.
            # No fixed val carve-out — CV folds rotate validation.
            items = split_data['internal_train'].copy()
            used_split = 'internal_train'
            print(f"Using full training set: {len(items)} samples")

        elif split in split_mapping:
            for possible_split in split_mapping.get(split, [split]):
                if possible_split in split_data:
                    items = split_data[possible_split]
                    used_split = possible_split
                    print(f"Using split '{used_split}' for requested split '{split}'")
                    break

        elif split in split_data:
            items = split_data[split]
            used_split = split
            print(f"Using direct split '{used_split}'")

        if items is None:
            raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")

        print(f"Loaded {len(items)} items from split '{used_split}'")
        return items

    def get_feature_statistics(self):
        """Compute mean/std for clinical features from training data only"""
        if not hasattr(self, '_feature_stats'):
            with open(self.json_file, 'r') as f:
                data = json.load(f)

            feature_values = {key: [] for key in ['DM', 'maxdiameter', 'HTN', 'age', 'eGFR']}

            # Only use training splits to avoid test data leakage
            train_splits = ['internal_train', 'train']
            for split_name in train_splits:
                if split_name in data and isinstance(data[split_name], list):
                    for item in data[split_name]:
                        for key in feature_values:
                            if key in item:
                                feature_values[key].append(float(item[key]))

            self._feature_stats = {}
            for key, values in feature_values.items():
                if values:
                    self._feature_stats[key] = (np.mean(values), np.std(values))

        return self._feature_stats

    def create_dataloader(self, split='train'):
        """
        Create a DataLoader for the specified split.

        Args:
            split: Data split ('train', 'val', 'test', 'internal_test', 'external_test')

        Returns:
            DataLoader with preprocessed data
        """
        items = self._load_data_from_json(split)
        feature_stats = self.get_feature_statistics()

        dataset = PreprocessedDataset(
            data_items=items,
            data_dir=self.data_dir,
            feature_stats=feature_stats
        )

        is_train = split in ['train', 'internal_train']

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and is_train,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=is_train,
            collate_fn=collate_fn
        )

        print(f"Created dataloader with {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader

    def get_dataset_items(self, split='train'):
        """Get raw dataset items for use with k-fold CV"""
        return self._load_data_from_json(split)
