import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from monai.data import CacheDataset, list_data_collate, pad_list_data_collate
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Resized,
    LoadImaged,
    RandRotated,
    RandZoomd
)
from typing import Dict, Tuple, List, Optional, Union


class SafeTransform:
    def __init__(self, transform, transform_name="Unknown"):
        self.transform = transform
        self.transform_name = transform_name
    
    def __call__(self, data):
        try:
            return self.transform(data)
        except Exception as e:
            print(f"Error in {self.transform_name}: {e}")
            return data


def safe_collate_fn(batch):
    
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    try:
        return pad_list_data_collate(batch)
    except Exception as e:
        print(f"Error in pad collation: {e}")
        try:
            return list_data_collate(batch)
        except Exception as e2:
            print(f"Error in regular collation: {e2}")
            valid_batch = []
            for i, item in enumerate(batch):
                try:
                    test_result = list_data_collate([item])
                    valid_batch.append(item)
                except Exception as item_error:
                    print(f"Skipping problematic item {i}: {item_error}")
                    continue
            
            if len(valid_batch) > 0:
                try:
                    return pad_list_data_collate(valid_batch)
                except:
                    return list_data_collate(valid_batch)
            else:
                return None


class SafeLoadImaged(LoadImaged):
    def __call__(self, data):
        try:
            result = super().__call__(data)
            if result is None:
                return None
            if 'image' in result and result['image'] is not None:
                return result
            else:
                print(f"Failed to load image data from {data.get('image', 'unknown')}")
                return None
        except Exception as e:
            print(f"Error loading file {data.get('image', 'unknown')}: {e}")
            return None


class IncludeMetadataTransform:
    def __init__(self, json_file=None):
        self.patient_metadata = {}
        if json_file:
            self._load_metadata(json_file)
    
    def _load_metadata(self, json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for split_name, items in data.items():
                if isinstance(items, list):
                    for item in items:
                        if 'filename' in item:
                            filename = os.path.basename(item['filename'])
                            self.patient_metadata[filename] = {}
                            
                            for key in ['DM', 'maxdiameter', 'HTN', 'age', 'eGFR']:
                                if key in item:
                                    self.patient_metadata[filename][key] = float(item[key])
                                elif 'metadata' in item and key in item['metadata']:
                                    self.patient_metadata[filename][key] = float(item['metadata'][key])
        except Exception as e:
            print(f"Error loading metadata from {json_file}: {e}")
    
    def __call__(self, data):
        if isinstance(data, dict) and 'filename' in data:
            filename = data['filename']
            if filename in self.patient_metadata:
                for key, value in self.patient_metadata[filename].items():
                    data[key] = value
        return data


class NormalizeNumericalFeatures:
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

class MaskImageTransform:
    def __call__(self, data):
        if 'image' in data and 'mask' in data:
            image = data['image']
            mask = data['mask']
            
            mask_volume = torch.sum(mask).float()
            data['mask_volume'] = mask_volume
            
        return data

class CTImageLoader:
    def __init__(self, data_dir=None, json_file=None, batch_size=4, num_workers=2, shuffle=True, 
                 roi_size=(128, 128, 32), cache_rate=0.5, use_augmentation=True):
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.roi_size = roi_size
        self.cache_rate = cache_rate
        self.use_augmentation = use_augmentation
        
        if not data_dir or not os.path.exists(data_dir):
            raise ValueError(f"data_dir must be provided and exist: {data_dir}")
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")

    def _load_data_from_json(self, split='train'):
        if not self.json_file:
            raise ValueError("JSON file must be provided for structured data loading")
            
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
        
        if split in ['train', 'val'] and 'internal_train' in split_data:
            import random
            random.seed(42)
            
            all_train_items = split_data['internal_train'].copy()
            random.shuffle(all_train_items)
            
            val_size = int(len(all_train_items) * 0.1)
            
            if split == 'val':
                items = all_train_items[:val_size]
                used_split = 'internal_train (validation split)'
                print(f"Created validation split from internal_train: {len(items)} samples (10% of {len(all_train_items)})")
            if split == 'train':
                items = split_data['internal_train'].copy()
                used_split = 'internal_train (training split)'
                print(f"Using training split from internal_train: {len(items)} samples (90% of {len(all_train_items)})")
        
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

        dataset_items = []
        for item in items:
            dataset_item = {
                "image": os.path.join(self.data_dir, item['filename'] + ".nii.gz")
            }
            
            mask_path = os.path.join(self.data_dir, item['filename'] + "_mask.nii.gz")
            if os.path.exists(mask_path):
                dataset_item["mask"] = mask_path
            
            if 'label' in item:
                dataset_item["label"] = torch.tensor(int(item['label']))
            elif 'rapid_decline' in item:
                dataset_item["label"] = torch.tensor(int(item['rapid_decline'][0]))
            
            for key in ['DM', 'maxdiameter', 'HTN', 'age', 'eGFR']:
                if key in item:
                    dataset_item[key] = torch.tensor(float(item[key]))
                
            dataset_item["filename"] = os.path.basename(item['filename'])
            
            dataset_items.append(dataset_item)
            
        print(f"Loaded {len(dataset_items)} items from split '{used_split}'")
        return dataset_items

    def _load_data_from_directory(self):
        images = []
        labels = []
        
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.endswith(('.nii', '.nii.gz')):
                        img_path = os.path.join(label_dir, img_file)
                        images.append({"image": img_path, "label": torch.tensor(int(label))})
                        
        return images

    def get_training_transforms(self):
        transforms = [
            SafeTransform(SafeLoadImaged(keys=["image", "mask"], allow_missing_keys=True), "SafeLoadImaged"),
            SafeTransform(EnsureChannelFirstd(keys=["image", "mask"], allow_missing_keys=True), "EnsureChannelFirstd"),
            SafeTransform(Orientationd(keys=["image", "mask"], axcodes="RAS", allow_missing_keys=True), "Orientationd"),
            SafeTransform(Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"], allow_missing_keys=True), "Spacingd"),
            SafeTransform(ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True), "ScaleIntensityRanged"),
            SafeTransform(Resized(keys=["image", "mask"], spatial_size=self.roi_size, mode=["trilinear", "nearest"], allow_missing_keys=True), "Resized"),
            SafeTransform(MaskImageTransform(), "MaskImageTransform"),
            SafeTransform(NormalizeNumericalFeatures(self.get_feature_statistics()), "NormalizeNumericalFeatures"),
        ]
        
        if self.use_augmentation:
            augmentation_transforms = [
                SafeTransform(RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.2, allow_missing_keys=True), "RandFlipd_X"),
                SafeTransform(RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.2, allow_missing_keys=True), "RandFlipd_Y"),
            ]
            transforms.extend(augmentation_transforms)
        
        transforms.append(SafeTransform(ToTensord(keys=["image", "mask"], allow_missing_keys=True), "ToTensord"))
        
        return Compose(transforms)

    def get_inference_transforms(self):
        return self.get_training_transforms()

    def create_dataloader(self, split='train', distributed=False, rank=0, world_size=1):
        if self.json_file:
            dataset_items = self._load_data_from_json(split)
        else:
            dataset_items = self._load_data_from_directory()
        
        transform = self.get_training_transforms()
        
        dataset = CacheDataset(
            data=dataset_items,
            transform=transform,
            cache_rate=self.cache_rate,
            num_workers=max(1, self.num_workers)
        )
        
        print(f"Created dataset with {len(dataset)} items")
        
        persistent = self.num_workers > 0 if split == 'internal_train' else False
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and split == 'internal_train',
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=split == 'internal_train',
            collate_fn=list_data_collate,
            persistent_workers=persistent
        )
        
        return dataloader

    def get_transforms(self):
        return self.get_training_transforms()

    def get_feature_statistics(self):
        if not hasattr(self, '_feature_stats'):
            if self.json_file:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                
                all_ages, all_dms, all_maxdiameters, all_htns, all_egfrs = [], [], [], [], []
                
                for split_name, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            if 'DM' in item:
                                all_dms.append(float(item['DM']))
                            if 'maxdiameter' in item:
                                all_maxdiameters.append(float(item['maxdiameter']))
                            if 'age' in item:
                                all_ages.append(float(item['age']))
                            if 'HTN' in item:
                                all_htns.append(float(item['HTN']))
                            if 'eGFR' in item:
                                all_egfrs.append(float(item['eGFR']))
                
                self._feature_stats = {}
                if all_dms:
                    self._feature_stats['DM'] = (np.mean(all_dms), np.std(all_dms))
                if all_maxdiameters:
                    self._feature_stats['maxdiameter'] = (np.mean(all_maxdiameters), np.std(all_maxdiameters))
                if all_ages:
                    self._feature_stats['age'] = (np.mean(all_ages), np.std(all_ages))
                if all_htns:
                    self._feature_stats['HTN'] = (np.mean(all_htns), np.std(all_htns))
                if all_egfrs:
                    self._feature_stats['eGFR'] = (np.mean(all_egfrs), np.std(all_egfrs))
            else:
                self._feature_stats = {}
                
        return self._feature_stats

    def get_preprocessing_transforms(self, output_size=None):
        target_size = output_size if output_size is not None else self.roi_size
        
        transforms = [
            SafeTransform(SafeLoadImaged(keys=["image", "mask"], allow_missing_keys=True), "SafeLoadImaged"),
            SafeTransform(EnsureChannelFirstd(keys=["image", "mask"], allow_missing_keys=True), "EnsureChannelFirstd"),
            SafeTransform(Orientationd(keys=["image", "mask"], axcodes="RAS", allow_missing_keys=True), "Orientationd"),
            SafeTransform(Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"], allow_missing_keys=True), "Spacingd"),
            SafeTransform(ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True), "ScaleIntensityRanged"),
            SafeTransform(Resized(keys=["image", "mask"], spatial_size=target_size, mode=["trilinear", "nearest"], allow_missing_keys=True), "Resized"),
            SafeTransform(MaskImageTransform(), "MaskImageTransform"),
            SafeTransform(NormalizeNumericalFeatures(self.get_feature_statistics()), "NormalizeNumericalFeatures"),
            SafeTransform(ToTensord(keys=["image", "mask"], allow_missing_keys=True), "ToTensord"),
        ]
        
        return Compose(transforms)

class CustomDataset(Dataset):
    def __init__(self, data_dir, target_size=(256, 256), transform=None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.transform = transform
        self.image_files = self._get_image_files()
    
    def _get_image_files(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.nii', '.nii.gz']
        files = []
        for file in os.listdir(self.data_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                files.append(os.path.join(self.data_dir, file))
        return files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        if self.transform:
            image = self.transform(image_path)
        else:
            image = image_path
        return image, image_path