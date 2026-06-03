import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import json
import random
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np

DEFAULT_CLASSIFICATION_THRESHOLD = 0.3
DEFAULT_SELECTION_METRIC = 'val_auc'
DEFAULT_LEARNING_RATE = 2e-3
DEFAULT_EPOCHS = 250
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_CLASS_WEIGHTS = [0.8, 1.2]
DEFAULT_RANDOM_SEED = 42
CLINICAL_FEATURES = ['DM', 'maxdiameter', 'HTN', 'age', 'eGFR']
SELECTION_METRIC_MODES = {
    'val_auc': 'max',
}


def _extract_label(item: Dict[str, Any]) -> int:
    if 'label' in item:
        label = item['label']
    elif 'rapid_decline' in item:
        label = item['rapid_decline']
        if isinstance(label, (list, tuple, np.ndarray)):
            label = label[0]
    else:
        raise ValueError("Training item is missing 'label' or 'rapid_decline'.")

    if torch.is_tensor(label):
        label = label.item()
    return int(label)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_feature_statistics(data_items: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Compute clinical feature normalization statistics from the provided items only."""
    feature_values = {key: [] for key in CLINICAL_FEATURES}

    for item in data_items:
        for key in CLINICAL_FEATURES:
            if key in item:
                feature_values[key].append(float(item[key]))

    feature_stats = {}
    for key, values in feature_values.items():
        if values:
            feature_stats[key] = (float(np.mean(values)), float(np.std(values)))

    return feature_stats


def _selection_mode(selection_metric: str) -> str:
    if selection_metric not in SELECTION_METRIC_MODES:
        valid = ', '.join(sorted(SELECTION_METRIC_MODES))
        raise ValueError(f"Unknown selection_metric '{selection_metric}'. Valid options: {valid}")
    return SELECTION_METRIC_MODES[selection_metric]


def _is_better_score(score: float, best_score: float, mode: str) -> bool:
    if mode == 'min':
        return score < best_score
    return score > best_score


def _best_epoch_index(history: Dict[str, List[Any]], selection_metric: str, mode: str) -> int:
    values = np.array(history.get(selection_metric, []), dtype=float)
    if values.size == 0:
        return 0

    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return 0

    masked_values = np.where(finite_mask, values, np.inf if mode == 'min' else -np.inf)
    return int(np.argmin(masked_values) if mode == 'min' else np.argmax(masked_values))


def _normalize_batch_features(
    batch: Dict[str, Any],
    feature_stats: Optional[Dict[str, Tuple[float, float]]]
) -> Dict[str, Any]:
    if not feature_stats:
        return batch

    normalized_batch = dict(batch)
    for feature in CLINICAL_FEATURES:
        if feature in batch and feature in feature_stats:
            values = batch[feature]
            if not torch.is_tensor(values):
                values = torch.tensor(values, dtype=torch.float32)
            mean, std = feature_stats[feature]
            normalized_batch[f'{feature}_normalized'] = (values.float() - mean) / (std + 1e-8)

    return normalized_batch


def _model_name_from_config(config: Optional[Dict[str, Any]], model: nn.Module) -> str:
    if config:
        model_name = config.get('model', {}).get('name')
        if model_name:
            return model_name

    class_name = model.__class__.__name__.lower()
    if class_name in ['multimodalctclassifier', 'hybridmodel']:
        return 'hybrid'
    if class_name == 'imagewithattentionmodel':
        return 'image_w_attn'
    if class_name == 'maskedresnetwithattention':
        return 'masked_resnet_attention'
    if class_name == 'maskedresnetclassifier':
        return 'masked_resnet'
    return 'image_tensor'


def _model_outputs(model: nn.Module, batch: Any, config: Optional[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    if isinstance(batch, dict):
        if 'image' not in batch and 'masked_image' not in batch:
            raise ValueError("Batch must contain 'image' or 'masked_image'.")

        model_name = _model_name_from_config(config, model)
        if model_name in ['hybrid', 'masked_resnet', 'masked_resnet_attention', 'image_w_attn']:
            return model(batch)

        images = batch.get('image', batch.get('masked_image')).to(device, non_blocking=True)
        return model(images)

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        if isinstance(batch[0], dict):
            return _model_outputs(model, batch[0], config, device)
        return model(batch[0].to(device, non_blocking=True))

    if torch.is_tensor(batch):
        return model(batch.to(device, non_blocking=True))

    raise ValueError(f"Unsupported batch type: {type(batch)!r}")


def _batch_labels(batch: Any, batch_size: int, device: torch.device, require_labels: bool = False) -> torch.Tensor:
    if isinstance(batch, dict):
        labels = batch.get('label')
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        if isinstance(batch[0], dict):
            labels = batch[0].get('label')
        else:
            labels = batch[1]
    else:
        labels = None

    if labels is None and require_labels:
        raise ValueError("Batch is missing labels required for training or evaluation.")
    if labels is None:
        labels = torch.zeros(batch_size, dtype=torch.long)

    labels = labels.to(device, non_blocking=True) if torch.is_tensor(labels) else torch.tensor(labels, device=device)
    if labels.dim() > 1:
        labels = labels.squeeze()
    return labels.long()


def _augment_image_batch(batch: Any, max_rotation_degrees: float = 15.0, flip_probability: float = 0.5) -> Any:
    if not isinstance(batch, dict) or 'image' not in batch or not torch.is_tensor(batch['image']):
        return batch

    images = batch['image']
    if images.dim() != 5:
        return batch

    def match_to_image_shape(tensor: torch.Tensor, mode: str) -> torch.Tensor:
        if tensor.dim() == images.dim() - 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == images.dim() - 1:
            if tensor.shape[0] == images.shape[0]:
                tensor = tensor.unsqueeze(1)
            else:
                tensor = tensor.unsqueeze(0)

        if tensor.shape[0] == 1 and images.shape[0] > 1:
            tensor = tensor.expand(images.shape[0], *tensor.shape[1:])
        if tensor.shape[1] == 1 and images.shape[1] > 1:
            tensor = tensor.expand(tensor.shape[0], images.shape[1], *tensor.shape[2:])
        if tensor.shape[2:] != images.shape[2:]:
            if mode == 'nearest':
                tensor = F.interpolate(tensor.float(), size=images.shape[2:], mode='nearest')
            else:
                tensor = F.interpolate(tensor.float(), size=images.shape[2:], mode='trilinear', align_corners=False)
        return tensor

    augmented = dict(batch)
    transform_modes = {}
    for key in ['image', 'masked_image']:
        if key in batch and torch.is_tensor(batch[key]):
            augmented[key] = match_to_image_shape(batch[key], mode='trilinear')
            transform_modes[key] = 'bilinear'
    if 'mask' in batch and torch.is_tensor(batch['mask']):
        augmented['mask'] = (match_to_image_shape(batch['mask'], mode='nearest') > 0.5).float()
        transform_modes['mask'] = 'nearest'

    for dim in [2, 3, 4]:
        if torch.rand(1).item() < flip_probability:
            for key in transform_modes:
                augmented[key] = torch.flip(augmented[key], dims=[dim])

    if max_rotation_degrees > 0:
        angle = (torch.rand(1).item() * 2.0 - 1.0) * max_rotation_degrees * np.pi / 180.0
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        theta = images.new_tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a, 0.0],
            [0.0, sin_a, cos_a, 0.0],
        ]).unsqueeze(0).repeat(images.shape[0], 1, 1)

        grid = F.affine_grid(theta, images.shape, align_corners=False)
        for key, mode in transform_modes.items():
            augmented[key] = F.grid_sample(augmented[key], grid, mode=mode, padding_mode='border', align_corners=False)
            if key == 'mask':
                augmented[key] = (augmented[key] > 0.5).float()

    return augmented

def calculate_auc_with_ci(y_true, y_pred_proba, confidence_level=0.95, n_bootstrap=1000):
    """
    Calculate AUC with 95% confidence interval using bootstrap method
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred_proba, np.ndarray):
        y_pred_proba = np.array(y_pred_proba)

    try:
        original_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        return 0.0, 0.0, 0.0

    bootstrap_aucs = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_bootstrap = y_true[bootstrap_indices]
        y_pred_bootstrap = y_pred_proba[bootstrap_indices]

        try:
            if len(np.unique(y_bootstrap)) > 1:
                bootstrap_auc = roc_auc_score(y_bootstrap, y_pred_bootstrap)
                bootstrap_aucs.append(bootstrap_auc)
        except ValueError:
            continue

    if len(bootstrap_aucs) == 0:
        return original_auc, original_auc, original_auc

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_aucs, (1 - alpha/2) * 100)

    return original_auc, ci_lower, ci_upper

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                epochs: int = DEFAULT_EPOCHS,
                learning_rate: float = DEFAULT_LEARNING_RATE,
                device: str = 'cuda',
                save_dir: str = './checkpoints',
                config: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
    """
    Train a model with the given data loaders

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader required for AUC-based model selection
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints

    Returns:
        Dictionary containing training history
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    requested_selection_metric = kwargs.get('selection_metric')
    if requested_selection_metric and requested_selection_metric != DEFAULT_SELECTION_METRIC:
        print(f"Ignoring selection_metric='{requested_selection_metric}'; model selection is fixed to {DEFAULT_SELECTION_METRIC}.")
    selection_metric = DEFAULT_SELECTION_METRIC
    selection_mode = _selection_mode(selection_metric)
    best_val_score = np.inf if selection_mode == 'min' else -np.inf
    start_epoch = 0
    random_seed = int(kwargs.get('random_seed', DEFAULT_RANDOM_SEED))
    _set_random_seed(random_seed)

    weight_decay = kwargs.get('weight_decay', 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    class_weights = torch.tensor(kwargs.get('class_weights', DEFAULT_CLASS_WEIGHTS), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=kwargs.get('min_learning_rate', 1e-6)
    )
    early_stopping_patience = kwargs.get('early_stopping_patience', DEFAULT_EARLY_STOPPING_PATIENCE)
    epochs_without_improvement = 0
    use_amp = bool(kwargs.get('use_amp', True)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_augmentation = bool(kwargs.get('use_augmentation', True))
    rotation_degrees = float(kwargs.get('rotation_degrees', 15.0))
    flip_probability = float(kwargs.get('flip_probability', 0.5))

    if val_loader is None:
        raise ValueError(
            "A validation loader is required for RDPM training so that AUC-based "
            "model selection and early stopping are pre-specified and reproducible."
        )

    os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_auc_ci_lower': [],
        'val_auc_ci_upper': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_confusion_matrix': [],
        'val_selection_score': [],
        'val_best_score': []
    }

    # Handle checkpoint resuming
    resume_checkpoint = kwargs.get('resume_checkpoint', None)
    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_score = checkpoint.get('best_selection_score', checkpoint.get('val_best_score', best_val_score))

        if 'history' in checkpoint:
            history = checkpoint['history']
            print(f"Loaded training history with {len(history['train_loss'])} epochs")

        print(f"Resuming training from epoch {start_epoch}")
        print(f"Previous best {selection_metric}: {best_val_score:.4f}")

    epoch_range = range(start_epoch, epochs)
    epoch_pbar = tqdm(epoch_range, desc="Training Progress", position=0)

    print(f"Model selection metric: {selection_metric} ({selection_mode})")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\nEpoch {epoch+1}/{epochs} - Training... (LR: {current_lr:.6f})")
        train_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            if use_augmentation:
                batch = _augment_image_batch(batch, rotation_degrees, flip_probability)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = _model_outputs(model, batch, config, device)
                labels = _batch_labels(batch, outputs.shape[0], device, require_labels=True)
                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch + 1}, batch {batch_idx}: {loss.item()}")

            if batch_idx == 0 and epoch == 0:
                if isinstance(batch, dict) and 'image' in batch:
                    print(f"  Image shape: {batch['image'].shape}, Label shape: {labels.shape}")
                if torch.cuda.is_available():
                    print(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.detach(), 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                current_acc = 100. * train_correct / train_total if train_total > 0 else 0
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={current_acc:.2f}%")

            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if train_total == 0:
            raise RuntimeError("No valid training samples were processed in this epoch.")

        train_time = time.time() - train_start_time

        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / max(1, train_total)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        print(f"  Training completed in {train_time:.1f}s - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Initialize validation variables
        val_acc = 0.0
        avg_val_loss = 0.0
        val_auc = 0.0
        val_auc_ci_lower = 0.0
        val_auc_ci_upper = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        val_best_score = 0.0
        cm = np.zeros((2, 2))

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            all_predictions = []
            all_labels = []
            all_probabilities = []

            print(f"  Validation...")
            val_start_time = time.time()

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch is None:
                        continue

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = _model_outputs(model, batch, config, device)
                        labels = _batch_labels(batch, outputs.shape[0], device, require_labels=True)
                        loss = criterion(outputs, labels)

                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError(f"Non-finite validation loss at epoch {epoch + 1}, batch {batch_idx}: {loss.item()}")

                    val_loss += loss.item() * labels.size(0)

                    probabilities = torch.softmax(outputs, dim=1)
                    classification_threshold = kwargs.get("classification_threshold", DEFAULT_CLASSIFICATION_THRESHOLD)
                    predicted = (probabilities[:, 1] >= classification_threshold).long()

                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

            val_time = time.time() - val_start_time

            if val_total > 0:
                val_acc = 100. * val_correct / val_total
                avg_val_loss = val_loss / val_total

                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                all_probabilities = np.array(all_probabilities)

                if len(np.unique(all_labels)) > 1:
                    val_auc, val_auc_ci_lower, val_auc_ci_upper = calculate_auc_with_ci(all_labels, all_probabilities[:, 1])
                else:
                    val_auc, val_auc_ci_lower, val_auc_ci_upper = 0.0, 0.0, 0.0

                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_predictions, average='binary', zero_division=0
                )
                val_precision = precision
                val_recall = recall
                val_f1 = f1

                cm = confusion_matrix(all_labels, all_predictions)

                selection_values = {
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                }
                val_best_score = float(selection_values[selection_metric])

            else:
                raise RuntimeError("No valid validation samples were processed.")

            # Store validation metrics
            for key, value in [
                ('val_loss', avg_val_loss), ('val_acc', val_acc), ('val_auc', val_auc),
                ('val_auc_ci_lower', val_auc_ci_lower), ('val_auc_ci_upper', val_auc_ci_upper),
                ('val_precision', val_precision), ('val_recall', val_recall), ('val_f1', val_f1),
                ('val_confusion_matrix', cm.tolist()), ('val_selection_score', val_best_score),
                ('val_best_score', val_best_score)
            ]:
                if key not in history:
                    history[key] = []
                history[key].append(value)

            print(f"  Validation completed in {val_time:.1f}s")
            print(f"    Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"    AUC: {val_auc:.4f} (95% CI: {val_auc_ci_lower:.4f}-{val_auc_ci_upper:.4f})")
            print(f"    F1: {val_f1:.4f}")
            print(f"    Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            print(f"    Selection Score ({selection_metric}): {val_best_score:.4f}")
            print(f"    Confusion Matrix:\n{cm}")

            if _is_better_score(val_best_score, best_val_score, selection_mode):
                best_val_score = val_best_score
                epochs_without_improvement = 0
                print(f"  New best {selection_metric}: {val_best_score:.4f}")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_auc_ci_lower': val_auc_ci_lower,
                    'val_auc_ci_upper': val_auc_ci_upper,
                    'val_f1': val_f1,
                    'val_best_score': val_best_score,
                    'best_selection_score': best_val_score,
                    'selection_metric': selection_metric,
                    'selection_mode': selection_mode,
                    'confusion_matrix': cm.tolist(),
                    'training_config': {
                        **kwargs,
                        'selection_metric': selection_metric,
                        'selection_mode': selection_mode,
                        'learning_rate': learning_rate,
                        'min_learning_rate': kwargs.get('min_learning_rate', 1e-6),
                        'epochs': epochs,
                        'device': str(device),
                        'weight_decay': weight_decay,
                        'class_weights': class_weights.detach().cpu().tolist(),
                        'early_stopping_patience': early_stopping_patience,
                        'random_seed': random_seed,
                        'use_amp': use_amp,
                        'use_augmentation': use_augmentation,
                        'rotation_degrees': rotation_degrees,
                        'flip_probability': flip_probability,
                    },
                    'history': history
                }, os.path.join(save_dir, 'best_model.pth'))
            else:
                epochs_without_improvement += 1

        epoch_time = time.time() - epoch_start_time
        if scheduler is not None:
            scheduler.step()

        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.1f}%',
            'Val_Acc': f'{val_acc:.1f}%',
            'AUC': f'{val_auc:.3f}',
            'AUC_CI': f'{val_auc_ci_lower:.3f}-{val_auc_ci_upper:.3f}',
            'F1': f'{val_f1:.3f}',
            selection_metric: f'{val_best_score:.3f}',
            'Best': f'{best_val_score:.3f}',
            'LR': f'{current_lr:.1e}',
            'Time': f'{epoch_time:.1f}s'
        })

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train_Acc={train_acc:.2f}%, Val_Acc={val_acc:.2f}%")
        print(f"  AUC={val_auc:.4f} (95% CI: {val_auc_ci_lower:.4f}-{val_auc_ci_upper:.4f})")
        print(f"  F1={val_f1:.4f}, {selection_metric}={val_best_score:.4f}, Best={best_val_score:.4f}")
        print(f"  Time={epoch_time:.1f}s\n")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs without {selection_metric} improvement for {early_stopping_patience} epochs.")
            break

    return history


def train_single_fold(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      fold: int,
                      epochs: int = DEFAULT_EPOCHS,
                      learning_rate: float = DEFAULT_LEARNING_RATE,
                      device: str = 'cuda',
                      save_dir: str = './checkpoints',
                      config: Optional[Dict[str, Any]] = None,
                      **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model for a single fold and return the trained model and history
    """
    fold_save_dir = os.path.join(save_dir, f'fold_{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*60}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=fold_save_dir,
        config=config,
        **kwargs
    )

    return model, history


def train_kfold_cv(model_class,
                   model_config: Dict[str, Any],
                   dataset_items: List[Dict],
                   n_folds: int = 5,
                   epochs: int = DEFAULT_EPOCHS,
                   learning_rate: float = DEFAULT_LEARNING_RATE,
                   device: str = 'cuda',
                   save_dir: str = './checkpoints',
                   batch_size: int = 4,
                   num_workers: int = 2,
                   config: Optional[Dict[str, Any]] = None,
                   data_dir: str = None,
                   feature_stats: Dict = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Train model using K-Fold Cross Validation

    Args:
        model_class: Class of the model to instantiate for each fold
        model_config: Configuration dictionary for model initialization
        dataset_items: List of data items (dictionaries with 'filename', 'label', etc.)
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs per fold
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        config: Additional configuration
        data_dir: Directory containing preprocessed tensor files
        feature_stats: Deprecated for k-fold training; fold-specific statistics are
            computed from each training fold to avoid validation leakage.

    Returns:
        Dictionary containing cross-validation results
    """
    from src.data.loaders import PreprocessedDataset, collate_fn

    print(f"\n{'='*80}")
    print(f"Starting {n_folds}-Fold Cross Validation Training")
    print(f"Total samples: {len(dataset_items)}")
    print(f"{'='*80}")

    requested_selection_metric = kwargs.get('selection_metric')
    if requested_selection_metric and requested_selection_metric != DEFAULT_SELECTION_METRIC:
        print(f"Ignoring selection_metric='{requested_selection_metric}'; k-fold model selection is fixed to {DEFAULT_SELECTION_METRIC}.")
    selection_metric = DEFAULT_SELECTION_METRIC
    selection_mode = _selection_mode(selection_metric)

    if feature_stats is not None:
        print("Ignoring global feature_stats for k-fold training; computing fold-specific statistics.")
    configured_class_weights = kwargs.get('class_weights', DEFAULT_CLASS_WEIGHTS)

    labels = np.array([_extract_label(item) for item in dataset_items])
    indices = np.arange(len(dataset_items))

    random_seed = int(kwargs.get('random_seed', DEFAULT_RANDOM_SEED))
    _set_random_seed(random_seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    cv_results = {
        'n_folds': n_folds,
        'fold_histories': [],
        'fold_metrics': [],
        'selection_metric': selection_metric,
        'selection_mode': selection_mode,
        'mean_metrics': {},
        'std_metrics': {}
    }

    all_val_aucs = []
    all_val_accs = []
    all_val_f1s = []
    all_val_precisions = []
    all_val_recalls = []

    cv_save_dir = os.path.join(save_dir, 'kfold_cv')
    os.makedirs(cv_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        fold_seed = random_seed + fold
        _set_random_seed(fold_seed)
        train_generator = torch.Generator()
        train_generator.manual_seed(fold_seed)

        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        print(f"{'='*60}")

        train_items = [dataset_items[i] for i in train_idx]
        val_items = [dataset_items[i] for i in val_idx]
        fold_feature_stats = compute_feature_statistics(train_items)
        fold_kwargs = {
            **kwargs,
            'feature_stats': fold_feature_stats,
            'class_weights': configured_class_weights,
            'selection_metric': selection_metric,
            'random_seed': fold_seed,
        }

        train_dataset = PreprocessedDataset(
            data_items=train_items,
            data_dir=data_dir,
            feature_stats=fold_feature_stats
        )

        val_dataset = PreprocessedDataset(
            data_items=val_items,
            data_dir=data_dir,
            feature_stats=fold_feature_stats
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
            generator=train_generator
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn
        )

        model = model_class(**model_config)

        fold_save_dir = os.path.join(cv_save_dir, f'fold_{fold}')

        model, history = train_single_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold=fold,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            save_dir=fold_save_dir,
            config=config,
            **fold_kwargs
        )

        # Select the fold model by the pre-specified metric only.
        best_epoch_idx = _best_epoch_index(history, selection_metric, selection_mode)
        fold_selection_score = history[selection_metric][best_epoch_idx] if history.get(selection_metric) else 0.0
        fold_best_auc = history['val_auc'][best_epoch_idx] if history['val_auc'] else 0.0
        fold_best_acc = history['val_acc'][best_epoch_idx] if history['val_acc'] else 0.0
        fold_best_f1 = history['val_f1'][best_epoch_idx] if history['val_f1'] else 0.0
        fold_best_precision = history['val_precision'][best_epoch_idx] if history['val_precision'] else 0.0
        fold_best_recall = history['val_recall'][best_epoch_idx] if history['val_recall'] else 0.0

        fold_metrics = {
            'fold': fold,
            'best_epoch': best_epoch_idx,
            'selection_metric': selection_metric,
            'selection_score': float(fold_selection_score),
            'val_auc': float(fold_best_auc),
            'val_acc': float(fold_best_acc),
            'val_f1': float(fold_best_f1),
            'val_precision': float(fold_best_precision),
            'val_recall': float(fold_best_recall),
            'val_auc_ci_lower': float(history['val_auc_ci_lower'][best_epoch_idx]) if history.get('val_auc_ci_lower') else 0.0,
            'val_auc_ci_upper': float(history['val_auc_ci_upper'][best_epoch_idx]) if history.get('val_auc_ci_upper') else 0.0,
            'feature_stats': fold_feature_stats,
            'class_weights': configured_class_weights
        }

        cv_results['fold_histories'].append(history)
        cv_results['fold_metrics'].append(fold_metrics)

        all_val_aucs.append(fold_best_auc)
        all_val_accs.append(fold_best_acc)
        all_val_f1s.append(fold_best_f1)
        all_val_precisions.append(fold_best_precision)
        all_val_recalls.append(fold_best_recall)

        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Epoch: {best_epoch_idx + 1}")
        print(f"  Selection ({selection_metric}): {fold_selection_score:.4f}")
        print(f"  AUC: {fold_best_auc:.4f}")
        print(f"  Accuracy: {fold_best_acc:.2f}%")
        print(f"  F1: {fold_best_f1:.4f}")
        print(f"  Precision: {fold_best_precision:.4f}, Recall: {fold_best_recall:.4f}")

        del model, train_dataset, val_dataset, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate cross-validation statistics
    cv_results['mean_metrics'] = {
        'val_auc': np.mean(all_val_aucs),
        'val_acc': np.mean(all_val_accs),
        'val_f1': np.mean(all_val_f1s),
        'val_precision': np.mean(all_val_precisions),
        'val_recall': np.mean(all_val_recalls)
    }

    cv_results['std_metrics'] = {
        'val_auc': np.std(all_val_aucs),
        'val_acc': np.std(all_val_accs),
        'val_f1': np.std(all_val_f1s),
        'val_precision': np.std(all_val_precisions),
        'val_recall': np.std(all_val_recalls)
    }

    print(f"\n{'='*80}")
    print(f"{n_folds}-Fold Cross Validation Results Summary")
    print(f"{'='*80}")
    print(f"\nMean Metrics (+/- Std):")
    print(f"  AUC:       {cv_results['mean_metrics']['val_auc']:.4f} +/- {cv_results['std_metrics']['val_auc']:.4f}")
    print(f"  Accuracy:  {cv_results['mean_metrics']['val_acc']:.2f}% +/- {cv_results['std_metrics']['val_acc']:.2f}%")
    print(f"  F1:        {cv_results['mean_metrics']['val_f1']:.4f} +/- {cv_results['std_metrics']['val_f1']:.4f}")
    print(f"  Precision: {cv_results['mean_metrics']['val_precision']:.4f} +/- {cv_results['std_metrics']['val_precision']:.4f}")
    print(f"  Recall:    {cv_results['mean_metrics']['val_recall']:.4f} +/- {cv_results['std_metrics']['val_recall']:.4f}")

    # Save cross-validation results
    cv_results_path = os.path.join(cv_save_dir, 'cv_results.json')
    with open(cv_results_path, 'w') as f:
        serializable_results = {
            'n_folds': cv_results['n_folds'],
            'selection_metric': cv_results['selection_metric'],
            'selection_mode': cv_results['selection_mode'],
            'mean_metrics': {k: float(v) for k, v in cv_results['mean_metrics'].items()},
            'std_metrics': {k: float(v) for k, v in cv_results['std_metrics'].items()},
            'fold_metrics': cv_results['fold_metrics']
        }
        json.dump(serializable_results, f, indent=2)
    print(f"\nCV results saved to: {cv_results_path}")

    return cv_results


def ensemble_evaluate(model_class,
                      model_config: Dict[str, Any],
                      test_loader: DataLoader,
                      checkpoint_dir: str,
                      n_folds: int = 5,
                      device: str = 'cuda',
                      config: Optional[Dict[str, Any]] = None,
                      **kwargs) -> Dict[str, Any]:
    """
    Evaluate test set using ensemble of models from K-Fold CV

    Args:
        model_class: Class of the model to instantiate
        model_config: Configuration dictionary for model initialization
        test_loader: DataLoader for test data
        checkpoint_dir: Directory containing fold checkpoints
        n_folds: Number of folds (models) to ensemble
        device: Device for inference
        config: Additional configuration

    Returns:
        Dictionary containing ensemble evaluation results
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"Ensemble Evaluation with {n_folds} Models")
    print("Ensemble Method: equal-weight mean probability")
    print(f"{'='*80}")

    fold_metadata = {}
    cv_results_path = os.path.join(checkpoint_dir, 'cv_results.json')
    if os.path.exists(cv_results_path):
        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)
            for fold_metric in cv_results.get('fold_metrics', []):
                fold_metadata[int(fold_metric.get('fold', len(fold_metadata)))] = fold_metric

    # Load all fold models
    models = []
    model_feature_stats = []
    model_fold_ids = []
    for fold in range(n_folds):
        fold_checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold}', 'best_model.pth')

        if not os.path.exists(fold_checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold} not found at {fold_checkpoint_path}")
            continue

        model = model_class(**model_config)
        checkpoint = torch.load(fold_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)
        model_fold_ids.append(fold)
        checkpoint_config = checkpoint.get('training_config', {})
        fold_feature_stats = (
            checkpoint_config.get('feature_stats')
            or checkpoint.get('feature_stats')
            or fold_metadata.get(fold, {}).get('feature_stats')
        )
        model_feature_stats.append(fold_feature_stats)
        print(f"Loaded model from fold {fold}")

    if len(models) == 0:
        raise ValueError("No models loaded for ensemble evaluation")

    print(f"\nLoaded {len(models)} models for ensemble")

    all_labels = []
    all_ensemble_probs = []
    all_ensemble_preds = []
    all_individual_preds = [[] for _ in range(len(models))]
    all_individual_probs = [[] for _ in range(len(models))]

    classification_threshold = kwargs.get('classification_threshold', DEFAULT_CLASSIFICATION_THRESHOLD)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Ensemble Evaluation")):
            if batch is None:
                continue

            batch_probs = []
            batch_preds = []
            labels = None

            for model_idx, model in enumerate(models):
                model_batch = _normalize_batch_features(batch, model_feature_stats[model_idx])
                outputs = _model_outputs(model, model_batch, config, device)
                labels = _batch_labels(model_batch, outputs.shape[0], device, require_labels=True)

                probs = torch.softmax(outputs, dim=1)
                preds = (probs[:, 1] >= classification_threshold).long()

                batch_probs.append(probs[:, 1].cpu().numpy())
                batch_preds.append(preds.cpu().numpy())

                all_individual_probs[model_idx].extend(probs[:, 1].cpu().numpy())
                all_individual_preds[model_idx].extend(preds.cpu().numpy())

            batch_probs = np.array(batch_probs)
            batch_preds = np.array(batch_preds)

            ensemble_prob = np.mean(batch_probs, axis=0)
            ensemble_pred = (ensemble_prob >= classification_threshold).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_ensemble_probs.extend(ensemble_prob)
            all_ensemble_preds.extend(ensemble_pred)

    if len(all_labels) == 0:
        raise RuntimeError("No valid test samples were processed during ensemble evaluation.")

    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)
    all_ensemble_preds = np.array(all_ensemble_preds)

    results = {
        'ensemble_method': 'equal_weight_mean_probability',
        'weight_source': 'equal_weight',
        'fold_ids': model_fold_ids,
        'n_models': len(models),
        'n_samples': len(all_labels)
    }

    if len(np.unique(all_labels)) > 1:
        ensemble_auc, ensemble_auc_ci_lower, ensemble_auc_ci_upper = calculate_auc_with_ci(
            all_labels, all_ensemble_probs
        )
    else:
        ensemble_auc = ensemble_auc_ci_lower = ensemble_auc_ci_upper = 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_ensemble_preds, average='binary', zero_division=0
    )

    cm = confusion_matrix(all_labels, all_ensemble_preds)

    ensemble_acc = 100.0 * np.mean(all_labels == all_ensemble_preds)

    results['ensemble_metrics'] = {
        'auc': float(ensemble_auc),
        'auc_ci_lower': float(ensemble_auc_ci_lower),
        'auc_ci_upper': float(ensemble_auc_ci_upper),
        'accuracy': float(ensemble_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

    individual_metrics = []
    for model_idx in range(len(models)):
        model_preds = np.array(all_individual_preds[model_idx])
        model_probs = np.array(all_individual_probs[model_idx])

        if len(np.unique(all_labels)) > 1:
            model_auc = roc_auc_score(all_labels, model_probs)
        else:
            model_auc = 0.0

        model_acc = 100.0 * np.mean(all_labels == model_preds)
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
            all_labels, model_preds, average='binary', zero_division=0
        )

        individual_metrics.append({
            'fold': model_fold_ids[model_idx],
            'auc': float(model_auc),
            'accuracy': float(model_acc),
            'precision': float(model_precision),
            'recall': float(model_recall),
            'f1': float(model_f1)
        })

    results['individual_metrics'] = individual_metrics

    print(f"\n{'='*60}")
    print("Ensemble Evaluation Results")
    print(f"{'='*60}")
    print("\nEnsemble (equal-weight mean probability):")
    print(f"  AUC:       {ensemble_auc:.4f} (95% CI: {ensemble_auc_ci_lower:.4f}-{ensemble_auc_ci_upper:.4f})")
    print(f"  Accuracy:  {ensemble_acc:.2f}%")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    print(f"\nIndividual Model Performance:")
    for metrics in individual_metrics:
        print(f"  Fold {metrics['fold']}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.4f}")

    # Save results
    results_path = os.path.join(checkpoint_dir, 'ensemble_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    for model in models:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_kfold_cv_and_ensemble_test(
    model_class,
    model_config: Dict[str, Any],
    train_dataset_items: List[Dict],
    test_loader: DataLoader,
    n_folds: int = 5,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    batch_size: int = 4,
    num_workers: int = 2,
    config: Optional[Dict[str, Any]] = None,
    data_dir: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete pipeline: Run K-Fold CV training and then ensemble evaluation on test set
    """
    print(f"\n{'#'*80}")
    print("Starting Complete K-Fold CV + Ensemble Pipeline")
    print(f"{'#'*80}")

    # Step 1: K-Fold Cross Validation Training
    cv_results = train_kfold_cv(
        model_class=model_class,
        model_config=model_config,
        dataset_items=train_dataset_items,
        n_folds=n_folds,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        config=config,
        data_dir=data_dir,
        **kwargs
    )

    # Step 2: Ensemble Evaluation on Test Set
    cv_save_dir = os.path.join(save_dir, 'kfold_cv')

    ensemble_results = ensemble_evaluate(
        model_class=model_class,
        model_config=model_config,
        test_loader=test_loader,
        checkpoint_dir=cv_save_dir,
        n_folds=n_folds,
        device=device,
        config=config,
        **kwargs
    )

    final_results = {
        'cv_results': cv_results,
        'ensemble_results': ensemble_results
    }

    final_results_path = os.path.join(save_dir, 'kfold_ensemble_final_results.json')

    serializable_final = {
        'cv_summary': {
            'n_folds': cv_results['n_folds'],
            'selection_metric': cv_results['selection_metric'],
            'selection_mode': cv_results['selection_mode'],
            'mean_metrics': cv_results['mean_metrics'],
            'std_metrics': cv_results['std_metrics']
        },
        'ensemble_summary': ensemble_results
    }

    with open(final_results_path, 'w') as f:
        json.dump(serializable_final, f, indent=2)

    print(f"\n{'#'*80}")
    print("Pipeline Complete!")
    print(f"{'#'*80}")
    print(f"Final results saved to: {final_results_path}")

    return final_results
