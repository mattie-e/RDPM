import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import time
import copy
import json
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.utils import resample

def calculate_auc_with_ci(y_true, y_pred_proba, confidence_level=0.95, n_bootstrap=1000):
    """
    Calculate AUC with 95% confidence interval using bootstrap method
    """
    # Convert to numpy arrays if they aren't already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred_proba, np.ndarray):
        y_pred_proba = np.array(y_pred_proba)
    
    try:
        original_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        return 0.0, 0.0, 0.0
    
    # Bootstrap sampling for confidence interval
    bootstrap_aucs = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample - use numpy indexing
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_bootstrap = y_true[bootstrap_indices]
        y_pred_bootstrap = y_pred_proba[bootstrap_indices]
        
        # Calculate AUC for bootstrap sample
        try:
            if len(np.unique(y_bootstrap)) > 1:  # Check if both classes present
                bootstrap_auc = roc_auc_score(y_bootstrap, y_pred_bootstrap)
                bootstrap_aucs.append(bootstrap_auc)
        except ValueError:
            continue
    
    if len(bootstrap_aucs) == 0:
        return original_auc, original_auc, original_auc
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_aucs, (1 - alpha/2) * 100)
    
    return original_auc, ci_lower, ci_upper

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                epochs: int = 100,
                learning_rate: float = 1e-3,
                device: str = 'cuda',
                save_dir: str = './checkpoints',
                config: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
    """
    Train a model with the given data loaders
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints
    
    Returns:
        Dictionary containing training history
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and loss function with minimal weight decay
    weight_decay = kwargs.get('weight_decay', 1e-5)  # Reduced regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    
    # Use weighted CrossEntropyLoss with class weights
    class_weights = torch.tensor(kwargs['class_weights']).to(device) if kwargs['class_weights'] else torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # No learning rate scheduler - let learning rate stay constant
    scheduler = None
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history with additional metrics
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
        'val_best_score': []
    }
    
    best_val_score = 0.0
    start_epoch = 0
    
    # Handle checkpoint resuming
    resume_checkpoint = kwargs.get('resume_checkpoint', None)
    if resume_checkpoint:
        try:
            print(f"Loading checkpoint from {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_score = checkpoint.get('val_best_score', 0.0)
            
            # Load history if available
            if 'history' in checkpoint:
                history = checkpoint['history']
                print(f"Loaded training history with {len(history['train_loss'])} epochs")
            
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Previous best validation score: {best_val_score:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
            best_val_score = 0.0
    
    # Adjust epoch range for resuming
    epoch_range = range(start_epoch, epochs)
    epoch_pbar = tqdm(epoch_range, desc="Training Progress", position=0)
    
    # Print GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training phase with dropout enabled
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs} - Training... (LR: {current_lr:.6f})")
        train_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Handle MONAI dictionary format correctly
                if isinstance(batch, dict):
                    if 'image' not in batch:
                        continue
                    
                    # Handle different model types
                    model_name = config.get('model', {}).get('name', 'resnet')
                    
                    if model_name == 'hybrid':
                        # For hybrid model, pass the whole batch
                        required_features = ['DM_normalized', 'maxdiameter_normalized', 'HTN_normalized', 'age_normalized', 'eGFR_normalized']
                        if all(feat in batch for feat in required_features):
                            outputs = model(batch)
                        else:
                            # Fall back to image-only if numerical features missing
                            outputs = model({'image': batch['image']})
                    elif model_name in ['masked_resnet', 'masked_resnet_attention']:
                        # For masked ResNet models, pass batch (will handle masking internally)
                        outputs = model(batch)
                    else:
                        # For regular ResNet models, just pass the image
                        images = batch['image'].to(device, non_blocking=True)
                        outputs = model(images)
                    
                    if 'label' in batch:
                        labels = batch['label'].to(device, non_blocking=True)
                    else:
                        labels = torch.zeros(batch['image'].size(0), dtype=torch.long).to(device)
                    
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long()
                    
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    if isinstance(batch[0], dict):
                        images = batch[0]['image'].to(device)
                        labels = batch[0].get('label', torch.zeros(images.size(0))).to(device)
                    else:
                        images, labels = batch[0].to(device), batch[1].to(device)
                    
                    outputs = model(images)
                else:
                    continue
                
                # Debug info for first batch only
                if batch_idx == 0 and epoch == 0:
                    if 'image' in batch:
                        print(f"  Image shape: {batch['image'].shape}, Label shape: {labels.shape}")
                    if torch.cuda.is_available():
                        print(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                if batch['image'].size(0) == 0:
                    continue
                
                optimizer.zero_grad()
                
                # Skip explicit forward pass since we already have outputs
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    current_acc = 100. * train_correct / train_total if train_total > 0 else 0
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={current_acc:.2f}%")
                
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error in training batch {batch_idx}: {e}")
                continue
        
        train_time = time.time() - train_start_time
        
        # Calculate training metrics
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"  Training completed in {train_time:.1f}s - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Initialize validation phase variables - always initialize these
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

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Store predictions and labels for metric calculation
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            print(f"  Validation...")
            val_start_time = time.time()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        # Handle MONAI dictionary format correctly
                        if isinstance(batch, dict):
                            if 'image' not in batch:
                                continue
                            
                            # Handle different model types
                            model_name = config.get('model', {}).get('name', 'resnet')
                            
                            if model_name == 'hybrid':
                                # For hybrid model, pass the whole batch
                                required_features = ['DM_normalized', 'maxdiameter_normalized', 'HTN_normalized', 'age_normalized', 'eGFR_normalized']
                                if all(feat in batch for feat in required_features):
                                    outputs = model(batch)
                                else:
                                    # Fall back to image-only if numerical features missing
                                    outputs = model({'image': batch['image']})
                            elif model_name in ['masked_resnet', 'masked_resnet_attention']:
                                # For masked ResNet models, pass batch (will handle masking internally)
                                outputs = model(batch)
                            else:
                                # For regular ResNet models, just pass the image
                                images = batch['image'].to(device, non_blocking=True)
                                outputs = model(images)
                            
                            if 'label' in batch:
                                labels = batch['label'].to(device, non_blocking=True)
                            else:
                                labels = torch.zeros(batch['image'].size(0), dtype=torch.long).to(device)
                            
                            if labels.dim() > 1:
                                labels = labels.squeeze()
                            labels = labels.long()
                            
                        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            if isinstance(batch[0], dict):
                                images = batch[0]['image'].to(device)
                                labels = batch[0].get('label', torch.zeros(images.size(0))).to(device)
                            else:
                                images, labels = batch[0].to(device), batch[1].to(device)
                            
                            outputs = model(images)
                        else:
                            continue
                        
                        if batch['image'].size(0) == 0:
                            continue
                        
                        loss = criterion(outputs, labels)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        val_loss += loss.item()
                        
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted = (probabilities[:, 1] > kwargs["classification_threshold"]).long()
                        
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())
                        
                    except Exception as e:
                        continue
            
            val_time = time.time() - val_start_time
            
            if val_total > 0:
                val_acc = 100. * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                all_probabilities = np.array(all_probabilities)
                
                try:
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
                    
                    val_best_score = val_f1
                    
                except Exception as e:
                    print(f"  Warning: Error calculating advanced metrics: {e}")
                    val_auc = val_auc_ci_lower = val_auc_ci_upper = val_precision = val_recall = val_f1 = val_best_score = 0.0
                    cm = np.zeros((2, 2))
                
            else:
                val_acc = val_auc = val_auc_ci_lower = val_auc_ci_upper = val_precision = val_recall = val_f1 = val_best_score = 0.0
                avg_val_loss = 0.0
                cm = np.zeros((2, 2))
            
            # Store all metrics
            # Add safety checks for all metrics keys
            if 'val_loss' not in history:
                history['val_loss'] = []
            history['val_loss'].append(avg_val_loss)

            if 'val_acc' not in history:
                history['val_acc'] = []
            history['val_acc'].append(val_acc)

            if 'val_auc' not in history:
                history['val_auc'] = []
            history['val_auc'].append(val_auc)

            if 'val_auc_ci_lower' not in history:
                history['val_auc_ci_lower'] = []
            history['val_auc_ci_lower'].append(val_auc_ci_lower)

            if 'val_auc_ci_upper' not in history:
                history['val_auc_ci_upper'] = []
            history['val_auc_ci_upper'].append(val_auc_ci_upper)

            if 'val_precision' not in history:
                history['val_precision'] = []
            history['val_precision'].append(val_precision)

            if 'val_recall' not in history:
                history['val_recall'] = []
            history['val_recall'].append(val_recall)

            if 'val_f1' not in history:
                history['val_f1'] = []
            history['val_f1'].append(val_f1)

            if 'val_confusion_matrix' not in history:
                history['val_confusion_matrix'] = []
            history['val_confusion_matrix'].append(cm.tolist())

            if 'val_best_score' not in history:
                history['val_best_score'] = []
            history['val_best_score'].append(val_best_score)
            
            print(f"  Validation completed in {val_time:.1f}s")
            print(f"    Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"    AUC: {val_auc:.4f} (95% CI: {val_auc_ci_lower:.4f}-{val_auc_ci_upper:.4f})")
            print(f"    F1: {val_f1:.4f}")
            print(f"    Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            print(f"    Best Score (F1+AUC)/2: {val_best_score:.4f}")
            print(f"    Confusion Matrix:\n{cm}")
            
            # Remove early stopping check - just save best model
            if val_best_score > best_val_score:
                best_val_score = val_best_score
                print(f"  New best validation score: {val_best_score:.4f}")
                
                # Save best model with complete state including full config
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
                    'confusion_matrix': cm.tolist(),
                    'training_config': {
                        **kwargs,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'device': str(device)
                    },
                    'history': history
                }, os.path.join(save_dir, 'best_model.pth'))
            
            # Save the latest model checkpoint with complete config
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
                    'confusion_matrix': cm.tolist(),
                    'training_config': {
                        **kwargs,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'device': str(device)
                    },
                    'history': history
                }, os.path.join(save_dir, 'latest.pth'))
        
        # Always append validation metrics to history (whether validation was run or not)
        # Check if keys exist in history, if not add them first
        if 'val_loss' not in history:
            history['val_loss'] = []
        history['val_loss'].append(avg_val_loss)

        if 'val_acc' not in history:
            history['val_acc'] = []
        history['val_acc'].append(val_acc)

        if 'val_auc' not in history:
            history['val_auc'] = []
        history['val_auc'].append(val_auc)

        if 'val_auc_ci_lower' not in history:
            history['val_auc_ci_lower'] = []
        history['val_auc_ci_lower'].append(val_auc_ci_lower)

        if 'val_auc_ci_upper' not in history:
            history['val_auc_ci_upper'] = []
        history['val_auc_ci_upper'].append(val_auc_ci_upper)

        if 'val_precision' not in history:
            history['val_precision'] = []
        history['val_precision'].append(val_precision)

        if 'val_recall' not in history:
            history['val_recall'] = []
        history['val_recall'].append(val_recall)

        if 'val_f1' not in history:
            history['val_f1'] = []
        history['val_f1'].append(val_f1)

        if 'val_confusion_matrix' not in history:
            history['val_confusion_matrix'] = []
        history['val_confusion_matrix'].append(cm.tolist())

        if 'val_best_score' not in history:
            history['val_best_score'] = []
        history['val_best_score'].append(val_best_score)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'val_best_score': val_best_score,
                'confusion_matrix': cm.tolist() if 'cm' in locals() else [[0, 0], [0, 0]],
                'training_config': kwargs,
                'history': history
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start_time
        
        # Update main epoch progress bar without early stopping counter
        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.1f}%',
            'Val_Acc': f'{val_acc:.1f}%',
            'AUC': f'{val_auc:.3f}',
            'AUC_CI': f'{val_auc_ci_lower:.3f}-{val_auc_ci_upper:.3f}',
            'F1': f'{val_f1:.3f}',
            'Best': f'{best_val_score:.3f}',
            'LR': f'{current_lr:.1e}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # Print comprehensive epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train_Acc={train_acc:.2f}%, Val_Acc={val_acc:.2f}%")
        print(f"  AUC={val_auc:.4f} (95% CI: {val_auc_ci_lower:.4f}-{val_auc_ci_upper:.4f})")
        print(f"  F1={val_f1:.4f}, Best_Score={val_best_score:.4f}")
        print(f"  Time={epoch_time:.1f}s\n")
    
    # Save final checkpoint with complete config
    final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'val_acc': val_acc,
        'val_auc': val_auc,
        'val_auc_ci_lower': val_auc_ci_lower if 'val_auc_ci_lower' in locals() else 0.0,
        'val_auc_ci_upper': val_auc_ci_upper if 'val_auc_ci_upper' in locals() else 0.0,
        'val_f1': val_f1,
        'val_best_score': best_val_score,
        'confusion_matrix': cm.tolist() if 'cm' in locals() else [[0, 0], [0, 0]],
        'training_config': {
            **kwargs,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'device': str(device)
        },
        'history': history
    }, final_checkpoint_path)
    
    return history


def train_single_fold(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      fold: int,
                      epochs: int = 100,
                      learning_rate: float = 1e-3,
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
    
    # Train the model
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
                   epochs: int = 100,
                   learning_rate: float = 1e-3,
                   device: str = 'cuda',
                   save_dir: str = './checkpoints',
                   batch_size: int = 4,
                   num_workers: int = 2,
                   transform=None,
                   config: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Train model using K-Fold Cross Validation
    
    Args:
        model_class: Class of the model to instantiate for each fold
        model_config: Configuration dictionary for model initialization
        dataset_items: List of data items (dictionaries with 'image', 'label', etc.)
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs per fold
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        transform: Transform to apply to data
        config: Additional configuration
    
    Returns:
        Dictionary containing cross-validation results
    """
    from monai.data import CacheDataset, list_data_collate
    
    print(f"\n{'='*80}")
    print(f"Starting {n_folds}-Fold Cross Validation Training")
    print(f"Total samples: {len(dataset_items)}")
    print(f"{'='*80}")
    
    # Extract labels for stratified splitting
    labels = []
    for item in dataset_items:
        if 'label' in item:
            label = item['label']
            if torch.is_tensor(label):
                labels.append(label.item())
            else:
                labels.append(int(label))
        else:
            labels.append(0)
    
    labels = np.array(labels)
    indices = np.arange(len(dataset_items))
    
    # Create stratified K-Fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results for all folds
    cv_results = {
        'n_folds': n_folds,
        'fold_histories': [],
        'fold_metrics': [],
        'best_fold': None,
        'best_val_auc': 0.0,
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
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        print(f"{'='*60}")
        
        # Create fold-specific datasets
        train_items = [dataset_items[i] for i in train_idx]
        val_items = [dataset_items[i] for i in val_idx]
        
        # Create CacheDatasets
        train_dataset = CacheDataset(
            data=train_items,
            transform=transform,
            cache_rate=0.5,
            num_workers=max(1, num_workers)
        )
        
        val_dataset = CacheDataset(
            data=val_items,
            transform=transform,
            cache_rate=0.5,
            num_workers=max(1, num_workers)
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=list_data_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=list_data_collate
        )
        
        # Initialize a fresh model for this fold
        model = model_class(**model_config)
        
        fold_save_dir = os.path.join(cv_save_dir, f'fold_{fold}')
        
        # Train model for this fold
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
            **kwargs
        )
        
        # Get best metrics from this fold
        best_epoch_idx = np.argmax(history.get('val_auc', [0]))
        fold_best_auc = history['val_auc'][best_epoch_idx] if history['val_auc'] else 0.0
        fold_best_acc = history['val_acc'][best_epoch_idx] if history['val_acc'] else 0.0
        fold_best_f1 = history['val_f1'][best_epoch_idx] if history['val_f1'] else 0.0
        fold_best_precision = history['val_precision'][best_epoch_idx] if history['val_precision'] else 0.0
        fold_best_recall = history['val_recall'][best_epoch_idx] if history['val_recall'] else 0.0
        
        fold_metrics = {
            'fold': fold,
            'best_epoch': best_epoch_idx,
            'val_auc': fold_best_auc,
            'val_acc': fold_best_acc,
            'val_f1': fold_best_f1,
            'val_precision': fold_best_precision,
            'val_recall': fold_best_recall,
            'val_auc_ci_lower': history['val_auc_ci_lower'][best_epoch_idx] if history.get('val_auc_ci_lower') else 0.0,
            'val_auc_ci_upper': history['val_auc_ci_upper'][best_epoch_idx] if history.get('val_auc_ci_upper') else 0.0
        }
        
        cv_results['fold_histories'].append(history)
        cv_results['fold_metrics'].append(fold_metrics)
        
        all_val_aucs.append(fold_best_auc)
        all_val_accs.append(fold_best_acc)
        all_val_f1s.append(fold_best_f1)
        all_val_precisions.append(fold_best_precision)
        all_val_recalls.append(fold_best_recall)
        
        # Track best fold
        if fold_best_auc > cv_results['best_val_auc']:
            cv_results['best_val_auc'] = fold_best_auc
            cv_results['best_fold'] = fold
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Epoch: {best_epoch_idx + 1}")
        print(f"  AUC: {fold_best_auc:.4f}")
        print(f"  Accuracy: {fold_best_acc:.2f}%")
        print(f"  F1: {fold_best_f1:.4f}")
        print(f"  Precision: {fold_best_precision:.4f}, Recall: {fold_best_recall:.4f}")
        
        # Clean up GPU memory
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
    
    # Print cross-validation summary
    print(f"\n{'='*80}")
    print(f"{n_folds}-Fold Cross Validation Results Summary")
    print(f"{'='*80}")
    print(f"Best Fold: {cv_results['best_fold'] + 1} (AUC: {cv_results['best_val_auc']:.4f})")
    print(f"\nMean Metrics (± Std):")
    print(f"  AUC:       {cv_results['mean_metrics']['val_auc']:.4f} ± {cv_results['std_metrics']['val_auc']:.4f}")
    print(f"  Accuracy:  {cv_results['mean_metrics']['val_acc']:.2f}% ± {cv_results['std_metrics']['val_acc']:.2f}%")
    print(f"  F1:        {cv_results['mean_metrics']['val_f1']:.4f} ± {cv_results['std_metrics']['val_f1']:.4f}")
    print(f"  Precision: {cv_results['mean_metrics']['val_precision']:.4f} ± {cv_results['std_metrics']['val_precision']:.4f}")
    print(f"  Recall:    {cv_results['mean_metrics']['val_recall']:.4f} ± {cv_results['std_metrics']['val_recall']:.4f}")
    
    # Save cross-validation results
    cv_results_path = os.path.join(cv_save_dir, 'cv_results.json')
    with open(cv_results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'n_folds': cv_results['n_folds'],
            'best_fold': cv_results['best_fold'],
            'best_val_auc': float(cv_results['best_val_auc']),
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
                      ensemble_method: str = 'soft_voting',
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
        ensemble_method: Method for combining predictions
            - 'soft_voting': Average probabilities, then predict
            - 'hard_voting': Majority vote of individual predictions
            - 'weighted_voting': Weight by validation AUC
        config: Additional configuration
    
    Returns:
        Dictionary containing ensemble evaluation results
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"Ensemble Evaluation with {n_folds} Models")
    print(f"Ensemble Method: {ensemble_method}")
    print(f"{'='*80}")
    
    # Load validation AUCs for weighted voting
    fold_weights = []
    cv_results_path = os.path.join(checkpoint_dir, 'cv_results.json')
    if os.path.exists(cv_results_path):
        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)
            for fold_metric in cv_results.get('fold_metrics', []):
                fold_weights.append(fold_metric.get('val_auc', 1.0))
    else:
        fold_weights = [1.0] * n_folds
    
    # Normalize weights
    fold_weights = np.array(fold_weights)
    fold_weights = fold_weights / fold_weights.sum()
    print(f"Fold weights: {fold_weights}")
    
    # Load all fold models
    models = []
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
        print(f"Loaded model from fold {fold}")
    
    if len(models) == 0:
        raise ValueError("No models loaded for ensemble evaluation")
    
    print(f"\nLoaded {len(models)} models for ensemble")
    
    # Collect predictions from all models
    all_labels = []
    all_ensemble_probs = []
    all_ensemble_preds = []
    all_individual_preds = [[] for _ in range(len(models))]
    all_individual_probs = [[] for _ in range(len(models))]
    
    classification_threshold = kwargs.get('classification_threshold', 0.5)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Ensemble Evaluation")):
            try:
                if isinstance(batch, dict):
                    if 'image' not in batch:
                        continue
                    
                    if 'label' in batch:
                        labels = batch['label'].to(device)
                    else:
                        labels = torch.zeros(batch['image'].size(0), dtype=torch.long).to(device)
                    
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long()
                    
                    # Get predictions from each model
                    batch_probs = []
                    batch_preds = []
                    
                    for model_idx, model in enumerate(models):
                        model_name = config.get('model', {}).get('name', 'resnet') if config else 'resnet'
                        
                        if model_name == 'hybrid':
                            required_features = ['DM_normalized', 'maxdiameter_normalized', 'HTN_normalized', 'age_normalized', 'eGFR_normalized']
                            if all(feat in batch for feat in required_features):
                                outputs = model(batch)
                            else:
                                outputs = model({'image': batch['image']})
                        elif model_name in ['masked_resnet', 'masked_resnet_attention']:
                            outputs = model(batch)
                        else:
                            images = batch['image'].to(device, non_blocking=True)
                            outputs = model(images)
                        
                        probs = torch.softmax(outputs, dim=1)
                        preds = (probs[:, 1] > classification_threshold).long()
                        
                        batch_probs.append(probs[:, 1].cpu().numpy())
                        batch_preds.append(preds.cpu().numpy())
                        
                        all_individual_probs[model_idx].extend(probs[:, 1].cpu().numpy())
                        all_individual_preds[model_idx].extend(preds.cpu().numpy())
                    
                    # Ensemble predictions
                    batch_probs = np.array(batch_probs)  # Shape: (n_models, batch_size)
                    batch_preds = np.array(batch_preds)  # Shape: (n_models, batch_size)
                    
                    if ensemble_method == 'soft_voting':
                        # Average probabilities
                        ensemble_prob = np.mean(batch_probs, axis=0)
                        ensemble_pred = (ensemble_prob > classification_threshold).astype(int)
                    elif ensemble_method == 'hard_voting':
                        # Majority vote
                        ensemble_pred = (np.mean(batch_preds, axis=0) >= 0.5).astype(int)
                        ensemble_prob = np.mean(batch_probs, axis=0)
                    elif ensemble_method == 'weighted_voting':
                        # Weighted average by validation AUC
                        weights = fold_weights[:len(models)]
                        weights = weights / weights.sum()
                        ensemble_prob = np.average(batch_probs, axis=0, weights=weights)
                        ensemble_pred = (ensemble_prob > classification_threshold).astype(int)
                    else:
                        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_ensemble_probs.extend(ensemble_prob)
                    all_ensemble_preds.extend(ensemble_pred)
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Calculate ensemble metrics
    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)
    all_ensemble_preds = np.array(all_ensemble_preds)
    
    results = {
        'ensemble_method': ensemble_method,
        'n_models': len(models),
        'n_samples': len(all_labels)
    }
    
    try:
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
        
        # Calculate individual model metrics for comparison
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
                'fold': model_idx,
                'auc': float(model_auc),
                'accuracy': float(model_acc),
                'precision': float(model_precision),
                'recall': float(model_recall),
                'f1': float(model_f1)
            })
        
        results['individual_metrics'] = individual_metrics
        
        # Print results
        print(f"\n{'='*60}")
        print("Ensemble Evaluation Results")
        print(f"{'='*60}")
        print(f"\nEnsemble ({ensemble_method}):")
        print(f"  AUC:       {ensemble_auc:.4f} (95% CI: {ensemble_auc_ci_lower:.4f}-{ensemble_auc_ci_upper:.4f})")
        print(f"  Accuracy:  {ensemble_acc:.2f}%")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        print(f"\nIndividual Model Performance:")
        for metrics in individual_metrics:
            print(f"  Fold {metrics['fold']}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.4f}")
        
        # Compare ensemble vs individual models
        mean_individual_auc = np.mean([m['auc'] for m in individual_metrics])
        print(f"\nEnsemble Improvement:")
        print(f"  Ensemble AUC: {ensemble_auc:.4f}")
        print(f"  Mean Individual AUC: {mean_individual_auc:.4f}")
        print(f"  Improvement: {(ensemble_auc - mean_individual_auc):.4f} ({(ensemble_auc - mean_individual_auc) / mean_individual_auc * 100:.2f}%)")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        results['error'] = str(e)
    
    # Save results
    results_path = os.path.join(checkpoint_dir, f'ensemble_results_{ensemble_method}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Clean up
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
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    batch_size: int = 4,
    num_workers: int = 2,
    transform=None,
    ensemble_method: str = 'soft_voting',
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete pipeline: Run K-Fold CV training and then ensemble evaluation on test set
    
    Args:
        model_class: Class of the model to instantiate
        model_config: Configuration dictionary for model initialization
        train_dataset_items: List of training data items for CV
        test_loader: DataLoader for test set
        n_folds: Number of folds
        epochs: Training epochs per fold
        learning_rate: Learning rate
        device: Training device
        save_dir: Directory to save checkpoints
        batch_size: Batch size
        num_workers: Number of workers
        transform: Data transform
        ensemble_method: Ensemble method ('soft_voting', 'hard_voting', 'weighted_voting')
        config: Additional configuration
    
    Returns:
        Dictionary containing both CV and ensemble results
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
        transform=transform,
        config=config,
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
        ensemble_method=ensemble_method,
        config=config,
        **kwargs
    )
    
    # Combine results
    final_results = {
        'cv_results': cv_results,
        'ensemble_results': ensemble_results
    }
    
    # Save combined results
    final_results_path = os.path.join(save_dir, 'kfold_ensemble_final_results.json')
    
    # Prepare serializable results
    serializable_final = {
        'cv_summary': {
            'n_folds': cv_results['n_folds'],
            'best_fold': cv_results['best_fold'],
            'best_val_auc': float(cv_results['best_val_auc']),
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


def threshold_scan(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda',
    thresholds: Optional[List[float]] = None,
    optimize_metric: str = 'f1',
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Scan different classification thresholds to find optimal threshold
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation (typically validation set)
        device: Device for inference
        thresholds: List of thresholds to scan. Default: 0.1 to 0.9 in 0.05 steps
        optimize_metric: Metric to optimize ('f1', 'accuracy', 'youden', 'sensitivity', 'specificity', 'balanced_accuracy')
        config: Additional configuration
        save_path: Path to save results
    
    Returns:
        Dictionary containing:
        - best_threshold: Optimal threshold for the selected metric
        - all_results: Results for all thresholds
        - optimal_metrics: Metrics at optimal threshold
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Default thresholds
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05).tolist()
    
    print(f"\n{'='*70}")
    print(f"Threshold Scanning")
    print(f"Optimizing for: {optimize_metric}")
    print(f"Scanning {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    print(f"{'='*70}")
    
    # Collect all predictions and labels
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Collecting predictions")):
            try:
                if isinstance(batch, dict):
                    if 'image' not in batch:
                        continue
                    
                    if 'label' in batch:
                        labels = batch['label'].to(device)
                    else:
                        continue
                    
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long()
                    
                    # Get model predictions
                    model_name = config.get('model', {}).get('name', 'resnet') if config else 'resnet'
                    
                    if model_name == 'hybrid':
                        required_features = ['DM_normalized', 'maxdiameter_normalized', 'HTN_normalized', 'age_normalized', 'eGFR_normalized']
                        if all(feat in batch for feat in required_features):
                            outputs = model(batch)
                        else:
                            outputs = model({'image': batch['image']})
                    elif model_name in ['masked_resnet', 'masked_resnet_attention']:
                        outputs = model(batch)
                    else:
                        images = batch['image'].to(device, non_blocking=True)
                        outputs = model(images)
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    
    # Scan all thresholds
    results = []
    
    for threshold in thresholds:
        predictions = (all_probs >= threshold).astype(int)
        
        # Calculate metrics
        try:
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(all_labels, predictions, labels=[0, 1]).ravel()
            
            # Basic metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # Youden's J statistic (Sensitivity + Specificity - 1)
            youden_j = sensitivity + specificity - 1
            
            # Balanced accuracy
            balanced_accuracy = (sensitivity + specificity) / 2
            
            # F2 score (weights recall higher)
            f2 = 5 * precision * sensitivity / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0
            
            result = {
                'threshold': float(threshold),
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),  # Recall / TPR
                'specificity': float(specificity),  # TNR
                'precision': float(precision),  # PPV
                'npv': float(npv),
                'f1': float(f1),
                'f2': float(f2),
                'youden_j': float(youden_j),
                'balanced_accuracy': float(balanced_accuracy),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error at threshold {threshold}: {e}")
            continue
    
    # Find optimal threshold based on selected metric
    metric_map = {
        'f1': 'f1',
        'f2': 'f2',
        'accuracy': 'accuracy',
        'youden': 'youden_j',
        'sensitivity': 'sensitivity',
        'specificity': 'specificity',
        'balanced_accuracy': 'balanced_accuracy',
        'recall': 'sensitivity',
        'precision': 'precision'
    }
    
    target_metric = metric_map.get(optimize_metric, 'f1')
    
    best_idx = np.argmax([r[target_metric] for r in results])
    best_threshold = results[best_idx]['threshold']
    best_metrics = results[best_idx]
    
    # Calculate AUC (threshold-independent)
    try:
        auc, auc_ci_lower, auc_ci_upper = calculate_auc_with_ci(all_labels, all_probs)
    except:
        auc = auc_ci_lower = auc_ci_upper = 0.0
    
    # Print results table
    print(f"\n{'='*120}")
    print(f"{'Threshold':>10} | {'Acc':>7} | {'Sens':>7} | {'Spec':>7} | {'Prec':>7} | {'NPV':>7} | {'F1':>7} | {'F2':>7} | {'Youden':>7} | {'TP':>5} | {'TN':>5} | {'FP':>5} | {'FN':>5}")
    print(f"{'='*120}")
    
    for r in results:
        marker = " *" if r['threshold'] == best_threshold else "  "
        print(f"{r['threshold']:>8.2f}{marker} | {r['accuracy']:>7.4f} | {r['sensitivity']:>7.4f} | {r['specificity']:>7.4f} | "
              f"{r['precision']:>7.4f} | {r['npv']:>7.4f} | {r['f1']:>7.4f} | {r['f2']:>7.4f} | {r['youden_j']:>7.4f} | "
              f"{r['tp']:>5} | {r['tn']:>5} | {r['fp']:>5} | {r['fn']:>5}")
    
    print(f"{'='*120}")
    print(f"* = Best threshold based on {optimize_metric}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Threshold Scanning Results Summary")
    print(f"{'='*70}")
    print(f"AUC: {auc:.4f} (95% CI: {auc_ci_lower:.4f} - {auc_ci_upper:.4f})")
    print(f"\nOptimal Threshold (optimized for {optimize_metric}): {best_threshold:.3f}")
    print(f"  Accuracy:     {best_metrics['accuracy']:.4f}")
    print(f"  Sensitivity:  {best_metrics['sensitivity']:.4f} (Recall/TPR)")
    print(f"  Specificity:  {best_metrics['specificity']:.4f} (TNR)")
    print(f"  Precision:    {best_metrics['precision']:.4f} (PPV)")
    print(f"  NPV:          {best_metrics['npv']:.4f}")
    print(f"  F1 Score:     {best_metrics['f1']:.4f}")
    print(f"  F2 Score:     {best_metrics['f2']:.4f}")
    print(f"  Youden's J:   {best_metrics['youden_j']:.4f}")
    print(f"  Balanced Acc: {best_metrics['balanced_accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={best_metrics['tp']}, TN={best_metrics['tn']}, FP={best_metrics['fp']}, FN={best_metrics['fn']}")
    
    # Compile final results
    scan_results = {
        'auc': float(auc),
        'auc_ci_lower': float(auc_ci_lower),
        'auc_ci_upper': float(auc_ci_upper),
        'optimize_metric': optimize_metric,
        'best_threshold': float(best_threshold),
        'best_metrics': best_metrics,
        'all_results': results,
        'n_samples': len(all_labels),
        'class_distribution': np.bincount(all_labels).tolist()
    }
    
    # Save results if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(scan_results, f, indent=2)
        print(f"\nResults saved to: {save_path}")
    
    return scan_results


def ensemble_threshold_scan(
    model_class,
    model_config: Dict[str, Any],
    data_loader: DataLoader,
    checkpoint_dir: str,
    n_folds: int = 5,
    device: str = 'cuda',
    ensemble_method: str = 'soft_voting',
    thresholds: Optional[List[float]] = None,
    optimize_metric: str = 'f1',
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Scan thresholds for ensemble model predictions
    
    Args:
        model_class: Class of the model to instantiate
        model_config: Configuration dictionary for model initialization
        data_loader: DataLoader for evaluation
        checkpoint_dir: Directory containing fold checkpoints
        n_folds: Number of folds in ensemble
        device: Device for inference
        ensemble_method: Method for combining predictions ('soft_voting', 'hard_voting', 'weighted_voting')
        thresholds: List of thresholds to scan
        optimize_metric: Metric to optimize
        config: Additional configuration
        save_path: Path to save results
    
    Returns:
        Dictionary containing threshold scanning results for ensemble
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Default thresholds
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05).tolist()
    
    print(f"\n{'='*70}")
    print(f"Ensemble Threshold Scanning")
    print(f"Ensemble method: {ensemble_method}")
    print(f"Optimizing for: {optimize_metric}")
    print(f"{'='*70}")
    
    # Load validation AUCs for weighted voting
    fold_weights = []
    cv_results_path = os.path.join(checkpoint_dir, 'cv_results.json')
    if os.path.exists(cv_results_path):
        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)
            for fold_metric in cv_results.get('fold_metrics', []):
                fold_weights.append(fold_metric.get('val_auc', 1.0))
    else:
        fold_weights = [1.0] * n_folds
    
    fold_weights = np.array(fold_weights)
    fold_weights = fold_weights / fold_weights.sum()
    
    # Load all fold models
    models = []
    for fold in range(n_folds):
        fold_checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold}', 'best_model.pth')
        
        if not os.path.exists(fold_checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold} not found")
            continue
        
        model = model_class(**model_config)
        checkpoint = torch.load(fold_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)
    
    if len(models) == 0:
        raise ValueError("No models loaded for ensemble")
    
    print(f"Loaded {len(models)} models for ensemble")
    
    # Collect all predictions
    all_labels = []
    all_ensemble_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Collecting ensemble predictions")):
            try:
                if isinstance(batch, dict):
                    if 'image' not in batch:
                        continue
                    
                    if 'label' in batch:
                        labels = batch['label'].to(device)
                    else:
                        continue
                    
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long()
                    
                    # Get predictions from each model
                    batch_probs = []
                    
                    for model in models:
                        model_name = config.get('model', {}).get('name', 'resnet') if config else 'resnet'
                        
                        if model_name == 'hybrid':
                            required_features = ['DM_normalized', 'maxdiameter_normalized', 'HTN_normalized', 'age_normalized', 'eGFR_normalized']
                            if all(feat in batch for feat in required_features):
                                outputs = model(batch)
                            else:
                                outputs = model({'image': batch['image']})
                        elif model_name in ['masked_resnet', 'masked_resnet_attention']:
                            outputs = model(batch)
                        else:
                            images = batch['image'].to(device, non_blocking=True)
                            outputs = model(images)
                        
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        batch_probs.append(probs.cpu().numpy())
                    
                    batch_probs = np.array(batch_probs)  # Shape: (n_models, batch_size)
                    
                    # Ensemble probabilities
                    if ensemble_method == 'soft_voting':
                        ensemble_prob = np.mean(batch_probs, axis=0)
                    elif ensemble_method == 'weighted_voting':
                        weights = fold_weights[:len(models)]
                        weights = weights / weights.sum()
                        ensemble_prob = np.average(batch_probs, axis=0, weights=weights)
                    else:  # hard_voting - use mean probs for threshold scanning
                        ensemble_prob = np.mean(batch_probs, axis=0)
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_ensemble_probs.extend(ensemble_prob)
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)
    
    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    
    # Scan all thresholds
    results = []
    
    for threshold in thresholds:
        predictions = (all_ensemble_probs >= threshold).astype(int)
        
        try:
            tn, fp, fn, tp = confusion_matrix(all_labels, predictions, labels=[0, 1]).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            youden_j = sensitivity + specificity - 1
            balanced_accuracy = (sensitivity + specificity) / 2
            f2 = 5 * precision * sensitivity / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0
            
            result = {
                'threshold': float(threshold),
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'npv': float(npv),
                'f1': float(f1),
                'f2': float(f2),
                'youden_j': float(youden_j),
                'balanced_accuracy': float(balanced_accuracy),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error at threshold {threshold}: {e}")
            continue
    
    # Find optimal threshold
    metric_map = {
        'f1': 'f1', 'f2': 'f2', 'accuracy': 'accuracy', 'youden': 'youden_j',
        'sensitivity': 'sensitivity', 'specificity': 'specificity',
        'balanced_accuracy': 'balanced_accuracy', 'recall': 'sensitivity', 'precision': 'precision'
    }
    
    target_metric = metric_map.get(optimize_metric, 'f1')
    best_idx = np.argmax([r[target_metric] for r in results])
    best_threshold = results[best_idx]['threshold']
    best_metrics = results[best_idx]
    
    # Calculate AUC
    try:
        auc, auc_ci_lower, auc_ci_upper = calculate_auc_with_ci(all_labels, all_ensemble_probs)
    except:
        auc = auc_ci_lower = auc_ci_upper = 0.0
    
    # Print results table
    print(f"\n{'='*120}")
    print(f"{'Threshold':>10} | {'Acc':>7} | {'Sens':>7} | {'Spec':>7} | {'Prec':>7} | {'NPV':>7} | {'F1':>7} | {'F2':>7} | {'Youden':>7} | {'TP':>5} | {'TN':>5} | {'FP':>5} | {'FN':>5}")
    print(f"{'='*120}")
    
    for r in results:
        marker = " *" if r['threshold'] == best_threshold else "  "
        print(f"{r['threshold']:>8.2f}{marker} | {r['accuracy']:>7.4f} | {r['sensitivity']:>7.4f} | {r['specificity']:>7.4f} | "
              f"{r['precision']:>7.4f} | {r['npv']:>7.4f} | {r['f1']:>7.4f} | {r['f2']:>7.4f} | {r['youden_j']:>7.4f} | "
              f"{r['tp']:>5} | {r['tn']:>5} | {r['fp']:>5} | {r['fn']:>5}")
    
    print(f"{'='*120}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Ensemble Threshold Scanning Results Summary")
    print(f"{'='*70}")
    print(f"Ensemble Method: {ensemble_method}")
    print(f"AUC: {auc:.4f} (95% CI: {auc_ci_lower:.4f} - {auc_ci_upper:.4f})")
    print(f"\nOptimal Threshold (optimized for {optimize_metric}): {best_threshold:.3f}")
    print(f"  Accuracy:     {best_metrics['accuracy']:.4f}")
    print(f"  Sensitivity:  {best_metrics['sensitivity']:.4f}")
    print(f"  Specificity:  {best_metrics['specificity']:.4f}")
    print(f"  Precision:    {best_metrics['precision']:.4f}")
    print(f"  F1 Score:     {best_metrics['f1']:.4f}")
    print(f"  Youden's J:   {best_metrics['youden_j']:.4f}")
    
    scan_results = {
        'ensemble_method': ensemble_method,
        'auc': float(auc),
        'auc_ci_lower': float(auc_ci_lower),
        'auc_ci_upper': float(auc_ci_upper),
        'optimize_metric': optimize_metric,
        'best_threshold': float(best_threshold),
        'best_metrics': best_metrics,
        'all_results': results,
        'n_samples': len(all_labels),
        'n_models': len(models)
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(scan_results, f, indent=2)
        print(f"\nResults saved to: {save_path}")
    
    # Clean up
    for model in models:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return scan_results


def find_optimal_threshold_by_metric(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    metric: str = 'youden',
    min_sensitivity: float = 0.0,
    min_specificity: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold based on specific metric with optional constraints
    
    Args:
        all_labels: Ground truth labels
        all_probs: Predicted probabilities
        metric: Optimization metric ('youden', 'f1', 'f2', 'balanced_accuracy')
        min_sensitivity: Minimum required sensitivity
        min_specificity: Minimum required specificity
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    best_threshold = 0.5
    best_metric_value = -np.inf
    best_metrics = {}
    
    for threshold in thresholds:
        predictions = (all_probs >= threshold).astype(int)
        
        try:
            tn, fp, fn, tp = confusion_matrix(all_labels, predictions, labels=[0, 1]).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Check constraints
            if sensitivity < min_sensitivity or specificity < min_specificity:
                continue
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            if metric == 'youden':
                metric_value = sensitivity + specificity - 1
            elif metric == 'f1':
                metric_value = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            elif metric == 'f2':
                metric_value = 5 * precision * sensitivity / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0
            elif metric == 'balanced_accuracy':
                metric_value = (sensitivity + specificity) / 2
            else:
                metric_value = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
                best_metrics = {
                    'threshold': float(threshold),
                    'sensitivity': float(sensitivity),
                    'specificity': float(specificity),
                    'precision': float(precision),
                    'f1': float(2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0),
                    'youden_j': float(sensitivity + specificity - 1),
                    metric: float(metric_value)
                }
        except:
            continue
    
    return best_threshold, best_metrics