import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from typing import Optional, Dict, Any
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Use Adam instead of AdamW
    
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