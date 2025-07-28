import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
from monai.networks.layers import Norm


class MaskedResNetClassifier(nn.Module):
    """ResNet classifier for masked CT images only (no numerical features)"""
    
    def __init__(self, 
                 spatial_dims=3,
                 in_channels=1,
                 num_classes=2,
                 backbone='resnet50',
                 dropout_rate=0.3,
                 pretrained=False):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        
        # 3D ResNet backbone for masked CT images
        if backbone == 'resnet50':
            self.backbone = ResNet(
                block='bottleneck',
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=512,
                norm=Norm.BATCH
            )
            backbone_features = 512
        else:
            # Default ResNet (similar to original)
            self.backbone = ResNet(
                block='basic',
                layers=[2, 2, 2, 2],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=512,
                norm=Norm.BATCH
            )
            backbone_features = 512
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        """Forward pass using only masked images"""
        device = next(self.parameters()).device
        
        if isinstance(batch, dict):
            if 'masked_image' in batch:
                images = batch['masked_image'].to(device)
            else:
                images = batch['image'].to(device)
                if 'mask' in batch:
                    mask = batch['mask'].to(device)
                    images = images * mask
        else:
            images = batch.to(device)
        
        features = self.backbone(images)
        logits = self.classifier(features)
        
        return logits


class MaskedResNetWithAttention(nn.Module):
    """Enhanced MaskedResNet with attention mechanism"""
    
    def __init__(self, 
                 spatial_dims=3,
                 in_channels=1,
                 num_classes=2,
                 backbone='resnet50',
                 dropout_rate=0.3,
                 attention_heads=8):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        
        # 3D ResNet backbone
        self.backbone = ResNet(
            block='bottleneck' if backbone == 'resnet50' else 'basic',
            layers=[3, 4, 6, 3] if backbone == 'resnet50' else [2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=512,
            norm=Norm.BATCH
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        device = next(self.parameters()).device
        
        # Get masked image
        if isinstance(batch, dict):
            if 'masked_image' in batch:
                images = batch['masked_image'].to(device)
            else:
                images = batch['image'].to(device)
                if 'mask' in batch:
                    mask = batch['mask'].to(device)
                    images = images * mask
        else:
            images = batch.to(device)
        
        # Extract features
        features = self.backbone(images)
        
        # Apply self-attention
        features_unsqueezed = features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits