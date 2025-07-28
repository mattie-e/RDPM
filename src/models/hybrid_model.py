import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNet
from monai.networks.layers import Norm
import numpy as np


class MaskAttentionModule(nn.Module):
    """Improved mask attention module with better gradient flow"""
    def __init__(self, feature_channels, reduction_ratio=16, min_scale=0.1):
        super(MaskAttentionModule, self).__init__()
        
        self.feature_channels = feature_channels
        self.min_scale = min_scale
        self.debug_mode = False
        
        self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // reduction_ratio, 1, bias=False),
            nn.BatchNorm3d(feature_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // reduction_ratio, feature_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_debug(self, debug_mode):
        self.debug_mode = debug_mode
    
    def forward(self, features, mask):
        """Apply mask-guided attention with improved gradient flow"""
        if self.debug_mode:
            print(f"MaskAttention input: features={features.shape}, mask={mask.shape}")
            print(f"Scale factor: {self.scale_factor.item():.4f}")
        
        if mask.requires_grad == False:
            mask = mask.detach().requires_grad_(True)
        
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_max > mask_min:
            normalized_mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)
        else:
            normalized_mask = torch.ones_like(mask) * 0.5
        
        baseline_attention = torch.full_like(normalized_mask, self.min_scale)
        enhanced_mask = normalized_mask + baseline_attention
        
        if self.debug_mode:
            print(f"Enhanced mask: min={enhanced_mask.min():.4f}, max={enhanced_mask.max():.4f}")
        
        channel_weights = self.channel_attention(features)
        
        avg_out = torch.mean(features, dim=1, keepdim=True)
        max_out, _ = torch.max(features, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        
        combined_attention = (
            channel_weights * 
            spatial_weights * 
            enhanced_mask * 
            self.scale_factor.abs()
        )
        
        attended_features = features * (1.0 + combined_attention)
        
        if self.debug_mode:
            print(f"Combined attention: min={combined_attention.min():.4f}, max={combined_attention.max():.4f}")
            print(f"Attended features: min={attended_features.min():.4f}, max={attended_features.max():.4f}")
            
            if combined_attention.requires_grad:
                print("Combined attention requires gradients: True")
            if self.scale_factor.grad is not None:
                print(f"Scale factor gradient: {self.scale_factor.grad.item():.6f}")
        
        return attended_features


class MultiModalCTClassifier(nn.Module):
    """Hybrid model combining 3D CNN for CT images and MLP for numerical features"""
    
    def __init__(self, 
                 spatial_dims=3,
                 in_channels=1,
                 num_classes=2,
                 backbone='efficientnet-b0',
                 numerical_features_dim=3,
                 fusion_method='multihead_cross_attention',
                 dropout_rate=0.3,
                 use_mask_attention=True):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.use_mask_attention = use_mask_attention
        
        if backbone.startswith('efficientnet'):
            self.image_encoder = EfficientNetBN(
                model_name=backbone,
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                num_classes=512,
                norm=Norm.BATCH,
                dropout_prob=dropout_rate
            )
            cnn_features_dim = 512
        elif backbone == 'resnet':
            self.image_encoder = ResNet(
                block='basic',
                layers=[2, 2, 2, 2],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=512,
                norm=Norm.BATCH
            )
            cnn_features_dim = 512
        elif backbone.startswith('resnet50'):
            self.image_encoder = ResNet(
                block='bottleneck',
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=512,
                norm=Norm.BATCH
            )
            cnn_features_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        if use_mask_attention:
            self.mask_attention = MaskAttentionModule(feature_channels=cnn_features_dim)
        
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        numerical_features_dim = 64
        
        if fusion_method == 'concat':
            fusion_dim = cnn_features_dim + numerical_features_dim
            self.fusion_layer = nn.Identity()
        elif fusion_method == 'attention':
            fusion_dim = cnn_features_dim
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=cnn_features_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.numerical_projection = nn.Linear(numerical_features_dim, cnn_features_dim)
        elif fusion_method == 'bilinear':
            fusion_dim = 256
            self.bilinear_fusion = nn.Bilinear(cnn_features_dim, numerical_features_dim, fusion_dim)
        elif fusion_method == 'multihead_cross_attention':
            num_clinical_features = 5
            self.clinical_feature_projection = nn.Linear(numerical_features_dim, cnn_features_dim)
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=cnn_features_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            
            fusion_dim = cnn_features_dim * 2
            
            self.attended_projection = nn.Sequential(
                nn.Linear(cnn_features_dim, cnn_features_dim),
                nn.LayerNorm(cnn_features_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            
            self.final_fusion = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            
            fusion_dim = fusion_dim // 2
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
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

        if not isinstance(batch, dict):
            raise ValueError("Input to MultiModalCTClassifier.forward must be a dict with keys for image data.")

        images = None
        mask = None

        image_tensor = batch.get('image', None)
        masked_image_tensor = batch.get('masked_image', None)
        mask_tensor = batch.get('mask', None)

        if isinstance(masked_image_tensor, torch.Tensor):
            images = masked_image_tensor.to(device)
        elif isinstance(image_tensor, torch.Tensor):
            images = image_tensor.to(device)
            if isinstance(mask_tensor, torch.Tensor):
                mask = mask_tensor.to(device)
                if mask.shape == images.shape:
                    images = images * mask
                elif mask.shape == images.shape[1:]:
                    mask = mask.unsqueeze(0) if mask.dim() == images.dim() - 1 else mask
                    images = images * mask
        else:
            raise ValueError("Batch must contain 'image' or 'masked_image' as a tensor.")

        required_features = ['DM_normalized', 'maxdiameter_normalized', 'HTN_normalized', 'age_normalized', 'eGFR_normalized']
        numerical_values = []
        batch_size = images.shape[0]

        for feature in required_features:
            value = batch.get(feature, None)
            if isinstance(value, torch.Tensor):
                numerical_values.append(value.to(device))
            else:
                numerical_values.append(torch.zeros(batch_size, dtype=torch.float32, device=device))

        numerical_features = torch.stack(numerical_values, dim=1)

        image_features = self.image_encoder(images)
        
        if self.use_mask_attention and mask is not None:
            if hasattr(self.image_encoder, 'feature_extractor'):
                with torch.no_grad():
                    feature_maps = self.image_encoder.feature_extractor(images)
                
                resized_mask = F.interpolate(
                    mask, 
                    size=feature_maps.shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
                
                attended_feature_maps = self.mask_attention(feature_maps, resized_mask)
                image_features = self.image_encoder.classifier(
                    self.image_encoder.pool(attended_feature_maps).flatten(1)
                )
            else:
                feature_shape = image_features.shape
                reshaped_features = image_features.view(feature_shape[0], feature_shape[1], 1, 1, 1)

                reduced_mask = F.adaptive_avg_pool3d(mask, (1, 1, 1))
                
                attended_features = self.mask_attention(reshaped_features, reduced_mask)
                image_features = attended_features.view(feature_shape)

        numerical_features = self.numerical_encoder(numerical_features)

        if self.fusion_method == 'concat':
            fused_features = torch.cat([image_features, numerical_features], dim=1)
        elif self.fusion_method == 'attention':
            numerical_proj = self.numerical_projection(numerical_features).unsqueeze(1)
            image_features_unsqueezed = image_features.unsqueeze(1)
            attended_features, _ = self.attention_layer(
                image_features_unsqueezed, 
                numerical_proj, 
                numerical_proj
            )
            fused_features = attended_features.squeeze(1)
        elif self.fusion_method == 'bilinear':
            fused_features = self.bilinear_fusion(image_features, numerical_features)
        elif self.fusion_method == 'multihead_cross_attention':
            clinical_features = self.clinical_feature_projection(numerical_features)
            image_query = image_features.unsqueeze(1)
            clinical_kv = clinical_features.unsqueeze(1)
            
            attended_features, attention_weights = self.cross_attention(
                query=image_query,
                key=clinical_kv,
                value=clinical_kv
            )
            attended_features = attended_features.squeeze(1)
            attended_features = self.attended_projection(attended_features)
            
            concat_features = torch.cat([image_features, attended_features], dim=1)
            fused_features = self.final_fusion(concat_features)

        logits = self.classifier(fused_features)

        return logits

    def load_pretrained_resnet50(self, checkpoint_path):
        """Load a 2D ResNet50 checkpoint into the 3D ResNet50 backbone"""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace('module.', '').replace('backbone.', '')
            if k_clean in self.image_encoder.state_dict():
                if self.image_encoder.state_dict()[k_clean].shape == v.shape:
                    new_state_dict[k_clean] = v
        missing, unexpected = self.image_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained ResNet50 weights. Missing keys: {missing}, Unexpected keys: {unexpected}")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss with class weights"""
    def __init__(self, class_weights=None, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
