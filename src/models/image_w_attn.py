import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

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
        
        return attended_features


class ImageWithAttentionModel(nn.Module):
    """Image model with mask attention mechanism"""
    def __init__(self, in_channels=1, num_classes=2, feature_dim=256):
        super(ImageWithAttentionModel, self).__init__()
        
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.feature_channels = 2048
        
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.attention_module = MaskAttentionModule(self.feature_channels)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.feature_channels, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, images, masks):
        """Forward pass with attention mechanism"""
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        image_features = self.layer4(x)
        
        mask_features = self.mask_encoder(masks)
        
        image_features_3d = image_features.unsqueeze(-1)
        mask_features_3d = mask_features.unsqueeze(-1)
        
        attended_features = self.attention_module(image_features_3d, mask_features_3d)
        
        attended_features = attended_features.squeeze(-1)
        
        x = self.decoder(attended_features)
        x = x.view(x.size(0), -1)
        
        output = self.classifier(x)
        
        return output
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, images, masks):
        """
        Forward pass with attention mechanism
        
        Args:
            images: Input images [B, C, H, W]
            masks: Binary masks [B, 1, H, W]
        
        Returns:
            Classification output
        """
        # Extract image features
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        image_features = self.layer4(x)  # [B, 2048, H/32, W/32]
        
        # Extract mask features
        mask_features = self.mask_encoder(masks)
        
        # Add dummy dimension to make 3D for attention module
        image_features_3d = image_features.unsqueeze(-1)
        mask_features_3d = mask_features.unsqueeze(-1)
        
        # Apply attention
        attended_features = self.attention_module(image_features_3d, mask_features_3d)
        
        # Remove dummy dimension
        attended_features = attended_features.squeeze(-1)
        
        # Decode
        x = self.decoder(attended_features)
        x = x.view(x.size(0), -1)
        
        # Classify
        output = self.classifier(x)
        
        return output
