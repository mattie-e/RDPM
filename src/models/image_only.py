import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
from monai.networks.layers import Norm


def _match_volume_shape(tensor, reference):
    if tensor.dim() == reference.dim() - 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == reference.dim() - 1:
        if tensor.shape[0] == reference.shape[0]:
            tensor = tensor.unsqueeze(1)
        else:
            tensor = tensor.unsqueeze(0)

    if tensor.shape[0] == 1 and reference.shape[0] > 1:
        tensor = tensor.expand(reference.shape[0], *tensor.shape[1:])

    if tensor.shape[1] == 1 and reference.shape[1] > 1:
        tensor = tensor.expand(tensor.shape[0], reference.shape[1], *tensor.shape[2:])

    if tensor.shape[2:] != reference.shape[2:]:
        tensor = F.interpolate(tensor.float(), size=reference.shape[2:], mode='nearest')

    return tensor


class MaskedResNetClassifier(nn.Module):
    """ResNet classifier for masked CT images only (no numerical features)"""

    def __init__(self,
                 spatial_dims=3,
                 in_channels=1,
                 num_classes=2,
                 backbone='resnet50',
                 dropout_rate=0.1,
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
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
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
                    mask = _match_volume_shape(mask.float(), images)
                    images = images * mask
        else:
            images = batch.to(device)

        features = self.backbone(images)
        logits = self.classifier(features)

        return logits


class MaskedResNetWithAttention(nn.Module):
    """3D CT-only model with visual attention over full, renal, and outer volumes."""

    def __init__(self,
                 spatial_dims=3,
                 in_channels=1,
                 num_classes=2,
                 backbone='resnet50',
                 dropout_rate=0.1,
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
        self.attention_norm = nn.LayerNorm(512)
        self.attention_projection = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
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
            mask = batch.get('mask')
            if 'image' in batch:
                images = batch['image'].to(device)
            elif 'masked_image' in batch:
                images = batch['masked_image'].to(device)
                mask = None
            else:
                raise ValueError("Batch must contain 'image' or 'masked_image'.")
            if isinstance(mask, torch.Tensor):
                mask = mask.to(device)
        else:
            images = batch.to(device)
            mask = None

        if mask is not None:
            mask = _match_volume_shape(mask.float(), images)
            visual_inputs = torch.cat([
                images,
                images * mask,
                images * (1.0 - mask)
            ], dim=0)
            visual_features = self.backbone(visual_inputs)
            feature_tokens = torch.stack(torch.chunk(visual_features, 3, dim=0), dim=1)
            attended_tokens, _ = self.attention(feature_tokens, feature_tokens, feature_tokens)
            attended_tokens = self.attention_norm(attended_tokens + feature_tokens)
            features = self.attention_projection(attended_tokens.flatten(1))
        else:
            features = self.backbone(images)

        # Classification
        logits = self.classifier(features)

        return logits
