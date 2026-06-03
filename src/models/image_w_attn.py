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


class ImageWithAttentionModel(nn.Module):
    """3D CT model with visual attention over full, renal, and outer volumes."""

    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        num_classes=2,
        backbone='resnet50',
        dropout_rate=0.1,
        feature_dim=512,
        attention_heads=8,
        use_mask_attention=True
    ):
        super(ImageWithAttentionModel, self).__init__()

        if backbone != 'resnet50':
            raise ValueError("ImageWithAttentionModel supports backbone='resnet50'.")

        self.use_mask_attention = use_mask_attention
        self.image_encoder = ResNet(
            block='bottleneck',
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=feature_dim,
            norm=Norm.BATCH
        )

        if use_mask_attention:
            self.visual_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.visual_norm = nn.LayerNorm(feature_dim)
            self.visual_projection = nn.Sequential(
                nn.Linear(feature_dim * 3, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def _get_inputs(self, batch, masks=None):
        device = next(self.parameters()).device

        if isinstance(batch, dict):
            images = batch.get('image', batch.get('masked_image'))
            mask = batch.get('mask', masks)
        else:
            images = batch
            mask = masks

        if images is None:
            raise ValueError("ImageWithAttentionModel requires image data.")

        images = images.to(device)
        mask = mask.to(device) if isinstance(mask, torch.Tensor) else None
        return images, mask

    def forward(self, batch, masks=None):
        """Forward pass using volumetric CT inputs."""
        images, mask = self._get_inputs(batch, masks)

        if self.use_mask_attention and mask is not None:
            mask = _match_volume_shape(mask.float(), images)
            visual_inputs = torch.cat([
                images,
                images * mask,
                images * (1.0 - mask)
            ], dim=0)
            visual_features = self.image_encoder(visual_inputs)
            visual_tokens = torch.stack(torch.chunk(visual_features, 3, dim=0), dim=1)
            attended_tokens, _ = self.visual_attention(visual_tokens, visual_tokens, visual_tokens)
            attended_tokens = self.visual_norm(attended_tokens + visual_tokens)
            features = self.visual_projection(attended_tokens.flatten(1))
        else:
            features = self.image_encoder(images)

        return self.classifier(features)
