import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNet
from monai.networks.layers import Norm


class MultiModalCTClassifier(nn.Module):
    """Hybrid model combining 3D CNN for CT images and MLP for numerical features"""

    def __init__(self,
                 spatial_dims=3,
                 in_channels=1,
                 num_classes=2,
                 backbone='resnet50',
                 numerical_features_dim=5,
                 fusion_method='multihead_cross_attention',
                 dropout_rate=0.1,
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
            self.visual_attention = nn.MultiheadAttention(
                embed_dim=cnn_features_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.visual_norm = nn.LayerNorm(cnn_features_dim)
            self.visual_projection = nn.Sequential(
                nn.Linear(cnn_features_dim * 3, cnn_features_dim),
                nn.LayerNorm(cnn_features_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

        clinical_token_dim = 64
        self.continuous_feature_names = ['maxdiameter_normalized', 'age_normalized', 'eGFR_normalized']
        self.categorical_feature_names = ['DM', 'HTN']
        self.continuous_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1, clinical_token_dim),
                nn.LayerNorm(clinical_token_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            for name in self.continuous_feature_names
        })
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(2, clinical_token_dim)
            for name in self.categorical_feature_names
        })
        self.clinical_self_attention = nn.MultiheadAttention(
            embed_dim=clinical_token_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        self.clinical_norm = nn.LayerNorm(clinical_token_dim)
        numerical_features_dim = clinical_token_dim

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
        else:
            valid = ['concat', 'attention', 'bilinear', 'multihead_cross_attention']
            raise ValueError(f"Unknown fusion_method: {fusion_method}. Valid options: {valid}")

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _encode_clinical_tokens(self, batch, batch_size, device):
        tokens = []

        for feature in self.continuous_feature_names:
            value = batch.get(feature, None)
            if isinstance(value, torch.Tensor):
                value = value.to(device).float().view(batch_size, 1)
            else:
                value = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
            tokens.append(self.continuous_encoders[feature](value))

        for feature in self.categorical_feature_names:
            value = batch.get(feature, None)
            if isinstance(value, torch.Tensor):
                value = value.to(device).long().view(batch_size).clamp(0, 1)
            else:
                value = torch.zeros(batch_size, dtype=torch.long, device=device)
            tokens.append(self.categorical_embeddings[feature](value))

        clinical_tokens = torch.stack(tokens, dim=1)
        attended_tokens, _ = self.clinical_self_attention(clinical_tokens, clinical_tokens, clinical_tokens)
        return self.clinical_norm(attended_tokens + clinical_tokens)

    @staticmethod
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

        if isinstance(mask_tensor, torch.Tensor):
            mask = mask_tensor.to(device)

        if isinstance(image_tensor, torch.Tensor):
            images = image_tensor.to(device)
            if mask is not None and not self.use_mask_attention:
                mask = self._match_volume_shape(mask.float(), images)
                images = images * mask
        elif isinstance(masked_image_tensor, torch.Tensor):
            images = masked_image_tensor.to(device)
        else:
            raise ValueError("Batch must contain 'image' or 'masked_image' as a tensor.")

        batch_size = images.shape[0]
        clinical_tokens = self._encode_clinical_tokens(batch, batch_size, device)

        if self.use_mask_attention and mask is not None:
            mask = self._match_volume_shape(mask.float(), images)
            visual_inputs = torch.cat([
                images,
                images * mask,
                images * (1.0 - mask)
            ], dim=0)
            visual_features = self.image_encoder(visual_inputs)
            visual_tokens = torch.stack(torch.chunk(visual_features, 3, dim=0), dim=1)
            attended_tokens, _ = self.visual_attention(visual_tokens, visual_tokens, visual_tokens)
            attended_tokens = self.visual_norm(attended_tokens + visual_tokens)
            image_features = self.visual_projection(attended_tokens.flatten(1))
        else:
            image_features = self.image_encoder(images)

        clinical_summary = clinical_tokens.mean(dim=1)

        if self.fusion_method == 'concat':
            fused_features = torch.cat([image_features, clinical_summary], dim=1)
        elif self.fusion_method == 'attention':
            numerical_proj = self.numerical_projection(clinical_summary).unsqueeze(1)
            image_features_unsqueezed = image_features.unsqueeze(1)
            attended_features, _ = self.attention_layer(
                image_features_unsqueezed,
                numerical_proj,
                numerical_proj
            )
            fused_features = attended_features.squeeze(1)
        elif self.fusion_method == 'bilinear':
            fused_features = self.bilinear_fusion(image_features, clinical_summary)
        elif self.fusion_method == 'multihead_cross_attention':
            clinical_features = self.clinical_feature_projection(clinical_tokens)
            image_query = image_features.unsqueeze(1)

            attended_features, attention_weights = self.cross_attention(
                query=image_query,
                key=clinical_features,
                value=clinical_features
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


# Alias for backward compatibility
HybridModel = MultiModalCTClassifier


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
