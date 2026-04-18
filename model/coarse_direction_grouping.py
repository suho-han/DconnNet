import math

import torch
import torch.nn as nn

COARSE_DIRECTION_GROUPS_DXDY = (
    # Canonical 8-direction order from connect_loss.CANONICAL_8_OFFSETS:
    # (dy, dx): [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
    # Stored below in (dx, dy) tuple convention.
    ((1, 1), (2, 2), (2, 1)),
    ((0, 1), (0, 2), (1, 2)),
    ((-1, 1), (-2, 2), (-1, 2)),
    ((1, 0), (2, 0), (2, -1)),
    ((-1, 0), (-2, 0), (-2, 1)),
    ((1, -1), (2, -2), (1, -2)),
    ((0, -1), (0, -2), (-1, -2)),
    ((-1, -1), (-2, -2), (-2, -1)),
)
COARSE_DIRECTION_GROUP_NAMES = ('SE', 'S', 'SW', 'E', 'W', 'NE', 'N', 'NW')

FUSION_TYPES = (
    'mean',
    'weighted_sum',
    'conv1x1',
    'attention_gating',
)


def get_coarse_direction_groups_dxdy():
    return [[tuple(offset) for offset in group] for group in COARSE_DIRECTION_GROUPS_DXDY]


def dxdy_to_dydx(offset):
    dx, dy = offset
    return (dy, dx)


def dydx_to_dxdy(offset):
    dy, dx = offset
    return (dx, dy)


def build_group_index_map_from_offsets(offsets_dydx):
    offset_to_index = {dydx_to_dxdy(offset): idx for idx, offset in enumerate(offsets_dydx)}
    group_index_map = []
    for group in COARSE_DIRECTION_GROUPS_DXDY:
        group_indices = []
        for offset in group:
            if offset not in offset_to_index:
                raise ValueError(f'Offset {offset} is missing from the provided 24-channel layout')
            group_indices.append(offset_to_index[offset])
        group_index_map.append(group_indices)
    return group_index_map


def compute_group_mean_vectors(groups_dxdy=None):
    if groups_dxdy is None:
        groups_dxdy = COARSE_DIRECTION_GROUPS_DXDY
    group_tensor = torch.tensor(groups_dxdy, dtype=torch.float32)
    return group_tensor.mean(dim=1)


def compute_representative_angles(groups_dxdy=None):
    mean_vectors = compute_group_mean_vectors(groups_dxdy)
    mean_dx = mean_vectors[:, 0]
    mean_dy = mean_vectors[:, 1]
    angles = torch.atan2(-mean_dy, mean_dx) * (180.0 / math.pi)
    return torch.remainder(angles + 360.0, 360.0)


def summarize_coarse_direction_metadata(offsets_dydx=None):
    mean_vectors = compute_group_mean_vectors()
    representative_angles = compute_representative_angles()
    group_index_map = None
    if offsets_dydx is not None:
        group_index_map = build_group_index_map_from_offsets(offsets_dydx)

    summary = []
    for idx, group_name in enumerate(COARSE_DIRECTION_GROUP_NAMES):
        entry = {
            'group_name': group_name,
            'group_index': idx,
            'offsets_dxdy': list(COARSE_DIRECTION_GROUPS_DXDY[idx]),
            'mean_vector_dxdy': tuple(mean_vectors[idx].tolist()),
            'representative_angle_deg': float(representative_angles[idx].item()),
        }
        if group_index_map is not None:
            entry['fine_channel_indices'] = list(group_index_map[idx])
        summary.append(entry)
    return summary


def _resolve_default_offsets_dydx():
    from connect_loss import resolve_connectivity_layout

    return resolve_connectivity_layout(24)['offsets']


class MeanDirectionFusion(nn.Module):
    def forward(self, grouped_features):
        return grouped_features.mean(dim=3)


class WeightedSumDirectionFusion(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_groups, 3))

    def forward(self, grouped_features):
        weights = torch.softmax(self.logits, dim=-1).view(1, 1, -1, 3, 1, 1)
        return (grouped_features * weights).sum(dim=3)


class Conv1x1DirectionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.conv.weight, 1.0 / 3.0)
        nn.init.zeros_(self.conv.bias)

    def forward(self, grouped_features):
        batch, num_class, num_groups, _, height, width = grouped_features.shape
        fused = self.conv(grouped_features.reshape(batch * num_class * num_groups, 3, height, width))
        return fused.view(batch, num_class, num_groups, height, width)


class AttentionGatingDirectionFusion(nn.Module):
    def __init__(self, hidden_dim=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, grouped_features):
        batch, num_class, num_groups, _, height, width = grouped_features.shape
        pooled = grouped_features.mean(dim=(-1, -2)).reshape(batch * num_class * num_groups, 3)
        weights = torch.softmax(self.mlp(pooled), dim=-1)
        weights = weights.view(batch, num_class, num_groups, 3, 1, 1)
        return (grouped_features * weights).sum(dim=3)


def build_direction_fusion_block(fusion_type, num_groups):
    if fusion_type == 'mean':
        return MeanDirectionFusion()
    if fusion_type == 'weighted_sum':
        return WeightedSumDirectionFusion(num_groups=num_groups)
    if fusion_type == 'conv1x1':
        return Conv1x1DirectionFusion()
    if fusion_type == 'attention_gating':
        return AttentionGatingDirectionFusion()
    raise ValueError(f'Unsupported direction fusion {fusion_type}, expected one of {FUSION_TYPES}')


class CoarseDirectionReducer(nn.Module):
    def __init__(self, num_classes, offsets_dydx=None, fusion_type='weighted_sum'):
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f'Unsupported direction fusion {fusion_type}, expected one of {FUSION_TYPES}')

        if offsets_dydx is None:
            offsets_dydx = _resolve_default_offsets_dydx()

        self.num_classes = num_classes
        self.num_fine_directions = len(offsets_dydx)
        self.num_groups = len(COARSE_DIRECTION_GROUPS_DXDY)
        self.group_size = len(COARSE_DIRECTION_GROUPS_DXDY[0])
        self.fusion_type = fusion_type

        group_indices = torch.tensor(build_group_index_map_from_offsets(offsets_dydx), dtype=torch.long,)
        self.register_buffer('group_indices', group_indices, persistent=False)
        self.register_buffer('group_mean_vectors', compute_group_mean_vectors(), persistent=False)
        self.register_buffer('representative_angles', compute_representative_angles(), persistent=False,)

        self.fusion = build_direction_fusion_block(fusion_type=fusion_type, num_groups=self.num_groups)

    def extra_repr(self):
        return (f'num_classes={self.num_classes}, 'f'fusion_type={self.fusion_type}')

    def get_metadata(self):
        summary = summarize_coarse_direction_metadata()
        for idx, entry in enumerate(summary):
            entry['fine_channel_indices'] = self.group_indices[idx].tolist()
        return summary

    def group_features(self, fine_features):
        if fine_features.dim() != 4:
            raise ValueError(f'Expected fine_features to have shape (B, C, H, W), got {fine_features.shape}')

        batch, channels, height, width = fine_features.shape
        expected_channels = self.num_classes * self.num_fine_directions
        if channels != expected_channels:
            raise ValueError(
                f'Expected {expected_channels} input channels for {self.num_classes} classes '
                f'and {self.num_fine_directions} directions, got {channels}'
            )

        # Split channel dimension into (class, fine_direction), then gather fine directions by predefined coarse groups.
        fine_features = fine_features.view(batch, self.num_classes, self.num_fine_directions, height, width)
        grouped = fine_features.index_select(2, self.group_indices.view(-1))
        return grouped.view(batch, self.num_classes, self.num_groups, self.group_size, height, width)

    def forward(self, fine_features):
        '''
        Args:
            fine_features: Tensor of shape (B, num_classes * num_fine_directions, H, W)
        Returns:
            Tensor of shape (B, num_classes * num_groups, H, W) where num_groups is typically 8 in canonical direction order.
        '''
        grouped = self.group_features(fine_features)
        fused = self.fusion(grouped)
        batch, num_class, num_groups, height, width = fused.shape
        return fused.reshape(batch, num_class * num_groups, height, width)
