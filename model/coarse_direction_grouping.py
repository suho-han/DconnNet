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
COARSE_DIRECTION_GROUP_SIZE = len(COARSE_DIRECTION_GROUPS_DXDY[0])

FUSION_TYPES = (
    'mean',
    'weighted_sum',
    'conv1x1',
    'attention_gating',
)


def build_group_index_map_from_offsets(offsets_dydx):
    offset_to_index = {(dx, dy): idx for idx, (dy, dx) in enumerate(offsets_dydx)}
    group_index_map = []
    for group in COARSE_DIRECTION_GROUPS_DXDY:
        try:
            group_index_map.append([offset_to_index[offset] for offset in group])
        except KeyError as exc:
            raise ValueError(f'Offset {exc.args[0]} is missing from the provided 24-channel layout') from exc
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


class DirectionFusion(nn.Module):
    def __init__(self, fusion_type, num_groups):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'weighted_sum':
            self.logits = nn.Parameter(torch.zeros(num_groups, COARSE_DIRECTION_GROUP_SIZE))
        elif fusion_type == 'conv1x1':
            self.conv = nn.Conv2d(COARSE_DIRECTION_GROUP_SIZE, 1, kernel_size=1, bias=True)
            nn.init.constant_(self.conv.weight, 1.0 / COARSE_DIRECTION_GROUP_SIZE)
            nn.init.zeros_(self.conv.bias)
        elif fusion_type == 'attention_gating':
            hidden_dim = COARSE_DIRECTION_GROUP_SIZE * 2
            self.mlp = nn.Sequential(
                nn.Linear(COARSE_DIRECTION_GROUP_SIZE, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, COARSE_DIRECTION_GROUP_SIZE),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        elif fusion_type != 'mean':
            raise ValueError(f'Unsupported direction fusion {fusion_type}, expected one of {FUSION_TYPES}')

    def forward(self, grouped_features, return_maps=False):
        if self.fusion_type == 'mean':
            fused = grouped_features.mean(dim=3)
            if return_maps:
                return fused, {}
            return fused

        batch, num_class, num_groups, _, height, width = grouped_features.shape

        if self.fusion_type == 'weighted_sum':
            weights = torch.softmax(self.logits, dim=-1).view(1, 1, -1, COARSE_DIRECTION_GROUP_SIZE, 1, 1)
            fused = (grouped_features * weights).sum(dim=3)
            if return_maps:
                return fused, {'logits': self.logits, 'weight_map': weights}
            return fused

        if self.fusion_type == 'conv1x1':
            fused = self.conv(grouped_features.reshape(batch * num_class * num_groups, COARSE_DIRECTION_GROUP_SIZE, height, width))
            conv_map = fused.view(batch, num_class, num_groups, height, width)
            if return_maps:
                return conv_map, {'conv_map': conv_map}
            return conv_map

        pooled = grouped_features.mean(dim=(-1, -2)).reshape(batch * num_class * num_groups, COARSE_DIRECTION_GROUP_SIZE)
        weights = torch.softmax(self.mlp(pooled), dim=-1)
        weights = weights.view(batch, num_class, num_groups, COARSE_DIRECTION_GROUP_SIZE, 1, 1)
        fused = (grouped_features * weights).sum(dim=3)
        if return_maps:
            return fused, {'attention_map': weights}
        return fused


class CoarseDirectionReducer(nn.Module):
    def __init__(self, num_classes, offsets_dydx=None, fusion_type='weighted_sum'):
        super().__init__()
        if offsets_dydx is None:
            offsets_dydx = _resolve_default_offsets_dydx()

        self.num_classes = num_classes
        self.num_fine_directions = len(offsets_dydx)

        group_indices = torch.tensor(build_group_index_map_from_offsets(offsets_dydx), dtype=torch.long)
        self.register_buffer('group_indices', group_indices, persistent=False)
        self.fusion = DirectionFusion(fusion_type=fusion_type, num_groups=group_indices.shape[0])

    def get_metadata(self):
        summary = summarize_coarse_direction_metadata()
        for idx, entry in enumerate(summary):
            entry['fine_channel_indices'] = self.group_indices[idx].tolist()
        return summary

    def group_features(self, fine_features):
        # Fine feature (n^2-1 channels) -> Groups (num_groups x group_size channels)
        # ex) 5^2-1=24 channels -> 8 groups x 3 channels
        if fine_features.dim() != 4:
            raise ValueError(f'Expected fine_features to have shape (B, C, H, W), got {fine_features.shape}')

        batch, channels, height, width = fine_features.shape
        expected_channels = self.num_classes * self.num_fine_directions
        if channels != expected_channels:
            raise ValueError(
                f'Expected {expected_channels} input channels for {self.num_classes} classes '
                f'and {self.num_fine_directions} directions, got {channels}'
            )

        num_groups, group_size = self.group_indices.shape
        fine_features = fine_features.view(batch, self.num_classes, self.num_fine_directions, height, width)
        grouped = fine_features.index_select(2, self.group_indices.view(-1))
        return grouped.view(batch, self.num_classes, num_groups, group_size, height, width)

    def forward(self, fine_features, return_maps=False):
        grouped = self.group_features(fine_features)
        if return_maps:
            fused, fusion_maps = self.fusion(grouped, return_maps=True)
            fused = fused.reshape(fused.shape[0], -1, *fused.shape[-2:])
            return fused, fusion_maps

        fused = self.fusion(grouped)
        return fused.reshape(fused.shape[0], -1, *fused.shape[-2:])
