import importlib

import torch

from connect_loss import resolve_connectivity_layout
from model.coarse_direction_grouping import (FUSION_TYPES, CoarseDirectionReducer, build_group_index_map_from_offsets, compute_group_mean_vectors, compute_representative_angles,
                                             summarize_coarse_direction_metadata)
from model.resnet import resnet34 as build_resnet34

EXPECTED_GROUP_INDEX_MAP = [
    [0, 23, 18],
    [1, 21, 22],
    [2, 19, 20],
    [3, 16, 14],
    [4, 15, 17],
    [5, 12, 11],
    [6, 10, 9],
    [7, 8, 13],
]

EXPECTED_MEAN_VECTORS = torch.tensor(
    [
        [5.0 / 3.0, 4.0 / 3.0],
        [1.0 / 3.0, 5.0 / 3.0],
        [-4.0 / 3.0, 5.0 / 3.0],
        [5.0 / 3.0, -1.0 / 3.0],
        [-5.0 / 3.0, 1.0 / 3.0],
        [4.0 / 3.0, -5.0 / 3.0],
        [-1.0 / 3.0, -5.0 / 3.0],
        [-5.0 / 3.0, -4.0 / 3.0],
    ],
    dtype=torch.float32,
)

EXPECTED_ANGLES = torch.tensor(
    [
        321.3401917459099,
        281.30993247402023,
        231.3401917459099,
        11.30993247402023,
        191.3099324740202,
        51.34019174590992,
        101.30993247402023,
        141.34019174590992,
    ],
    dtype=torch.float32,
)


def test_group_index_map_matches_repo_conn24_layout():
    offsets = resolve_connectivity_layout(24)['offsets']
    assert build_group_index_map_from_offsets(offsets) == EXPECTED_GROUP_INDEX_MAP


def test_mean_vectors_and_representative_angles_follow_image_coordinates():
    mean_vectors = compute_group_mean_vectors()
    representative_angles = compute_representative_angles()

    assert torch.allclose(mean_vectors, EXPECTED_MEAN_VECTORS, atol=1e-6)
    assert torch.allclose(representative_angles, EXPECTED_ANGLES, atol=1e-4)
    assert abs(representative_angles[6].item() - EXPECTED_ANGLES[6].item()) < 1e-4
    assert abs(representative_angles[7].item() - EXPECTED_ANGLES[7].item()) < 1e-4


def test_metadata_summary_keeps_canonical_direction_order():
    metadata = summarize_coarse_direction_metadata(resolve_connectivity_layout(24)['offsets'])

    assert [entry['group_name'] for entry in metadata] == ['SE', 'S', 'SW', 'E', 'W', 'NE', 'N', 'NW']
    assert metadata[0]['fine_channel_indices'] == EXPECTED_GROUP_INDEX_MAP[0]
    assert abs(metadata[6]['representative_angle_deg'] - EXPECTED_ANGLES[6].item()) < 1e-4
    assert abs(metadata[7]['representative_angle_deg'] - EXPECTED_ANGLES[7].item()) < 1e-4


def test_all_fusion_modes_reduce_24_to_8_per_class():
    x = torch.randn(2, 48, 5, 6)

    for fusion_type in FUSION_TYPES:
        reducer = CoarseDirectionReducer(num_classes=2, fusion_type=fusion_type)
        out = reducer(x)
        assert out.shape == (2, 16, 5, 6)
        assert torch.isfinite(out).all()


def test_weighted_conv_and_attention_start_from_mean_like_behavior():
    x = torch.randn(2, 24, 4, 5)
    mean_reducer = CoarseDirectionReducer(num_classes=1, fusion_type='mean')
    mean_out = mean_reducer(x)

    for fusion_type in ('weighted_sum', 'conv1x1', 'attention_gating'):
        reducer = CoarseDirectionReducer(num_classes=1, fusion_type=fusion_type)
        out = reducer(x)
        assert torch.allclose(out, mean_out, atol=1e-6, rtol=1e-6)


def test_reducer_outputs_canonical8_order_directly():
    x = torch.zeros((1, 24, 1, 1), dtype=torch.float32)
    for fine_idx in EXPECTED_GROUP_INDEX_MAP[0]:
        x[0, fine_idx, 0, 0] = 1.0

    reducer = CoarseDirectionReducer(
        num_classes=1,
        fusion_type='mean',
    )
    out = reducer(x).view(1, 1, 8, 1, 1)

    assert torch.isclose(out[0, 0, 0, 0, 0], torch.tensor(1.0))
    assert torch.count_nonzero(out) == 1


def test_grouped_dconnnet_path_returns_canonical_8_direction_outputs(monkeypatch):
    dconnnet_module = importlib.import_module('model.DconnNet')
    monkeypatch.setattr(
        dconnnet_module,
        'resnet34',
        lambda pretrained=True, **kwargs: build_resnet34(pretrained=False, **kwargs),
    )

    model = dconnnet_module.DconnNet(
        num_class=1,
        conn_num=8,
        direction_grouping='coarse24to8',
        direction_fusion='weighted_sum',
    ).eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        main_out, aux_out = model(x)

    assert main_out.shape == (1, 8, 64, 64)
    assert aux_out.shape == (1, 8, 64, 64)


def test_grouped_dconnnet_path_rejects_non8_final_branch():
    dconnnet_module = importlib.import_module('model.DconnNet')
    try:
        dconnnet_module.DconnNet(
            num_class=1,
            conn_num=24,
            direction_grouping='coarse24to8',
        )
    except ValueError as exc:
        assert 'requires conn_num=8' in str(exc)
    else:
        raise AssertionError('Expected grouped DconnNet path to reject conn_num=24')
