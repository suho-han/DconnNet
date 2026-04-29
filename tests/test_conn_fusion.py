import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn

import model.DconnNet as dconn_module
import train
from model.DconnNet import (
    DconnNet,
    OUTER_8_NATIVE_ORDER,
    OUTER_8_STANDARD_ORDER,
    OUTER_8_TO_STANDARD8_INDEX,
    fuse_directional_logits,
    reorder_outer8_to_standard8,
)
from scripts import train_launcher_from_config as launcher
from solver import Solver, compose_fusion_profile_loss_terms


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


def _set_identity_1x1(conv: nn.Conv2d):
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        channels = min(conv.in_channels, conv.out_channels)
        for c_idx in range(channels):
            conv.weight[c_idx, c_idx, 0, 0] = 1.0


class _SequenceLoss:
    def __init__(self, terms_sequence, label_mode='binary'):
        self._terms_sequence = list(terms_sequence)
        self.label_mode = label_mode

    def __call__(self, logits, target, return_details=False):
        assert return_details
        assert self._terms_sequence, "No more mocked loss terms available."
        terms = self._terms_sequence.pop(0)
        return terms.get('total', torch.tensor(0.0)), terms


def test_outer8_reorder_mapping_is_fixed():
    assert OUTER_8_NATIVE_ORDER == [
        (-2, -2),
        (-2, 0),
        (-2, 2),
        (0, -2),
        (0, 2),
        (2, -2),
        (2, 0),
        (2, 2),
    ]
    assert OUTER_8_STANDARD_ORDER == [
        (2, 2),
        (2, 0),
        (2, -2),
        (0, 2),
        (0, -2),
        (-2, 2),
        (-2, 0),
        (-2, -2),
    ]
    assert OUTER_8_TO_STANDARD8_INDEX == [7, 6, 5, 4, 3, 2, 1, 0]

    x = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)
    y = reorder_outer8_to_standard8(x)
    assert torch.equal(y.view(-1), torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.float32))


def test_outer8_reorder_supports_multi_group_channels():
    x = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1)
    y = reorder_outer8_to_standard8(x)
    expected = torch.tensor(
        [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
        dtype=torch.float32,
    )
    assert torch.equal(y.view(-1), expected)


def test_gate_fusion_operates_on_logits():
    c3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)
    c5 = torch.zeros((1, 8, 2, 2), dtype=torch.float32)
    gate_conv = nn.Conv2d(16, 8, kernel_size=1, bias=True)
    with torch.no_grad():
        gate_conv.weight.zero_()
        gate_conv.bias.zero_()

    fused = fuse_directional_logits(c3, c5, mode='gate', gate_conv=gate_conv)
    assert torch.allclose(fused, torch.full_like(fused, 0.5), atol=1e-6)


def test_scaled_sum_fusion_respects_residual_scale():
    c3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)
    c5 = torch.full((1, 8, 2, 2), 2.0, dtype=torch.float32)
    fused = fuse_directional_logits(c3, c5, mode='scaled_sum', residual_scale=0.2)
    assert torch.allclose(fused, torch.full_like(fused, 1.4), atol=1e-6)


def test_conv_residual_fusion_uses_residual_projection():
    c3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)
    c5 = torch.full((1, 8, 2, 2), 2.0, dtype=torch.float32)
    residual_conv = nn.Conv2d(8, 8, kernel_size=1, bias=True)
    _set_identity_1x1(residual_conv)
    fused = fuse_directional_logits(c3, c5, mode='conv_residual', residual_conv=residual_conv)
    assert torch.allclose(fused, torch.full_like(fused, 3.0), atol=1e-6)


def test_compose_fusion_profile_uses_vote_dice_affinity_only():
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
        'edge': torch.tensor(100.0),
        'bicon': torch.tensor(100.0),
        'total': torch.tensor(1000.0),
    }
    inner_terms = {
        'affinity': torch.tensor(5.0),
        'edge': torch.tensor(200.0),
        'bicon': torch.tensor(200.0),
        'total': torch.tensor(2000.0),
    }
    outer_terms = {
        'affinity': torch.tensor(7.0),
        'edge': torch.tensor(300.0),
        'bicon': torch.tensor(300.0),
        'total': torch.tensor(3000.0),
    }

    total_a, terms_a = compose_fusion_profile_loss_terms('A', 0.2, 0.05, 0.3, fused_terms, inner_terms, outer_terms)
    assert torch.isclose(total_a, torch.tensor(3.9), atol=1e-6)
    assert 'inner_affinity' not in terms_a
    assert 'outer_affinity' not in terms_a

    total_b, _ = compose_fusion_profile_loss_terms('B', 0.2, 0.05, 0.3, fused_terms, inner_terms, outer_terms)
    assert torch.isclose(total_b, torch.tensor(4.9), atol=1e-6)

    total_c, _ = compose_fusion_profile_loss_terms('C', 0.2, 0.05, 0.3, fused_terms, inner_terms, outer_terms)
    assert torch.isclose(total_c, torch.tensor(5.25), atol=1e-6)


def test_dgrf_additive_profile_loss_includes_branch_aux_and_gate():
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
        'total': torch.tensor(30.0),
    }
    inner_terms = {
        'affinity': torch.tensor(5.0),
        'total': torch.tensor(20.0),
    }
    outer_terms = {
        'affinity': torch.tensor(7.0),
        'total': torch.tensor(30.0),
    }

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_loss_profile = 'A'
    solver.fusion_lambda_inner = 0.2
    solver.fusion_lambda_outer = 0.05
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='decoder_guided',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'fusion_gate': torch.full((1, 8, 4, 4), 0.5, dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    expected = torch.tensor(3.9 + 0.3 * 20.0 + 0.2 * 30.0 + 0.01 * 0.5, dtype=torch.float32)
    assert torch.isclose(total, expected, atol=1e-6)
    assert torch.isclose(terms['dgrf_fused_main'], torch.tensor(3.9), atol=1e-6)
    assert torch.isclose(terms['dgrf_c3_aux'], torch.tensor(20.0), atol=1e-6)
    assert torch.isclose(terms['dgrf_c5_aux'], torch.tensor(30.0), atol=1e-6)
    assert torch.isclose(terms['gate'], torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(terms['total'], expected, atol=1e-6)


def test_dgrf_terms_are_not_added_for_non_dgrf_fusion():
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
    }
    inner_terms = {'affinity': torch.tensor(5.0)}
    outer_terms = {'affinity': torch.tensor(7.0)}

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_loss_profile = 'A'
    solver.fusion_lambda_inner = 0.2
    solver.fusion_lambda_outer = 0.05
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='gate',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    assert torch.isclose(total, torch.tensor(3.9), atol=1e-6)
    assert 'dgrf_fused_main' not in terms
    assert 'dgrf_c3_aux' not in terms
    assert 'dgrf_c5_aux' not in terms
    assert 'gate' not in terms
    assert torch.isclose(terms['total'], total, atol=1e-6)


def test_train_parse_args_validates_fusion_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_layout', 'standard8',
            '--conn_fusion', 'gate',
            '--fusion_loss_profile', 'A',
            '--output_dir', str(tmp_path),
        ],
    )
    args = train.parse_args()
    assert args.conn_fusion == 'gate'
    assert args.fusion_loss_profile == 'A'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '24',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--label_mode', 'dist_signed',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_layout', 'out8',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'retouch-Spectrailis',
            '--num-class', '4',
            '--conn_num', '8',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_experiment_name_is_legacy_stable_and_fusion_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--output_dir', str(tmp_path),
        ],
    )
    legacy_args = train.parse_args()
    assert train.get_experiment_output_name(legacy_args) == 'binary_8_bce'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_fusion', 'gate',
            '--fusion_loss_profile', 'A',
            '--output_dir', str(tmp_path),
        ],
    )
    fusion_args = train.parse_args()
    assert train.get_experiment_output_name(fusion_args) == 'binary_gate_A_8_bce'


def test_launcher_supports_fusion_names_and_scaled_sum_sweep():
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='binary',
        dist_aux_loss='smooth_l1',
    ) == 'binary_8_bce'
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='dist',
        dist_aux_loss='smooth_l1',
        conn_fusion='scaled_sum',
        fusion_loss_profile='C',
        fusion_residual_scale=0.3,
    ) == 'dist_scaled_sum_C_rs0.3_8_smooth_l1'

    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'conn_fusions': ['scaled_sum'],
            'fusion_loss_profiles': ['A'],
            'fusion_residual_scales': [0.1, 0.2, 0.3],
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    runs = launcher.build_multi_schedule(config, device=0, datasets=['drive'])
    assert {round(float(run['fusion_residual_scale']), 3) for run in runs} == {0.1, 0.2, 0.3}
    assert all(run['conn_fusion'] == 'scaled_sum' for run in runs)
    assert all(run['fusion_loss_profile'] == 'A' for run in runs)


def test_launcher_rejects_unsupported_label_mode():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'label_modes': ['dist_signed'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    with pytest.raises(ValueError):
        launcher.build_multi_schedule(config, device=0, datasets=['drive'])


def test_launcher_rejects_removed_decoder_fusion_keys():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'conn_fusions': ['gate'],
            'fusion_loss_profiles': ['C'],
            'decoder_fusions': ['residual_gate'],
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    with pytest.raises(ValueError, match='removed key'):
        launcher.build_multi_schedule(config, device=0, datasets=['drive'])


def test_launcher_supports_fusion_matrix_without_unintended_cartesian_product():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
            'fusion_matrix': [
                {'conn_fusion': 'gate', 'fusion_loss_profiles': ['A', 'B', 'C']},
                {'conn_fusion': 'conv_residual', 'fusion_loss_profiles': ['A', 'C']},
                {
                    'conn_fusion': 'scaled_sum',
                    'fusion_loss_profiles': ['A'],
                    'fusion_residual_scales': [0.1, 0.2, 0.3, 0.5],
                },
            ],
        },
    }
    runs = launcher.build_multi_schedule(config, device=0, datasets=['drive'])
    combos = {
        (
            str(run['conn_fusion']),
            str(run['fusion_loss_profile']),
            round(float(run['fusion_residual_scale']), 3),
        )
        for run in runs
    }
    expected = {
        ('gate', 'A', 0.2),
        ('gate', 'B', 0.2),
        ('gate', 'C', 0.2),
        ('conv_residual', 'A', 0.2),
        ('conv_residual', 'C', 0.2),
        ('scaled_sum', 'A', 0.1),
        ('scaled_sum', 'A', 0.2),
        ('scaled_sum', 'A', 0.3),
        ('scaled_sum', 'A', 0.5),
    }
    assert combos == expected
    assert ('conv_residual', 'B', 0.2) not in combos


def test_launcher_rejects_multiclass_fusion():
    config = {
        'mode': 'single',
        'dataset': 'retouch-Spectrailis',
        'device': 0,
        'single': {
            'conn_num': 8,
            'conn_layout': 'standard8',
            'conn_fusion': 'gate',
            'label_mode': 'binary',
            'folds': 3,
            'target_folds': [1],
        },
    }
    with pytest.raises(ValueError):
        launcher.build_single_schedule(config, device=0)


def test_model_output_contract_legacy_vs_fusion(monkeypatch):
    monkeypatch.setattr(dconn_module, 'resnet34', lambda pretrained=True: _DummyBackbone())
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)

    legacy_model = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='none',
    )
    fusion_model = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='gate',
    )

    legacy_model.eval()
    fusion_model.eval()
    with torch.no_grad():
        legacy_out = legacy_model(x)
        fusion_out = fusion_model(x)

    assert isinstance(legacy_out, dict)
    assert {'fused', 'aux'} <= set(legacy_out.keys())
    assert torch.is_tensor(legacy_out['fused'])
    assert torch.is_tensor(legacy_out['aux'])

    assert isinstance(fusion_out, dict)
    assert {'fused', 'inner', 'outer', 'outer_aligned', 'aux'} <= set(fusion_out.keys())
    assert torch.is_tensor(fusion_out['fused'])
    assert 'seg' not in fusion_out

    legacy_state = legacy_model.state_dict()
    assert not any(key.startswith('cls_pred_inner_conv.') for key in legacy_state)
    assert not any(key.startswith('cls_pred_outer_conv.') for key in legacy_state)
    assert not any(key.startswith('fusion_gate_conv.') for key in legacy_state)

    legacy_model_2 = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='none',
    )
    legacy_model_2.load_state_dict(legacy_state)
