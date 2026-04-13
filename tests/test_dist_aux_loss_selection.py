from types import SimpleNamespace

import torch

from connect_loss import connect_loss


def _patch_cuda_to_noop(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self, raising=False)


def _build_shift_matrix(size):
    matrix = torch.zeros((1, 1, size, size), dtype=torch.float32)
    for i in range(size - 1):
        matrix[0, 0, i, i + 1] = 1.0
    return matrix


def _make_args(dataset='chase', dist_aux_loss='smooth_l1', gamma=1.0):
    return SimpleNamespace(
        num_class=1,
        dataset=dataset,
        dist_aux_loss=dist_aux_loss,
        dist_sf_l1_gamma=gamma,
    )


def _make_loss_module(dataset='chase', dist_aux_loss='smooth_l1', gamma=1.0, size=4):
    args = _make_args(dataset=dataset, dist_aux_loss=dist_aux_loss, gamma=gamma)
    return connect_loss(
        args,
        _build_shift_matrix(size),
        _build_shift_matrix(size),
        label_mode='dist_signed',
        conn_num=8,
        sigma=2.0,
    )


def test_dist_aux_regression_loss_is_finite_for_both_modes(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    pred = torch.full((2, 8, 4, 4), 0.4, dtype=torch.float32)
    target = torch.full((2, 8, 4, 4), 0.6, dtype=torch.float32)

    smooth_module = _make_loss_module(dist_aux_loss='smooth_l1')
    gjml_module = _make_loss_module(dist_aux_loss='gjml_sf_l1')

    smooth_loss = smooth_module.dist_aux_regression_loss(pred, target)
    gjml_loss = gjml_module.dist_aux_regression_loss(pred, target)

    assert torch.isfinite(smooth_loss)
    assert torch.isfinite(gjml_loss)


def test_gjml_sf_l1_is_near_zero_when_prediction_matches_target(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dist_aux_loss='gjml_sf_l1')
    target = torch.linspace(0.1, 0.9, steps=2 * 8 * 4 * 4, dtype=torch.float32).view(2, 8, 4, 4)
    loss = module.dist_aux_regression_loss(target, target)
    assert float(loss.item()) < 1e-6


def test_chase_dist_mode_keeps_bicon_zero(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dataset='chase', dist_aux_loss='gjml_sf_l1')
    affinity_map = torch.zeros((1, 8, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[:, :, 1:3, 1:3] = 0.8

    _, loss_dict = module(affinity_map, target, return_details=True)

    assert float(loss_dict['bicon'].item()) == 0.0


def test_non_chase_dist_mode_computes_bicon(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dataset='isic', dist_aux_loss='gjml_sf_l1')
    affinity_map = torch.zeros((1, 8, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[:, :, 1:3, 1:3] = 0.8

    _, loss_dict = module(affinity_map, target, return_details=True)

    assert float(loss_dict['bicon'].item()) > 0.0
