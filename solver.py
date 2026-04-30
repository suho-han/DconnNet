import csv
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    print(
        '[WARN] TensorBoard backend is unavailable; '
        'install `tensorboard` in the active environment to enable TB logging.'
    )

    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from connect_loss import Bilateral_voting, Bilateral_voting_kxk, resolve_connectivity_layout
from lr_update import get_lr
from metrics.cal_betti import getBettiErrors
from metrics.cldice import clDice
from src.losses import compose_fusion_profile_loss_terms as _compose_fusion_profile_loss_terms
from src.losses.factory import build_loss_functions
from src.metrics import compute_binary_precision_accuracy, compute_multiclass_precision_accuracy
from src.metrics import get_mask as get_mask_from_logits
from src.metrics import one_hot as one_hot_encode
from src.metrics import per_class_dice as compute_per_class_dice
from src.scripts import run_test_only_eval, run_validation_epoch
from src.utils.monitoring import EarlyStopping as _EarlyStopping
from src.utils.monitoring import build_epoch_postfix, format_elapsed_hms, is_nan_metric
from src.utils.results import create_exp_directory as _create_exp_directory
from src.utils.results import write_epoch_result_row as _write_epoch_result_row
from src.utils.results import write_eval_summary as _write_eval_summary

from model.DconnNet import normalize_conn_fusion_mode

try:
    from apex import amp
except Exception:
    class _AmpFallback:
        def initialize(self, model, optim, opt_level='O2'):
            return model, optim

        @contextmanager
        def scale_loss(self, loss, optimizer):
            yield loss

    amp = _AmpFallback()


def compose_fusion_profile_loss_terms(
    profile,
    lambda_inner,
    lambda_outer,
    lambda_fused,
    fused_terms,
    inner_terms,
    outer_terms,
):
    return _compose_fusion_profile_loss_terms(
        profile=profile,
        lambda_inner=lambda_inner,
        lambda_outer=lambda_outer,
        lambda_fused=lambda_fused,
        fused_terms=fused_terms,
        inner_terms=inner_terms,
        outer_terms=outer_terms,
    )


class EarlyStopping(_EarlyStopping):
    pass


class Solver(object):
    def __init__(self, args, optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        self.conn_fusion = normalize_conn_fusion_mode(getattr(self.args, 'conn_fusion', 'none'))
        self.fusion_enabled = self.conn_fusion != 'none'
        self.fusion_loss_profile = str(getattr(self.args, 'fusion_loss_profile', 'A')).upper()
        self.fusion_lambda_inner = float(getattr(self.args, 'fusion_lambda_inner', 0.2))
        self.fusion_lambda_outer = float(getattr(self.args, 'fusion_lambda_outer', 0.05))
        self.fusion_lambda_fused = float(getattr(self.args, 'fusion_lambda_fused', 0.3))
        self.connectivity_layout = resolve_connectivity_layout(
            self.args.conn_num,
            getattr(self.args, 'conn_layout', None),
        )
        self.conn_channels = self.connectivity_layout['channel_count']
        if self.fusion_enabled:
            if self.NumClass != 1:
                raise ValueError("conn_fusion currently supports only num_class=1")
            if self.conn_channels != 8:
                raise ValueError("conn_fusion currently supports only conn_num=8")
            if self.connectivity_layout['name'] != 'standard8':
                raise ValueError("conn_fusion currently supports only conn_layout=standard8")
        H, W = args.resize

        self.hori_translation = torch.zeros([1, self.NumClass, W, W])
        for i in range(W - 1):
            self.hori_translation[:, :, i, i + 1] = torch.tensor(1.0)

        self.verti_translation = torch.zeros([1, self.NumClass, H, H])
        for j in range(H - 1):
            self.verti_translation[:, :, j, j + 1] = torch.tensor(1.0)

        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def create_exp_directory(self):
        _create_exp_directory(self.args.save)

    def _format_elapsed_hms(self, elapsed_seconds):
        return format_elapsed_hms(elapsed_seconds)

    def _write_epoch_result_row(self, epoch, metrics, elapsed_hms=''):
        _write_epoch_result_row(self.args.save, epoch, metrics, elapsed_hms)

    def _write_eval_summary(
        self,
        split_name,
        metrics,
        checkpoint_name='',
        evaluated_split='',
        eval_epoch='',
        elapsed_hms='',
    ):
        _write_eval_summary(
            save_dir=self.args.save,
            split_name=split_name,
            metrics=metrics,
            checkpoint_name=checkpoint_name,
            evaluated_split=evaluated_split,
            eval_epoch=eval_epoch,
            elapsed_hms=elapsed_hms,
        )

    def _compute_binary_precision_accuracy(self, pred_mask, gt_mask, eps=1e-6):
        return compute_binary_precision_accuracy(pred_mask, gt_mask, eps=eps)

    def _compute_multiclass_precision_accuracy(self, pred_label, true_label, num_class, eps=1e-6):
        return compute_multiclass_precision_accuracy(pred_label, true_label, num_class, eps=eps)

    def save_checkpoint_batch_triplet(self, images, preds, masks, epoch, batch_idx):
        base_dir = os.path.join(
            self.args.save, 'models', 'checkpoint_batches', f'epoch_{epoch:03d}'
        )
        image_dir = os.path.join(base_dir, 'image')
        pred_dir = os.path.join(base_dir, 'pred')
        mask_dir = os.path.join(base_dir, 'mask')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        image_path = os.path.join(image_dir, f'batch_{batch_idx:04d}.png')
        pred_path = os.path.join(pred_dir, f'batch_{batch_idx:04d}.png')
        mask_path = os.path.join(mask_dir, f'batch_{batch_idx:04d}.png')

        utils.save_image(images.detach().cpu(), image_path, normalize=True, scale_each=True)
        utils.save_image(preds.detach().cpu().float(), pred_path, normalize=True, scale_each=True)
        utils.save_image(masks.detach().cpu().float(), mask_path, normalize=True, scale_each=True)

    def _normalize_sample_names(self, raw_name, batch_size, batch_idx):
        if raw_name is None:
            return [f"batch_{batch_idx:04d}_idx_{i:02d}" for i in range(batch_size)]

        if isinstance(raw_name, str):
            return [raw_name] if batch_size == 1 else [f"{raw_name}_{i}" for i in range(batch_size)]

        if torch.is_tensor(raw_name):
            flat = raw_name.detach().cpu().view(-1).tolist()
            names = [str(v) for v in flat]
            if len(names) < batch_size:
                names += [f"batch_{batch_idx:04d}_idx_{i:02d}" for i in range(len(names), batch_size)]
            return names[:batch_size]

        if isinstance(raw_name, (list, tuple)):
            names = [str(v) for v in raw_name]
            if len(names) < batch_size:
                names += [f"batch_{batch_idx:04d}_idx_{i:02d}" for i in range(len(names), batch_size)]
            return names[:batch_size]

        return [str(raw_name)] if batch_size == 1 else [f"{str(raw_name)}_{i}" for i in range(batch_size)]

    def _unpack_test_batch(self, test_data, batch_idx):
        sample_name_source = None
        binary_gt = None

        if isinstance(test_data, dict):
            X_test = test_data['image']
            y_test_raw = test_data['label']
            binary_gt = test_data.get('binary_gt', None)
            sample_name_source = test_data.get('name', None)
        elif isinstance(test_data, (list, tuple)):
            if len(test_data) < 2:
                raise ValueError('test_data must contain at least image and label tensors')

            if len(test_data) >= 3 and (not torch.is_tensor(test_data[0])) and torch.is_tensor(test_data[1]) and torch.is_tensor(test_data[2]):
                X_test = test_data[1]
                y_test_raw = test_data[2]
                sample_name_source = test_data[0]
            else:
                X_test = test_data[0]
                y_test_raw = test_data[1]
                if len(test_data) >= 3:
                    sample_name_source = test_data[2]
        else:
            raise TypeError('test_data must be a dict or tuple/list from DataLoader')

        if not torch.is_tensor(X_test) or not torch.is_tensor(y_test_raw):
            raise TypeError('image and label in test_data must be tensors')

        if binary_gt is not None:
            binary_gt = binary_gt.cuda()

        batch_size = int(X_test.shape[0])
        sample_names = self._normalize_sample_names(sample_name_source, batch_size, batch_idx)
        return X_test, y_test_raw, sample_names, binary_gt

    def dist_to_binary(self, x, label_mode):
        """
        Convert a distance-label ground truth tensor to a binary mask.

        Expected input shapes:
            - (B, C, 8, H, W)
            - or any tensor where foreground/background thresholding is elementwise.

        IMPORTANT:
            If your signed distance map defines foreground as negative values,
            change the 'dist' branch from (x > 0) to (x < 0).
        """
        if label_mode in ['dist', 'dist_inverted']:
            return (x > 0).float()

        else:
            return (x > 0.5).float()

    def dist_score_to_binary(self, x):
        """
        Convert the distance-mode prediction score map to a binary mask.

        In the dist path, `pred_score` is produced from sigmoid affinities via
        bilateral voting, so it is non-negative almost everywhere. Thresholding
        at `> 0` collapses to all-foreground masks; use a probability-style
        threshold instead.
        """
        return (x > 0.5).float()

    def apply_connectivity_voting(self, conn_map, hori_translation, verti_translation):
        """
        Convert directional affinity / connectivity maps to a voted score map.
        """
        layout = self.connectivity_layout
        if layout['name'] == 'standard8':
            pred_map, vote_map = Bilateral_voting(conn_map, hori_translation, verti_translation)
        else:
            pred_map, vote_map = Bilateral_voting_kxk(
                conn_map,
                hori_translation,
                verti_translation,
                conn_num=layout['kernel_size'],
                offsets=layout['offsets'],
            )
        return pred_map, vote_map

    def connectivity_to_mask(self, conn_map, hori_translation, verti_translation):
        """
        Convert connectivity map to final binary segmentation mask
        using bilateral voting.
        """
        pred_mask, _ = self.apply_connectivity_voting(
            conn_map, hori_translation, verti_translation
        )
        pred_mask = (pred_mask > 0).float()
        return pred_mask

    def _unpack_model_outputs(self, model_output):
        if isinstance(model_output, dict):
            if self.fusion_enabled:
                required = {'fused', 'inner', 'outer'}
                missing = sorted(required - set(model_output.keys()))
                if missing:
                    raise ValueError(f"Fusion model output is missing keys: {missing}")
            else:
                required = {'fused', 'aux'}
                missing = sorted(required - set(model_output.keys()))
                if missing:
                    raise ValueError(f"Baseline model output is missing keys: {missing}")

            return model_output

        if isinstance(model_output, (tuple, list)) and len(model_output) == 2:
            return {
                'fused': model_output[0],
                'aux': model_output[1],
            }

        raise TypeError(
            "Unexpected model output type. Expected tuple(len=2) for legacy mode "
            "or dict with keys {fused, inner, outer} for fusion mode."
        )

    def _compute_fusion_profile_loss(self, output_dict, target, binary_gt=None, collect_edge_stats=False):
        if not self.fusion_enabled:
            raise RuntimeError("_compute_fusion_profile_loss can be used only when conn_fusion is enabled")
        if not hasattr(self, 'loss_func_outer'):
            raise RuntimeError("Fusion outer loss function is not initialized")

        fused_logits = output_dict['fused']
        inner_logits = output_dict['inner']
        outer_logits = output_dict['outer']

        if (
            collect_edge_stats
            and self.loss_func.label_mode in ['dist', 'dist_inverted']
            and hasattr(self.loss_func, 'set_dist_edge_stat_collection')
        ):
            self.loss_func.set_dist_edge_stat_collection(True)
        _, fused_terms = self.loss_func(fused_logits, target, return_details=True)
        if (
            collect_edge_stats
            and self.loss_func.label_mode in ['dist', 'dist_inverted']
            and hasattr(self.loss_func, 'set_dist_edge_stat_collection')
        ):
            self.loss_func.set_dist_edge_stat_collection(False)

        _, inner_terms = self.loss_func(inner_logits, target, return_details=True)
        _, outer_terms = self.loss_func_outer(outer_logits, target, return_details=True)

        total, terms = compose_fusion_profile_loss_terms(
            profile=self.fusion_loss_profile,
            lambda_inner=self.fusion_lambda_inner,
            lambda_outer=self.fusion_lambda_outer,
            lambda_fused=self.fusion_lambda_fused,
            fused_terms=fused_terms,
            inner_terms=inner_terms,
            outer_terms=outer_terms,
        )

        active_conn_fusion = getattr(
            self,
            'conn_fusion',
            normalize_conn_fusion_mode(getattr(self.args, 'conn_fusion', 'none')),
        )
        if active_conn_fusion in {'dg', 'dg_direct'}:
            base_fused_total = total
            c3_aux = inner_terms.get('total', inner_terms['affinity'])
            c5_aux = outer_terms.get('total', outer_terms['affinity'])
            c3_weight = float(getattr(self.args, 'conn_aux_c3_weight', 0.0))
            c5_weight = float(getattr(self.args, 'conn_aux_c5_weight', 0.0))
            total = base_fused_total + c3_weight * c3_aux + c5_weight * c5_aux
            terms['dgrf_fused_main'] = base_fused_total
            terms['dgrf_c3_aux'] = c3_aux
            terms['dgrf_c5_aux'] = c5_aux
            if 'fusion_gate' in output_dict:
                gate_loss = output_dict['fusion_gate'].mean()
                total = total + float(getattr(self.args, 'fusion_gate_reg_weight', 0.0)) * gate_loss
                terms['gate'] = gate_loss

        if getattr(self.args, 'use_seg_aux', False) and 'mask_logit' in output_dict and binary_gt is not None:
            if binary_gt.ndim == 3:
                binary_gt = binary_gt.unsqueeze(1)
            seg_loss = F.binary_cross_entropy_with_logits(output_dict['mask_logit'], binary_gt)
            total = total + getattr(self.args, 'seg_aux_weight', 0.0) * seg_loss
            terms['seg_aux'] = seg_loss

        terms['total'] = total

        return total, terms

    def _resolve_training_runtime_configs(self):
        monitoring_config = getattr(self.args, 'monitoring_config', {}) or {}
        early_stopping_config = getattr(self.args, 'early_stopping_config', {}) or {}
        checkpoint_config = getattr(self.args, 'checkpoint_config', {}) or {}

        monitor_metric = monitoring_config.get(
            'monitor_metric',
            getattr(self.args, 'monitor_metric', 'val_dice'),
        )
        if monitor_metric not in ('val_dice', 'val_loss'):
            raise ValueError(
                f"Unsupported monitor_metric={monitor_metric}; supported: val_dice, val_loss"
            )

        save_best_only = bool(
            checkpoint_config.get(
                'save_best_only',
                getattr(self.args, 'save_best_only', False),
            )
        )
        save_per_epochs = int(
            checkpoint_config.get(
                'save_per_epochs',
                getattr(self.args, 'save_per_epochs', 50),
            )
        )
        if save_per_epochs < 1:
            raise ValueError('save_per_epochs must be >= 1')

        early_stopping_kwargs = {
            'patience': int(
                early_stopping_config.get(
                    'patience',
                    getattr(self.args, 'early_stopping_patience', 20),
                )
            ),
            'min_delta': float(
                early_stopping_config.get(
                    'min_delta',
                    getattr(self.args, 'early_stopping_min_delta', 0.001),
                )
            ),
            'tie_break_with_loss': bool(
                early_stopping_config.get(
                    'tie_break_with_loss',
                    getattr(self.args, 'tie_break_with_loss', True),
                )
            ),
            'tie_eps': float(
                early_stopping_config.get(
                    'tie_eps',
                    getattr(self.args, 'early_stopping_tie_eps', 1e-4),
                )
            ),
            'stop_interval': int(
                early_stopping_config.get(
                    'stop_interval',
                    getattr(self.args, 'early_stopping_stop_interval', 10),
                )
            ),
        }
        return monitor_metric, save_best_only, save_per_epochs, early_stopping_kwargs

    def train(self, model, train_loader, val_loader, num_epochs=10, label_mode='binary', test_loader=None):
        optim = self.optim(model.parameters(), lr=self.lr)

        print('START TRAIN.')
        self.create_exp_directory()
        tb_dir = os.path.join(self.args.save, 'tensorboard', f'exp')
        writer = SummaryWriter(log_dir=tb_dir)

        self.loss_func, self.loss_func_outer = build_loss_functions(
            args=self.args,
            hori_translation=self.hori_translation,
            verti_translation=self.verti_translation,
            label_mode=label_mode,
            fusion_enabled=self.fusion_enabled,
        )

        net, optimizer = amp.initialize(model, optim, opt_level='O2')

        monitor_metric, save_best_only, save_per_epochs, early_stopping_kwargs = (
            self._resolve_training_runtime_configs()
        )
        monitor_mode = 'max' if monitor_metric == 'val_dice' else 'min'
        early_stopper = None
        if val_loader is not None:
            early_stopper = EarlyStopping(
                monitor_metric=monitor_metric,
                mode=monitor_mode,
                **early_stopping_kwargs,
            )

        scheduled_modes = ['CosineAnnealingWarmRestarts']
        if self.args.lr_update in scheduled_modes:
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=15, T_mult=2, eta_min=0.00001
                )
        else:
            scheduled = False

        if self.args.test_only:
            run_test_only_eval(self, net, val_loader, test_loader, writer)
        else:
            global_step = 0
            train_start_time = time.perf_counter()
            completed_epoch = 0
            dataset_name = str(getattr(self.args, 'dataset', 'unknown'))
            output_dir_name = str(getattr(self.args, 'output_dir', ''))
            if output_dir_name == '':
                output_dir_name = str(getattr(self.args, 'save', 'output'))
            progress_prefix = f'({dataset_name}) *({output_dir_name})'
            epoch_progress = tqdm(
                range(self.args.epochs),
                desc=f'{progress_prefix} Epoch',
                dynamic_ncols=True,
                leave=True,
            )
            epoch_metrics = {
                'dice': float('nan'),
                'jac': float('nan'),
                'cldice': float('nan'),
                'precision': float('nan'),
                'accuracy': float('nan'),
                'betti_error_0': float('nan'),
                'betti_error_1': float('nan'),
                'val_loss': float('nan'),
                'train_loss': float('nan'),
                'val_loss_terms': {},
            }
            for epoch in epoch_progress:
                current_epoch = epoch + 1
                completed_epoch = current_epoch
                net.train()
                train_loss_epoch = []
                epoch_progress.set_description(
                    f'{progress_prefix} Epoch {current_epoch}/{self.args.epochs}'
                )
                if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'reset_dist_edge_stats'):
                    self.loss_func.reset_dist_edge_stats()

                if scheduled:
                    scheduler.step()
                else:
                    curr_lr = get_lr(
                        self.lr,
                        self.args.lr_update,
                        epoch,
                        num_epochs,
                        gamma=self.args.gamma,
                        step=self.args.lr_step
                    )
                    for param_group in optim.param_groups:
                        param_group['lr'] = curr_lr

                with tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f'{progress_prefix} Train',
                    dynamic_ncols=True,
                    leave=False,
                ) as train_progress:
                    for i_batch, sample_batched in train_progress:
                        if isinstance(sample_batched, dict):
                            X = sample_batched['image']
                            y = sample_batched['label']
                            binary_gt = sample_batched.get('binary_gt', None)
                        else:
                            X = sample_batched[0]
                            y = sample_batched[1]
                            binary_gt = None

                        X = X.cuda()
                        y = y.float().cuda()
                        if binary_gt is not None:
                            binary_gt = binary_gt.cuda()

                        optim.zero_grad()
                        model_output = self._unpack_model_outputs(net(X))

                        if self.fusion_enabled:
                            loss, loss_main_dict = self._compute_fusion_profile_loss(
                                model_output,
                                y,
                                binary_gt=binary_gt,
                                collect_edge_stats=True,
                            )
                            loss_aux_dict = {}
                        else:
                            output = model_output['fused']
                            aux_out = model_output['aux']
                            if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'set_dist_edge_stat_collection'):
                                self.loss_func.set_dist_edge_stat_collection(True)
                            loss_main, loss_main_dict = self.loss_func(output, y, return_details=True)
                            if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'set_dist_edge_stat_collection'):
                                self.loss_func.set_dist_edge_stat_collection(False)
                            loss_aux, loss_aux_dict = self.loss_func(aux_out, y, return_details=True)
                            loss = loss_main + 0.3 * loss_aux

                            if getattr(self.args, 'use_seg_aux', False) and 'mask_logit' in model_output and binary_gt is not None:
                                if binary_gt.ndim == 3:
                                    binary_gt = binary_gt.unsqueeze(1)
                                seg_loss = F.binary_cross_entropy_with_logits(model_output['mask_logit'], binary_gt)
                                loss = loss + getattr(self.args, 'seg_aux_weight', 0.0) * seg_loss
                                loss_main_dict['seg_aux'] = seg_loss

                        with amp.scale_loss(loss, optimizer) as scale_loss:
                            scale_loss.backward()

                        optim.step()
                        train_loss_epoch.append(loss.item())
                        writer.add_scalar('train/loss_total', float(loss.item()), global_step)
                        for key, value in loss_main_dict.items():
                            writer.add_scalar(f'train/main/{key}', float(value.detach().item()), global_step)
                        for key, value in loss_aux_dict.items():
                            writer.add_scalar(f'train/aux/{key}', float(value.detach().item()), global_step)
                        global_step += 1

                        train_progress.set_postfix(
                            loss=f'{float(loss.item()):.4f}',
                            lr=f'{float(optim.param_groups[0]["lr"]):.2e}',
                        )

                mean_train_loss = float(np.mean(train_loss_epoch)) if train_loss_epoch else float('nan')
                should_save_batch_triplet = ((epoch + 1) % save_per_epochs == 0)

                epoch_metrics = {
                    'dice': float('nan'),
                    'jac': float('nan'),
                    'cldice': float('nan'),
                    'precision': float('nan'),
                    'accuracy': float('nan'),
                    'betti_error_0': float('nan'),
                    'betti_error_1': float('nan'),
                    'val_loss': float('nan'),
                    'train_loss': mean_train_loss,
                    'val_loss_terms': {},
                }

                stop_training = False
                if val_loader is not None:
                    epoch_metrics, stop_training = run_validation_epoch(
                        solver=self,
                        net=net,
                        model=model,
                        optimizer=optimizer,
                        val_loader=val_loader,
                        epoch=epoch,
                        current_epoch=current_epoch,
                        mean_train_loss=mean_train_loss,
                        save_batch_triplet=should_save_batch_triplet,
                        monitor_metric=monitor_metric,
                        early_stopper=early_stopper,
                    )

                if (not save_best_only) and (current_epoch % save_per_epochs == 0):
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.args.save, 'models', str(current_epoch) + '_model.pth')
                    )

                if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'get_dist_edge_stats'):
                    dist_edge_stats = self.loss_func.get_dist_edge_stats()
                else:
                    dist_edge_stats = None
                writer.add_scalar('epoch/train_loss', float(mean_train_loss), current_epoch)
                if not is_nan_metric(epoch_metrics['val_loss']):
                    writer.add_scalar('epoch/val_loss', float(epoch_metrics['val_loss']), current_epoch)
                if not is_nan_metric(epoch_metrics['dice']):
                    writer.add_scalar('epoch/dice', float(epoch_metrics['dice']), current_epoch)
                if not is_nan_metric(epoch_metrics['jac']):
                    writer.add_scalar('epoch/jac', float(epoch_metrics['jac']), current_epoch)
                if not is_nan_metric(epoch_metrics['cldice']):
                    writer.add_scalar('epoch/cldice', float(epoch_metrics['cldice']), current_epoch)
                if not is_nan_metric(epoch_metrics['precision']):
                    writer.add_scalar('epoch/precision', float(epoch_metrics['precision']), current_epoch)
                if not is_nan_metric(epoch_metrics['accuracy']):
                    writer.add_scalar('epoch/accuracy', float(epoch_metrics['accuracy']), current_epoch)
                if not is_nan_metric(epoch_metrics['betti_error_0']):
                    writer.add_scalar(
                        'epoch/betti_error_0',
                        float(epoch_metrics['betti_error_0']),
                        current_epoch,
                    )
                if not is_nan_metric(epoch_metrics['betti_error_1']):
                    writer.add_scalar(
                        'epoch/betti_error_1',
                        float(epoch_metrics['betti_error_1']),
                        current_epoch,
                    )
                for key, value in epoch_metrics.get('val_loss_terms', {}).items():
                    writer.add_scalar(f'val/{key}', float(value), current_epoch)
                if early_stopper is not None and val_loader is not None:
                    if not is_nan_metric(early_stopper.best_val_dice):
                        writer.add_scalar('epoch/best_val_dice', float(early_stopper.best_val_dice), current_epoch)
                    writer.add_scalar('epoch/early_stopping_counter', float(early_stopper.counter), current_epoch)
                if dist_edge_stats is not None:
                    writer.add_scalar('train/edge_mean', float(dist_edge_stats['edge_mean']), current_epoch)
                    writer.add_scalar('train/edge_nonzero_ratio', float(dist_edge_stats['edge_nonzero_ratio']), current_epoch)

                epoch_postfix = build_epoch_postfix(
                    mean_train_loss=mean_train_loss,
                    val_loader=val_loader,
                    epoch_metrics=epoch_metrics,
                    early_stopper=early_stopper,
                    dist_edge_stats=dist_edge_stats,
                )
                epoch_progress.set_postfix(epoch_postfix)
                elapsed_hms = self._format_elapsed_hms(time.perf_counter() - train_start_time)
                self._write_epoch_result_row(current_epoch, epoch_metrics, elapsed_hms)
                if stop_training:
                    break
            epoch_progress.close()

            if completed_epoch == 0:
                raise ValueError('Training loop completed zero epochs; please set --epochs >= 1.')

            final_eval_split = 'test' if test_loader is not None else 'val'
            final_eval_loader = test_loader if test_loader is not None else val_loader
            if final_eval_loader is None:
                raise ValueError('Final evaluation requires test_loader or val_loader, but both are None.')
            best_model_path = os.path.join(self.args.save, 'models', 'best_model.pth')
            final_checkpoint_name = ''
            if val_loader is not None and os.path.exists(best_model_path):
                print(f'LOAD BEST MODEL FOR FINAL {final_eval_split.upper()} EVAL: {best_model_path}')
                net.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
                net = net.cuda()
                final_checkpoint_name = 'best_model.pth'
            else:
                if val_loader is None:
                    print(f'VAL LOADER IS NONE; SKIP BEST MODEL SELECTION. USE LAST EPOCH MODEL FOR FINAL {final_eval_split.upper()} EVAL.')
                else:
                    print(f'BEST MODEL NOT FOUND; USE LAST EPOCH MODEL FOR FINAL {final_eval_split.upper()} EVAL.')
                final_checkpoint_name = 'last_epoch_model'

            print(f'RUN FINAL {final_eval_split.upper()} EVAL AFTER TRAINING.')
            final_eval_epoch = int(completed_epoch)
            total_elapsed_hms = self._format_elapsed_hms(time.perf_counter() - train_start_time)
            final_eval_metrics = self.test_epoch(
                net,
                final_eval_loader,
                final_eval_epoch - 1,
                train_loss=epoch_metrics['train_loss'],
                save_batch_triplet=True,
                split_name=final_eval_split,
            )
            self._write_eval_summary(
                'final',
                final_eval_metrics,
                checkpoint_name=final_checkpoint_name,
                evaluated_split=final_eval_split,
                eval_epoch=final_eval_epoch,
                elapsed_hms=total_elapsed_hms,
            )

            print('FINISH.')
            writer.close()

    def test_epoch(self, model, loader, epoch, train_loss=float('nan'), save_batch_triplet=False, split_name='test'):
        model.eval()
        self.dice_ls = []
        self.Jac_ls = []
        self.cldc_ls = []
        self.precision_ls = []
        self.accuracy_ls = []
        self.betti_error_0_ls = []
        self.betti_error_1_ls = []
        self.val_loss_ls = []
        sample_metric_rows = []

        sample_csv = os.path.join(
            self.args.save, 'models', f'{split_name}_sample_metrics.csv'
        )
        os.makedirs(os.path.dirname(sample_csv), exist_ok=True)
        dataset_name = str(getattr(self.args, 'dataset', 'unknown'))
        output_dir_name = str(getattr(self.args, 'output_dir', ''))
        if output_dir_name == '':
            output_dir_name = str(getattr(self.args, 'save', 'output'))
        progress_prefix = f'({dataset_name}) *({output_dir_name})'

        with torch.no_grad():
            with tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f'{progress_prefix} {split_name.capitalize()} Eval',
                dynamic_ncols=True,
                leave=False,
            ) as eval_progress:
                for j_batch, test_data in eval_progress:
                    X_test_raw, y_test_raw_raw, sample_names, binary_gt = self._unpack_test_batch(test_data, j_batch)
                    X_test = X_test_raw
                    y_test_raw = y_test_raw_raw

                    X_test = X_test.cuda()
                    y_test_raw = y_test_raw.float().cuda()

                    model_output = self._unpack_model_outputs(model(X_test))
                    if self.fusion_enabled:
                        output_test = model_output['fused']
                        val_loss, val_main_dict = self._compute_fusion_profile_loss(
                            model_output,
                            y_test_raw,
                            binary_gt=binary_gt,
                            collect_edge_stats=False,
                        )
                    else:
                        output_test = model_output['fused']
                        aux_out = model_output['aux']
                        val_loss_main, val_main_dict = self.loss_func(output_test, y_test_raw, return_details=True)
                        val_loss_aux, _ = self.loss_func(aux_out, y_test_raw, return_details=True)
                        val_loss = val_loss_main + 0.3 * val_loss_aux

                        if getattr(self.args, 'use_seg_aux', False) and 'mask_logit' in model_output and binary_gt is not None:
                            if binary_gt.ndim == 3:
                                binary_gt = binary_gt.unsqueeze(1)
                            seg_loss = F.binary_cross_entropy_with_logits(model_output['mask_logit'], binary_gt)
                            val_loss = val_loss + getattr(self.args, 'seg_aux_weight', 0.0) * seg_loss
                            val_main_dict['seg_aux'] = seg_loss
                    self.val_loss_ls.append(val_loss.item())
                    if j_batch == 0:
                        val_term_sums = {k: 0.0 for k in val_main_dict.keys()}
                        val_term_counts = 0
                    for k, v in val_main_dict.items():
                        val_term_sums[k] += float(v.detach().item())
                    val_term_counts += 1

                    batch, _, H, W = X_test.shape

                    hori_translation = self.hori_translation.repeat(batch, 1, 1, 1).cuda()
                    verti_translation = self.verti_translation.repeat(batch, 1, 1, 1).cuda()

                    if self.args.num_class == 1:
                        if self.loss_func.label_mode in ['dist', 'dist_inverted']:
                            # Keep eval path aligned with the distance training path:
                            # sigmoid affinity -> bilateral voting -> mask-probability threshold.
                            pred_conn_prob = torch.sigmoid(output_test).view(
                                [batch, -1, self.conn_channels, H, W]
                            )
                            pred_score, _ = self.apply_connectivity_voting(
                                pred_conn_prob, hori_translation, verti_translation
                            )
                            pred = self.dist_score_to_binary(pred_score.view(batch, 1, H, W))
                            gt_mask = self.dist_to_binary(y_test_raw, self.loss_func.label_mode)
                        else:
                            # Preserve upstream binary evaluation path.
                            output_prob = torch.sigmoid(output_test)
                            pred_conn_prob = output_prob.view([batch, -1, self.conn_channels, H, W])
                            pred_bin = (pred_conn_prob > 0.5).float()
                            gt_mask = (y_test_raw > 0.5).float()
                            pred = self.connectivity_to_mask(
                                pred_bin, hori_translation, verti_translation
                            )

                        if gt_mask.dim() == 3:
                            gt_mask = gt_mask.unsqueeze(1)

                        pred_to_save = pred
                        mask_to_save = gt_mask

                        dice, Jac = self.per_class_dice(pred, gt_mask)
                        sample_cldc = []
                        sample_betti_error_0 = []
                        sample_betti_error_1 = []
                        if self.args.dataset in ('isic', 'cremi'):
                            sample_precision, sample_accuracy = self._compute_binary_precision_accuracy(pred, gt_mask)
                        else:
                            sample_precision = torch.full((batch,), float('nan'), device=pred.device)
                            sample_accuracy = torch.full((batch,), float('nan'), device=pred.device)
                        self.precision_ls += sample_precision.detach().cpu().tolist()
                        self.accuracy_ls += sample_accuracy.detach().cpu().tolist()
                        for b_idx in range(batch):
                            pred_np = pred[b_idx, 0].detach().cpu().numpy()
                            target_np = gt_mask[b_idx, 0].detach().cpu().numpy()
                            cldc = float(clDice(pred_np, target_np))
                            sample_cldc.append(cldc)
                            self.cldc_ls.append(cldc)
                            betti0_error_ls, betti1_error_ls = getBettiErrors(
                                pred[b_idx, 0], gt_mask[b_idx, 0]
                            )
                            sample_betti_error_0.append(
                                float(np.mean(betti0_error_ls)) if len(betti0_error_ls) > 0 else float('nan')
                            )
                            sample_betti_error_1.append(
                                float(np.mean(betti1_error_ls)) if len(betti1_error_ls) > 0 else float('nan')
                            )
                            self.betti_error_0_ls.append(sample_betti_error_0[-1])
                            self.betti_error_1_ls.append(sample_betti_error_1[-1])
                    else:
                        # multi-class branch
                        y_test = y_test_raw.long()
                        class_pred = output_test.view([batch, -1, self.conn_channels, H, W])
                        final_pred, _ = self.apply_connectivity_voting(
                            class_pred, hori_translation, verti_translation
                        )
                        pred = get_mask(final_pred)
                        pred = self.one_hot(pred, X_test.shape)

                        pred_to_save = torch.argmax(pred, dim=1, keepdim=True)
                        mask_to_save = torch.argmax(y_test_raw, dim=1, keepdim=True)

                        dice, Jac = self.per_class_dice(pred, y_test)
                        sample_cldc = [float('nan')] * batch
                        sample_betti_error_0 = [float('nan')] * batch
                        sample_betti_error_1 = [float('nan')] * batch
                        if self.args.dataset in ('isic', 'cremi'):
                            pred_label = pred_to_save.squeeze(1).long()
                            true_label = mask_to_save.squeeze(1).long()
                            sample_precision, sample_accuracy = self._compute_multiclass_precision_accuracy(
                                pred_label,
                                true_label,
                                self.args.num_class,
                            )
                        else:
                            sample_precision = torch.full((batch,), float('nan'), device=pred.device)
                            sample_accuracy = torch.full((batch,), float('nan'), device=pred.device)
                        self.precision_ls += sample_precision.detach().cpu().tolist()
                        self.accuracy_ls += sample_accuracy.detach().cpu().tolist()
                        self.betti_error_0_ls += sample_betti_error_0
                        self.betti_error_1_ls += sample_betti_error_1

                    if save_batch_triplet:
                        self.save_checkpoint_batch_triplet(
                            X_test, pred_to_save, mask_to_save, epoch + 1, j_batch
                        )

                    # For multi-class segmentation, exclude BG class from averaged metrics
                    if self.args.num_class > 1:
                        sample_dice = torch.mean(dice[:, 1:], 1)
                        sample_jac = torch.mean(Jac[:, 1:], 1)
                        self.dice_ls += sample_dice.tolist()
                        self.Jac_ls += sample_jac.tolist()
                    else:
                        sample_dice = dice[:, 0]
                        sample_jac = Jac[:, 0]
                        self.dice_ls += sample_dice.tolist()
                        self.Jac_ls += sample_jac.tolist()

                    for b_idx in range(batch):
                        sample_metric_rows.append({
                            'epoch': int(epoch + 1),
                            'batch': int(j_batch),
                            'sample_in_batch': int(b_idx),
                            'sample_name': sample_names[b_idx],
                            'val_loss': float(val_loss.item()),
                            'dice': float(sample_dice[b_idx].item()),
                            'jac': float(sample_jac[b_idx].item()),
                            'cldice': float(sample_cldc[b_idx]),
                            'precision': float(sample_precision[b_idx].item()),
                            'accuracy': float(sample_accuracy[b_idx].item()),
                            'betti_error_0': float(sample_betti_error_0[b_idx]),
                            'betti_error_1': float(sample_betti_error_1[b_idx]),
                        })

                    if j_batch % (max(1, int(len(loader) / 5))) == 0:
                        eval_progress.set_postfix(dice=f'{float(np.mean(self.dice_ls)):.3f}')

            Jac_ls = np.array(self.Jac_ls)
            total_val_loss = float(np.mean(self.val_loss_ls)) if self.val_loss_ls else float('nan')
            total_cldice = float(np.mean(self.cldc_ls)) if len(self.cldc_ls) > 0 else float('nan')
            total_precision = float(np.nanmean(np.array(self.precision_ls))) if len(self.precision_ls) > 0 else float('nan')
            total_accuracy = float(np.nanmean(np.array(self.accuracy_ls))) if len(self.accuracy_ls) > 0 else float('nan')
            total_betti_error_0 = (
                float(np.mean(self.betti_error_0_ls))
                if len(self.betti_error_0_ls) > 0 else float('nan')
            )
            total_betti_error_1 = (
                float(np.mean(self.betti_error_1_ls))
                if len(self.betti_error_1_ls) > 0 else float('nan')
            )
            if 'val_term_counts' in locals() and val_term_counts > 0:
                val_loss_terms = {k: (v / val_term_counts) for k, v in val_term_sums.items()}
            else:
                val_loss_terms = {}

            write_header = not os.path.exists(sample_csv)
            with open(sample_csv, 'a', newline='') as f:
                fieldnames = [
                    'epoch',
                    'batch',
                    'sample_in_batch',
                    'sample_name',
                    'val_loss',
                    'dice',
                    'jac',
                    'cldice',
                    'precision',
                    'accuracy',
                    'betti_error_0',
                    'betti_error_1',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(sample_metric_rows)

            return {
                'dice': float(np.mean(self.dice_ls)),
                'jac': float(np.mean(Jac_ls)),
                'cldice': total_cldice,
                'precision': total_precision,
                'accuracy': total_accuracy,
                'betti_error_0': total_betti_error_0,
                'betti_error_1': total_betti_error_1,
                'val_loss': float(total_val_loss),
                'train_loss': float(train_loss),
                'val_loss_terms': val_loss_terms,
            }

    def per_class_dice(self, y_pred, y_true):
        return compute_per_class_dice(y_pred, y_true)

    def one_hot(self, target, shape):
        return one_hot_encode(target, shape, self.args.num_class)


def get_mask(output):
    return get_mask_from_logits(output)
