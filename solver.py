import csv
import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch.optim import lr_scheduler

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

from connect_loss import Bilateral_voting, Bilateral_voting_kxk, connect_loss, resolve_connectivity_layout
from lr_update import get_lr
from metrics.cal_betti import getBettiErrors
from metrics.cldice import clDice

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


class Solver(object):
    def __init__(self, args, optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        H, W = args.resize

        self.hori_translation = torch.zeros([1, self.NumClass, W, W])
        for i in range(W - 1):
            self.hori_translation[:, :, i, i + 1] = torch.tensor(1.0)

        self.verti_translation = torch.zeros([1, self.NumClass, H, H])
        for j in range(H - 1):
            self.verti_translation[:, :, j, j + 1] = torch.tensor(1.0)

        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def create_exp_directory(self, exp_id):
        exp_model_dir = os.path.join(self.args.save, 'models', str(exp_id))
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)

        results_csv = 'results_' + str(exp_id) + '.csv'
        with open(os.path.join(self.args.save, results_csv), 'w') as f:
            f.write(
                'epoch,train_loss,val_loss,dice,Jac,clDice,precision,accuracy,betti_error_0,betti_error_1,elapsed_hms\n'
            )

    def _format_elapsed_hms(self, elapsed_seconds):
        if elapsed_seconds is None or math.isnan(float(elapsed_seconds)):
            return ''

        total_seconds = max(0, int(round(float(elapsed_seconds))))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def _write_epoch_result_row(self, exp_id, epoch, metrics, elapsed_hms=''):
        results_csv = 'results_' + str(exp_id) + '.csv'
        with open(os.path.join(self.args.save, results_csv), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f'{int(epoch):03d}',
                    f'{float(metrics["train_loss"]):0.6f}',
                    f'{float(metrics["val_loss"]):0.6f}',
                    f'{float(metrics["dice"]):0.6f}',
                    f'{float(metrics["jac"]):0.6f}',
                    f'{float(metrics["cldice"]):0.6f}',
                    f'{float(metrics.get("precision", float("nan"))):0.6f}',
                    f'{float(metrics.get("accuracy", float("nan"))):0.6f}',
                    f'{float(metrics["betti_error_0"]):0.6f}',
                    f'{float(metrics["betti_error_1"]):0.6f}',
                    elapsed_hms or '',
                ]
            )

    def _write_eval_summary(
        self,
        exp_id,
        split_name,
        metrics,
        checkpoint_name='',
        evaluated_split='',
        eval_epoch='',
        elapsed_hms='',
    ):
        summary_csv = os.path.join(
            self.args.save,
            f'{split_name}_results_{exp_id}.csv',
        )
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    'result_type',
                    'evaluated_split',
                    'eval_epoch',
                    'checkpoint',
                    'train_loss',
                    'eval_loss',
                    'dice',
                    'jac',
                    'cldice',
                    'precision',
                    'accuracy',
                    'betti_error_0',
                    'betti_error_1',
                    'elapsed_hms',
                ]
            )
            writer.writerow(
                [
                    split_name,
                    evaluated_split,
                    eval_epoch,
                    checkpoint_name,
                    f'{float(metrics["train_loss"]):0.6f}',
                    f'{float(metrics["val_loss"]):0.6f}',
                    f'{float(metrics["dice"]):0.6f}',
                    f'{float(metrics["jac"]):0.6f}',
                    f'{float(metrics["cldice"]):0.6f}',
                    f'{float(metrics.get("precision", float("nan"))):0.6f}',
                    f'{float(metrics.get("accuracy", float("nan"))):0.6f}',
                    f'{float(metrics["betti_error_0"]):0.6f}',
                    f'{float(metrics["betti_error_1"]):0.6f}',
                    elapsed_hms or '',
                ]
            )

    def _compute_binary_precision_accuracy(self, pred_mask, gt_mask, eps=1e-6):
        pred_bin = (pred_mask > 0.5).float()
        gt_bin = (gt_mask > 0.5).float()

        tp = torch.sum(pred_bin * gt_bin, dim=(1, 2, 3))
        fp = torch.sum(pred_bin * (1.0 - gt_bin), dim=(1, 2, 3))
        tn = torch.sum((1.0 - pred_bin) * (1.0 - gt_bin), dim=(1, 2, 3))
        fn = torch.sum((1.0 - pred_bin) * gt_bin, dim=(1, 2, 3))

        precision = (tp + eps) / (tp + fp + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        return precision, accuracy

    def _compute_multiclass_precision_accuracy(self, pred_label, true_label, num_class, eps=1e-6):
        batch_size = pred_label.shape[0]
        precision_vals = []
        accuracy_vals = []

        class_ids = list(range(1, num_class))
        if len(class_ids) == 0:
            class_ids = list(range(num_class))

        for b_idx in range(batch_size):
            pred_b = pred_label[b_idx]
            true_b = true_label[b_idx]

            class_precisions = []
            for class_id in class_ids:
                pred_pos = (pred_b == class_id)
                true_pos = (true_b == class_id)
                tp = torch.sum(pred_pos & true_pos).float()
                fp = torch.sum(pred_pos & (~true_pos)).float()
                class_precisions.append((tp + eps) / (tp + fp + eps))

            precision_vals.append(torch.mean(torch.stack(class_precisions)))
            accuracy_vals.append(torch.mean((pred_b == true_b).float()))

        return torch.stack(precision_vals), torch.stack(accuracy_vals)

    def save_checkpoint_batch_triplet(self, images, preds, masks, exp_id, epoch, batch_idx):
        base_dir = os.path.join(
            self.args.save, 'models', str(exp_id), 'checkpoint_batches', f'epoch_{epoch:03d}'
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
        if isinstance(test_data, (list, tuple)):
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
            raise TypeError('test_data must be a tuple/list from DataLoader')

        if not torch.is_tensor(X_test) or not torch.is_tensor(y_test_raw):
            raise TypeError('image and label in test_data must be tensors')

        batch_size = int(X_test.shape[0])
        sample_names = self._normalize_sample_names(sample_name_source, batch_size, batch_idx)
        return X_test, y_test_raw, sample_names

    def get_density(self, pos_cnt, bins=50):
        # only used for Retouch in this code
        val_in_bin_ = [[], [], []]
        density_ = [[], [], []]
        bin_wide_ = []

        for n in range(3):
            density = []
            val_in_bin = []
            c1 = [i for i in pos_cnt[n] if i != 0]
            c1_t = torch.tensor(c1)
            bin_wide = (c1_t.max() + 50) / bins
            bin_wide_.append(bin_wide)

            edges = torch.arange(bins + 1).float() * bin_wide
            for i in range(bins):
                val = [c1[j] for j in range(len(c1)) if (
                    (c1[j] >= edges[i]) & (c1[j] < edges[i + 1]))]
                val_in_bin.append(val)
                inds = (c1_t >= edges[i]) & (c1_t < edges[i + 1])
                num_in_bin = inds.sum().item()
                density.append(num_in_bin)

            denominator = torch.tensor(density).sum()
            density = torch.tensor(density) / denominator
            density_[n] = density
            val_in_bin_[n] = val_in_bin

        print(density_)
        return density_, val_in_bin_, bin_wide_

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
        layout = resolve_connectivity_layout(self.args.conn_num)
        if self.args.conn_num == 8:
            pred_map, vote_map = Bilateral_voting(conn_map, hori_translation, verti_translation)
        elif self.args.conn_num == 24:
            kxk_size = layout['kernel_size']
            pred_map, vote_map = Bilateral_voting_kxk(
                conn_map,
                hori_translation,
                verti_translation,
                conn_num=kxk_size,
            )
        else:
            raise ValueError(f"Unsupported conn_num {self.args.conn_num}, only 8 and 24 are supported")
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

    def train(self, model, train_loader, val_loader, exp_id, num_epochs=10, label_mode='binary', test_loader=None):
        optim = self.optim(model.parameters(), lr=self.lr)

        print('START TRAIN.')
        self.create_exp_directory(exp_id)
        tb_dir = os.path.join(self.args.save, 'tensorboard', f'exp_{exp_id}')
        writer = SummaryWriter(log_dir=tb_dir)

        if self.args.use_SDL:
            assert 'retouch' in self.args.dataset, (
                'Please input the calculated distribution data of your own dataset, '
                'if you are now using Retouch'
            )
            device_name = self.args.dataset.split('retouch-')[1]
            pos_cnt = np.load(
                self.args.weights + device_name + '/training_positive_pixel_' + str(exp_id) + '.npy',
                allow_pickle=True
            )
            density, val_in_bin, bin_wide = self.get_density(pos_cnt)
            self.loss_func = connect_loss(
                self.args,
                self.hori_translation,
                self.verti_translation,
                density,
                bin_wide,
                label_mode=label_mode,
                conn_num=self.args.conn_num,
                sigma=self.args.sigma,
            )
        else:
            self.loss_func = connect_loss(
                self.args,
                self.hori_translation,
                self.verti_translation,
                label_mode=label_mode,
                conn_num=self.args.conn_num,
                sigma=self.args.sigma,
            )

        net, optimizer = amp.initialize(model, optim, opt_level='O2')

        best_p = 0
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
            eval_split_name = 'test' if test_loader is not None else 'val'
            print(f'START {eval_split_name.upper()}-ONLY EVAL.')
            test_metrics = self.test_epoch(
                net,
                val_loader if test_loader is None else test_loader,
                0,
                exp_id,
                split_name=eval_split_name,
            )
            self._write_epoch_result_row(exp_id, 1, test_metrics, elapsed_hms='')
            self._write_eval_summary(
                exp_id,
                'final',
                test_metrics,
                checkpoint_name=os.path.basename(self.args.pretrained) if self.args.pretrained else '',
                evaluated_split=eval_split_name,
                eval_epoch=1,
                elapsed_hms='',
            )
            writer.close()
        else:
            global_step = 0
            train_start_time = time.perf_counter()
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
            for epoch in range(self.args.epochs):
                net.train()
                train_loss_epoch = []
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

                for i_batch, sample_batched in enumerate(train_loader):
                    X = sample_batched[0]
                    y = sample_batched[1]

                    X = X.cuda()
                    y = y.float().cuda()

                    optim.zero_grad()
                    output, aux_out = net(X)

                    if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'set_dist_edge_stat_collection'):
                        self.loss_func.set_dist_edge_stat_collection(True)
                    loss_main, loss_main_dict = self.loss_func(output, y, return_details=True)
                    if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'set_dist_edge_stat_collection'):
                        self.loss_func.set_dist_edge_stat_collection(False)
                    loss_aux, loss_aux_dict = self.loss_func(aux_out, y, return_details=True)
                    loss = loss_main + 0.3 * loss_aux

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

                    print(
                        '[epoch:' + str(epoch) + '][Iteration : ' + str(i_batch) + '/' +
                        str(len(train_loader)) + '] Total:%.3f' % (loss.item())
                    )

                mean_train_loss = float(np.mean(train_loss_epoch)) if train_loss_epoch else float('nan')
                should_save_batch_triplet = ((epoch + 1) % self.args.save_per_epochs == 0)

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

                if val_loader is not None:
                    print('RUN VALIDATION ON validation split.')

                    epoch_metrics = self.test_epoch(
                        net,
                        val_loader,
                        epoch,
                        exp_id,
                        train_loss=mean_train_loss,
                        save_batch_triplet=should_save_batch_triplet,
                        split_name='val',
                    )
                    dice_p = epoch_metrics['dice']
                    if best_p < dice_p:
                        best_p = dice_p
                        best_model_dir = os.path.join(self.args.save, 'models', str(exp_id))
                        torch.save(model.state_dict(), os.path.join(best_model_dir, 'best_model.pth'))
                        with open(os.path.join(best_model_dir, 'best_model_meta.txt'), 'w') as f:
                            f.write(f'best_epoch={epoch + 1}\n')
                            f.write(f'best_dice={best_p:.6f}\n')
                            f.write(f'best_train_loss={epoch_metrics["train_loss"]:.6f}\n')
                            f.write(f'best_val_loss={epoch_metrics["val_loss"]:.6f}\n')
                            f.write(f'best_jac={epoch_metrics["jac"]:.6f}\n')
                            f.write(f'best_clDice={epoch_metrics["cldice"]:.6f}\n')
                            f.write(
                                f'best_betti_error_0={epoch_metrics["betti_error_0"]:.6f}\n'
                            )
                            f.write(
                                f'best_betti_error_1={epoch_metrics["betti_error_1"]:.6f}\n'
                            )

                if (epoch + 1) % self.args.save_per_epochs == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.args.save, 'models', str(exp_id), str(epoch + 1) + '_model.pth')
                    )

                if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'get_dist_edge_stats'):
                    dist_edge_stats = self.loss_func.get_dist_edge_stats()
                else:
                    dist_edge_stats = None
                writer.add_scalar('epoch/train_loss', float(mean_train_loss), epoch + 1)
                if not math.isnan(float(epoch_metrics['val_loss'])):
                    writer.add_scalar('epoch/val_loss', float(epoch_metrics['val_loss']), epoch + 1)
                if not math.isnan(float(epoch_metrics['dice'])):
                    writer.add_scalar('epoch/dice', float(epoch_metrics['dice']), epoch + 1)
                if not math.isnan(float(epoch_metrics['jac'])):
                    writer.add_scalar('epoch/jac', float(epoch_metrics['jac']), epoch + 1)
                if not math.isnan(float(epoch_metrics['cldice'])):
                    writer.add_scalar('epoch/cldice', float(epoch_metrics['cldice']), epoch + 1)
                if not math.isnan(float(epoch_metrics['precision'])):
                    writer.add_scalar('epoch/precision', float(epoch_metrics['precision']), epoch + 1)
                if not math.isnan(float(epoch_metrics['accuracy'])):
                    writer.add_scalar('epoch/accuracy', float(epoch_metrics['accuracy']), epoch + 1)
                if not math.isnan(float(epoch_metrics['betti_error_0'])):
                    writer.add_scalar(
                        'epoch/betti_error_0',
                        float(epoch_metrics['betti_error_0']),
                        epoch + 1,
                    )
                if not math.isnan(float(epoch_metrics['betti_error_1'])):
                    writer.add_scalar(
                        'epoch/betti_error_1',
                        float(epoch_metrics['betti_error_1']),
                        epoch + 1,
                    )
                for key, value in epoch_metrics.get('val_loss_terms', {}).items():
                    writer.add_scalar(f'val/{key}', float(value), epoch + 1)
                if dist_edge_stats is not None:
                    writer.add_scalar('train/edge_mean', float(dist_edge_stats['edge_mean']), epoch + 1)
                    writer.add_scalar('train/edge_nonzero_ratio', float(dist_edge_stats['edge_nonzero_ratio']), epoch + 1)
                if dist_edge_stats is None:
                    print('[Epoch :%d] total loss:%.3f ' % (epoch, mean_train_loss))
                else:
                    print(
                        '[Epoch :%d] total loss:%.3f edge_mean:%.6f edge_nonzero:%.6f '
                        % (
                            epoch,
                            mean_train_loss,
                            dist_edge_stats['edge_mean'],
                            dist_edge_stats['edge_nonzero_ratio'],
                        )
                    )
                elapsed_hms = self._format_elapsed_hms(time.perf_counter() - train_start_time)
                if val_loader is not None:
                    self._write_epoch_result_row(exp_id, epoch + 1, epoch_metrics, elapsed_hms)

            final_eval_split = 'test' if test_loader is not None else 'val'
            final_eval_loader = test_loader if test_loader is not None else val_loader
            if final_eval_loader is None:
                raise ValueError('Final evaluation requires test_loader or val_loader, but both are None.')
            best_model_path = os.path.join(self.args.save, 'models', str(exp_id), 'best_model.pth')
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
            final_eval_epoch = epoch + 1
            total_elapsed_hms = self._format_elapsed_hms(time.perf_counter() - train_start_time)
            final_eval_metrics = self.test_epoch(
                net,
                final_eval_loader,
                epoch,
                exp_id,
                train_loss=epoch_metrics['train_loss'],
                save_batch_triplet=True,
                split_name=final_eval_split,
            )
            self._write_eval_summary(
                exp_id,
                'final',
                final_eval_metrics,
                checkpoint_name=final_checkpoint_name,
                evaluated_split=final_eval_split,
                eval_epoch=final_eval_epoch,
                elapsed_hms=total_elapsed_hms,
            )

            print('FINISH.')
            writer.close()

    def test_epoch(self, model, loader, epoch, exp_id, train_loss=float('nan'), save_batch_triplet=False, split_name='test'):
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
            self.args.save, 'models', str(exp_id), f'{split_name}_sample_metrics.csv'
        )
        os.makedirs(os.path.dirname(sample_csv), exist_ok=True)

        with torch.no_grad():
            for j_batch, test_data in enumerate(loader):
                X_test_raw, y_test_raw_raw, sample_names = self._unpack_test_batch(test_data, j_batch)
                X_test = X_test_raw
                y_test_raw = y_test_raw_raw

                X_test = X_test.cuda()
                y_test_raw = y_test_raw.float().cuda()

                output_test, aux_out = model(X_test)

                val_loss_main, val_main_dict = self.loss_func(output_test, y_test_raw, return_details=True)
                val_loss_aux, _ = self.loss_func(aux_out, y_test_raw, return_details=True)
                val_loss = val_loss_main + 0.3 * val_loss_aux
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
                            [batch, -1, self.args.conn_num, H, W]
                        )
                        pred_score, _ = self.apply_connectivity_voting(
                            pred_conn_prob, hori_translation, verti_translation
                        )
                        pred = self.dist_score_to_binary(pred_score.view(batch, 1, H, W))
                        gt_mask = self.dist_to_binary(y_test_raw, self.loss_func.label_mode)
                    else:
                        # Preserve upstream binary evaluation path.
                        output_prob = torch.sigmoid(output_test)
                        pred_conn_prob = output_prob.view([batch, -1, self.args.conn_num, H, W])
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
                    if self.args.dataset == 'isic':
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
                    class_pred = output_test.view([batch, -1, self.args.conn_num, H, W])
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
                    if self.args.dataset == 'isic':
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
                        X_test, pred_to_save, mask_to_save, exp_id, epoch + 1, j_batch
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
                    print(
                        '[Iteration : ' + str(j_batch) + '/' + str(len(loader)) +
                        '] Total DSC:%.3f ' % (np.mean(self.dice_ls))
                    )

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
        eps = 0.0001

        FN = torch.sum((1 - y_pred) * y_true, dim=(2, 3))
        FP = torch.sum((1 - y_true) * y_pred, dim=(2, 3))
        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT * Pred, dim=(2, 3))

        union = torch.sum(GT, dim=(2, 3)) + torch.sum(Pred, dim=(2, 3))
        dice = (2 * inter + eps) / (union + eps)
        Jac = (inter + eps) / (inter + FP + FN + eps)

        return dice, Jac

    def one_hot(self, target, shape):
        one_hot_mat = torch.zeros(
            [shape[0], self.args.num_class, shape[2], shape[3]]
        ).cuda()
        target = target.cuda()
        one_hot_mat.scatter_(1, target, 1)
        return one_hot_mat


def get_mask(output):
    output = F.softmax(output, dim=1)
    _, pred = output.topk(1, dim=1)
    return pred
