import argparse
import glob
import os
import random

import autorootcwd
# from GetDataset import MyDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread, imsave
from torch.autograd import Variable
from torchvision import datasets, transforms

from data_loader.GetDataset_CHASE import MyDataset_CHASE, MyDataset_DRIVE, MyDataset_OCTA500
from data_loader.GetDataset_CREMI import getdataset_cremi
from data_loader.GetDataset_ISIC2018 import ISIC2018_dataset
from data_loader.GetDataset_Retouch import MyDataset
from model.DconnNet import DconnNet
from solver import Solver


def get_experiment_output_name(args):
    if args.label_mode == 'binary':
        base_name = f'binary_{args.conn_num}_bce'
    else:
        base_name = f"{args.label_mode}_{args.conn_num}_{args.dist_aux_loss}"

    return base_name


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _resolve_retouch_case_roots(data_root, device_name):
    device_root = os.path.join(data_root, device_name)
    candidate_roots = [
        os.path.join(device_root, 'all'),
        os.path.join(device_root, 'train'),
        device_root,
    ]

    for candidate in candidate_roots:
        case_dirs = sorted(glob.glob(os.path.join(candidate, 'TRAIN*')))
        case_dirs = [p for p in case_dirs if os.path.isdir(p)]
        if case_dirs:
            return case_dirs

    searched = ', '.join(candidate_roots)
    raise FileNotFoundError(
        f'Could not find RETOUCH TRAIN* case folders for {device_name}. '
        f'Searched: {searched}'
    )


def _get_retouch_fold_indices(device_name, exp_id):
    if device_name in ('Cirrus', 'Spectrailis'):
        total = 24
        start = exp_id * 8
        end = (exp_id + 1) * 8
        test_id = list(range(start, end))
    elif device_name == 'Topcon':
        total = 22
        if exp_id < 2:
            test_id = list(range(exp_id * 7, (exp_id + 1) * 7))
        else:
            test_id = list(range(14, 22))
    else:
        raise ValueError(f'Unsupported RETOUCH device: {device_name}')

    train_id = sorted(set(range(total)) - set(test_id))
    return total, train_id, test_id


def parse_args():
    parser = argparse.ArgumentParser(
        description='DconnNet Training With Pytorch')

    parser.add_argument('--device', type=int, default=1,)
    parser.add_argument('--seed', type=int, default=42,
                        help='global random seed for reproducible training/evaluation')

    # dataset info
    parser.add_argument('--dataset', type=str, default='retouch-Spectrailis',
                        help='retouch-Spectrailis,retouch-Cirrus,retouch-Topcon, isic, chase, drive, octa500, cremi')

    parser.add_argument('--data_root', type=str, default='/retouch',
                        help='dataset directory (e.g. data/CREMI for --dataset cremi)')
    parser.add_argument('--resize', type=int, default=[256, 256], nargs='+',
                        help='image size: [height, width]')
    parser.add_argument('--label_mode', type=str, default='binary',
                        help='label mode: binary, dist, dist_inverted')

    # network option & hyper-parameters
    parser.add_argument('--num-class', type=int, default=4, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=45, metavar='N',
                        help='number of epochs to train (default: 45)')
    parser.add_argument('--lr', type=float, default=0.00085, metavar='LR',
                        help='learning rate (default: 8.5e-4)')
    parser.add_argument('--lr-update', type=str, default='step',
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts')
    parser.add_argument('--lr-step', type=int, default=12,
                        help='define only when you select step lr optimization: what is the step size for reducing your lr')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)')

    parser.add_argument('--use_SDL', action='store_true', default=False,
                        help='set as True if use SDL loss; only for Retouch dataset in this code. If you use it with other dataset please define your own path of label distribution in solver.py')
    parser.add_argument('--folds', type=int, default=1,
                        help='define folds number K for K-fold validation')
    parser.add_argument('--target_fold', type=int, default=None,
                        help='1-based fold index to run; default runs all folds')
    parser.add_argument('--conn_num', type=int, default=8, choices=[8, 24],
                        help='the number of connections for DconnNet (supported: 8, 24)')
    parser.add_argument('--tau', type=float, default=3.0,
                        help='the temperature parameter tau for the distance connectivity loss')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='the weight parameter sigma for the distance connectivity loss')
    parser.add_argument('--dist_aux_loss', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'gjml_sf_l1', 'cl_dice'],
                        help='auxiliary regression loss for dist/dist_inverted affinity targets')
    parser.add_argument('--dist_sf_l1_gamma', type=float, default=1.0,
                        help='gamma for Stable Focal-L1 when --dist_aux_loss=gjml_sf_l1')

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--weights', type=str, default='/data_loader/retouch_weights/',
                        help='path of SDL weights')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='base output directory; save path becomes output_dir/<fold_scope>/<experiment_name> (binary: binary_conn_bce, dist: label_mode_conn_dist_aux_loss)')

    parser.add_argument('--save-per-epochs', type=int, default=50,
                        help='per epochs to save')
    parser.add_argument('--monitor_metric', '--monitor-metric', type=str, default='val_dice',
                        choices=['val_dice', 'val_loss'],
                        help='metric for best checkpoint and early stopping')
    parser.add_argument('--early_stopping_patience', '--early-stopping-patience', type=int, default=20,
                        help='patience for validation-based early stopping (<=0 disables early stopping)')
    parser.add_argument('--early_stopping_min_delta', '--early-stopping-min-delta', type=float, default=0.001,
                        help='minimum improvement threshold for monitor metric')
    parser.add_argument('--early_stopping_tie_eps', '--early-stopping-tie-eps', type=float, default=1e-4,
                        help='tie epsilon for val_dice tie-break using val_loss')
    parser.add_argument('--early_stopping_stop_interval', '--early-stopping-stop-interval', type=int, default=10,
                        help='actual stopping happens only on epochs divisible by this interval')
    parser.add_argument('--tie_break_with_loss', '--tie-break-with-loss', dest='tie_break_with_loss',
                        action='store_true', default=True,
                        help='when val_dice is tied within eps, prefer lower val_loss')
    parser.add_argument('--no_tie_break_with_loss', '--no-tie-break-with-loss', dest='tie_break_with_loss',
                        action='store_false',
                        help='disable val_loss tie-break when val_dice values are tied')
    parser.add_argument('--save_best_only', '--save-best-only', dest='save_best_only',
                        action='store_true', default=False,
                        help='if set, disable periodic epoch checkpoints and keep best checkpoint only')
    parser.add_argument('--no_save_best_only', '--no-save-best-only', dest='save_best_only',
                        action='store_false',
                        help='enable periodic epoch checkpoints')

    # evaluation only
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='test only, please load the pretrained model')
    args = parser.parse_args()

    if args.output_dir:
        args.save = os.path.join(
            args.output_dir,
            args.dataset,
            get_experiment_output_name(args),
        )

    os.makedirs(args.save, exist_ok=True)

    if args.target_fold is not None:
        if args.target_fold < 1 or args.target_fold > args.folds:
            parser.error('--target_fold must be within [1, --folds]')

    if args.early_stopping_patience < 0:
        parser.error('--early_stopping_patience must be >= 0')
    if args.early_stopping_min_delta < 0:
        parser.error('--early_stopping_min_delta must be >= 0')
    if args.early_stopping_tie_eps < 0:
        parser.error('--early_stopping_tie_eps must be >= 0')
    if args.early_stopping_stop_interval < 1:
        parser.error('--early_stopping_stop_interval must be >= 1')

    # Group related runtime controls for clearer downstream consumption.
    args.monitoring_config = {
        'monitor_metric': args.monitor_metric,
    }
    args.early_stopping_config = {
        'patience': int(args.early_stopping_patience),
        'min_delta': float(args.early_stopping_min_delta),
        'tie_eps': float(args.early_stopping_tie_eps),
        'stop_interval': int(args.early_stopping_stop_interval),
        'tie_break_with_loss': bool(args.tie_break_with_loss),
    }
    args.checkpoint_config = {
        'save_best_only': bool(args.save_best_only),
        'save_per_epochs': int(args.save_per_epochs),
    }

    return args


def main(args):

    torch.cuda.set_device(args.device)  # GPU id
    seed_everything(args.seed)

    ## K-fold cross validation ##
    if args.target_fold is None:
        exp_indices = range(args.folds)
    else:
        exp_indices = [args.target_fold - 1]

    seed_everything(args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    validset = None
    testset = None
    val_loader = None
    test_loader = None

    if args.dataset == 'isic':
        trainset = ISIC2018_dataset(dataset_folder=args.data_root, folder='0', train_type='train', with_name=False)
        validset = ISIC2018_dataset(dataset_folder=args.data_root, folder='0', train_type='validation', with_name=False)
        testset = ISIC2018_dataset(dataset_folder=args.data_root, folder='0', train_type='test', with_name=False)

    elif 'retouch' in args.dataset:
        if len(exp_indices) != 1:
            raise ValueError(
                'RETOUCH uses fixed 3-fold patient splits and currently supports one fold per run. '
                'Please set --folds=3 and provide --target_fold.'
            )

        device_name = args.dataset.split('-', 1)[1]
        exp_id = exp_indices[0]
        case_roots = _resolve_retouch_case_roots(args.data_root, device_name)
        total_cases, train_id, test_id = _get_retouch_fold_indices(device_name, exp_id)

        if len(case_roots) < total_cases:
            raise ValueError(
                f'Found only {len(case_roots)} RETOUCH cases for {device_name}, '
                f'but expected at least {total_cases}.'
            )

        case_roots = case_roots[:total_cases]
        train_root = [case_roots[i] for i in train_id]
        test_root = [case_roots[i] for i in test_id]

        trainset = MyDataset(args, train_root=train_root, mode='train')
        validset = MyDataset(args, train_root=test_root, mode='test')

    elif args.dataset == 'chase':
        overall_id = ['01', '02', '03', '04', '05', '06',
                      '07', '08', '09', '10', '11', '12', '13', '14']
        train_id = overall_id[:10]
        test_id = overall_id[10:]
        # print(train_id)
        trainset = MyDataset_CHASE(args, train_root=args.data_root, pat_ls=train_id, mode='train', label_mode=args.label_mode)
        # CHASE uses the held-out fold as the evaluation split during training
        # and for the final post-training evaluation.
        testset = MyDataset_CHASE(args, train_root=args.data_root, pat_ls=test_id, mode='test', label_mode=args.label_mode)

    elif args.dataset == 'drive':
        trainset = MyDataset_DRIVE(args, train_root=args.data_root, mode='train', label_mode=args.label_mode)
        testset = MyDataset_DRIVE(args, train_root=args.data_root, mode='test', label_mode=args.label_mode)
    elif args.dataset == 'octa500-6M' or args.dataset == 'octa500-3M':
        trainset = MyDataset_OCTA500(args, train_root=args.data_root, mode='train', label_mode=args.label_mode)
        validset = MyDataset_OCTA500(args, train_root=args.data_root, mode='val', label_mode=args.label_mode)
        testset = MyDataset_OCTA500(args, train_root=args.data_root, mode='test', label_mode=args.label_mode)
    elif args.dataset == 'cremi':
        trainset = getdataset_cremi(args, train_root=args.data_root, mode='train', label_mode=args.label_mode)
        testset = getdataset_cremi(args, train_root=args.data_root, mode='test', label_mode=args.label_mode)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
        pass

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    print("Train batch number: %i" % len(train_loader))
    if validset is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=validset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
        print("Validation batch number: %i" % len(val_loader))
    if testset is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
        print("Test batch number: %i" % len(test_loader))
    elif val_loader is not None:
        # Backward-compatible fallback for datasets that expose only one held-out split.
        test_loader = val_loader

        #### Above: define how you get the data on your own dataset ######
    model = DconnNet(
        num_class=args.num_class,
        conn_num=args.conn_num,
    ).cuda()

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=torch.device('cpu')))
        model = model.cuda()
        print("Loaded pretrained model from %s" % args.pretrained)

    solver = Solver(args)

    solver.train(model, train_loader, val_loader,
                 num_epochs=args.epochs, label_mode=args.label_mode,
                 test_loader=test_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
