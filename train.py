import argparse
import glob
import os
import random

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
from data_loader.GetDataset_ISIC2018 import ISIC2018_dataset
from data_loader.GetDataset_Retouch import MyDataset
from model.DconnNet import DconnNet
from solver import Solver


def get_experiment_output_name(args):
    if args.label_mode == 'binary':
        base_name = f'binary_{args.conn_num}_bce'
    else:
        base_name = f"{args.label_mode}_{args.conn_num}_{args.dist_aux_loss}"

    if getattr(args, 'direction_grouping', 'none') != 'none':
        base_name = f"{base_name}_{args.direction_grouping}_{args.direction_fusion}"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='DconnNet Training With Pytorch')

    parser.add_argument('--device', type=int, default=1,)
    parser.add_argument('--seed', type=int, default=42,
                        help='global random seed for reproducible training/evaluation')

    # dataset info
    parser.add_argument('--dataset', type=str, default='retouch-Spectrailis',
                        help='retouch-Spectrailis,retouch-Cirrus,retouch-Topcon, isic, chase, drive, octa500')

    parser.add_argument('--data_root', type=str, default='/retouch',
                        help='dataset directory')
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
    parser.add_argument('--direction_grouping', type=str, default='none',
                        choices=['none', 'coarse24to8'],
                        help='optional fork-specific directional grouping path; coarse24to8 compresses 24 proto-directions into the canonical 8-direction output layout')
    parser.add_argument('--direction_fusion', type=str, default='weighted_sum',
                        choices=['mean', 'weighted_sum', 'conv1x1', 'attention_gating'],
                        help='fusion block used when --direction_grouping=coarse24to8')
    parser.add_argument('--tau', type=float, default=3.0,
                        help='the temperature parameter tau for the distance connectivity loss')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='the weight parameter sigma for the distance connectivity loss')
    parser.add_argument('--dist_aux_loss', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'gjml_sf_l1'],
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

    if args.direction_grouping == 'coarse24to8' and args.conn_num != 8:
        parser.error('--direction_grouping=coarse24to8 requires --conn_num=8 because the grouped path feeds the canonical 8-direction branch')

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
        trainset = ISIC2018_dataset(dataset_folder=args.data_root, folder=0, train_type='train', with_name=False)
        validset = ISIC2018_dataset(dataset_folder=args.data_root, folder=0, train_type='validation', with_name=False)
        testset = ISIC2018_dataset(dataset_folder=args.data_root, folder=0, train_type='test', with_name=False)

    elif 'retouch' in args.dataset:
        raise NotImplementedError("Retouch dataset loading is not implemented in this code. Please implement your own data loading logic based on the commented-out code below or your specific data organization.")
        # TODO: change when data is ready
        # device_name = args.dataset.split('-')[1]
        # path = args.data_root + '/'+device_name + '/train'
        # pat_ls = glob.glob(path+'/*')

        # # for Cirrus
        # if device_name == 'Cirrus':
        #     # total_id = [i for i in range(24)]
        #     # test_id = [i for i in range(exp_id*8, (exp_id+1)*8)]

        # if device_name == 'Spectrailis':
        # total_id = [i for i in range(24)]
        # test_id = [i for i in range(exp_id*8, (exp_id+1)*8)]

        # if device_name == 'Topcon':

        # total_id = [i for i in range(22)]
        # if exp_id < 2:
        #     test_id = [i for i in range(exp_id*7, (exp_id+1)*7)]
        # else:
        #     test_id = [i for i in range(14, 22)]

        # train_id = set(total_id) - set(test_id)
        # test_root = [pat_ls[i] for i in test_id]
        # train_root = [pat_ls[i] for i in train_id]

        # trainset = MyDataset(args, train_root=train_root, mode='train')
        # validset = MyDataset(args, train_root=test_root, mode='test')

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
        direction_grouping=args.direction_grouping,
        direction_fusion=args.direction_fusion,
    ).cuda()

    if args.pretrained:
        model.load_state_dict(torch.load(
            args.pretrained, map_location=torch.device('cpu')))
        model = model.cuda()

    solver = Solver(args)

    solver.train(model, train_loader, val_loader,
                 num_epochs=args.epochs, label_mode=args.label_mode,
                 test_loader=test_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
