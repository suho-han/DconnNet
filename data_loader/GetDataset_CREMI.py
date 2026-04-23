# -*- coding: UTF-8 -*-

import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data as data


def _normalize_image_to_dconnnet(x):
    x = np.clip(x, 0.0, 255.0)
    x = x / 255.0 * 3.2 - 1.6
    return x.astype(np.float32)


class MyDataset_CREMI(data.Dataset):
    def __init__(self, args, train_root, mode='train', label_mode='binary'):
        if mode not in ('train', 'test'):
            raise ValueError(f"Unsupported mode for CREMI: {mode}")
        if label_mode not in ('binary', 'dist', 'dist_inverted'):
            raise ValueError(f"Unsupported label_mode for CREMI: {label_mode}")

        self.args = args
        self.mode = mode
        self.train = mode == 'train'
        self.label_mode = label_mode

        img_path = os.path.join(train_root, mode, 'images')
        gt_dir = 'labels'
        gt_suffix = '.npy'
        if label_mode == 'dist':
            gt_dir = 'labels_dist'
            gt_suffix = '_dist.npy'
        elif label_mode == 'dist_inverted':
            gt_dir = 'labels_dist_inverted'
            gt_suffix = '_dist_inverted.npy'
        gt_path = os.path.join(train_root, mode, gt_dir)

        self.img_ls = sorted(glob(os.path.join(img_path, '*.npy')))
        if len(self.img_ls) == 0:
            raise FileNotFoundError(f'No CREMI image .npy files found in {img_path}')

        self.mask_ls = []
        self.name_ls = []
        missing = []
        for img in self.img_ls:
            stem = os.path.splitext(os.path.basename(img))[0]
            mask = os.path.join(gt_path, stem + gt_suffix)
            if not os.path.exists(mask):
                missing.append(mask)
            self.mask_ls.append(mask)
            self.name_ls.append(stem)

        if missing:
            preview = ', '.join(missing[:3])
            raise FileNotFoundError(
                f'Missing {len(missing)} CREMI label files for mode={mode}, label_mode={label_mode}. '
                f'Examples: {preview}'
            )

    def __getitem__(self, index):
        resize_h, resize_w = tuple(self.args.resize[:2])

        img = np.load(self.img_ls[index]).astype(np.float32)
        mask = np.load(self.mask_ls[index], allow_pickle=True).astype(np.float32)

        if img.ndim != 2:
            raise ValueError(f'CREMI image must be 2D, got shape {img.shape} for {self.img_ls[index]}')
        if mask.ndim != 2:
            raise ValueError(f'CREMI label must be 2D, got shape {mask.shape} for {self.mask_ls[index]}')

        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        mask_interp = cv2.INTER_NEAREST if self.label_mode == 'binary' else cv2.INTER_LINEAR
        mask = cv2.resize(mask, (resize_w, resize_h), interpolation=mask_interp)

        img = np.stack([img, img, img], axis=0)
        img = _normalize_image_to_dconnnet(img)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        if self.label_mode == 'binary':
            mask = (mask > 0.5).astype(np.float32)

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        if self.train:
            return img, mask
        return img, mask, self.name_ls[index]

    def __len__(self):
        return len(self.img_ls)


def getdataset_cremi(args, train_root, mode='train', label_mode='binary'):
    return MyDataset_CREMI(
        args=args,
        train_root=train_root,
        mode=mode,
        label_mode=label_mode,
    )
