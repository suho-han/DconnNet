import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread, imsave
from torch.autograd import Function, Variable
from torch.nn.modules.loss import _Loss


def connectivity_matrix(multimask, class_num):

    ##### converting segmentation masks to directional affinity maps ####

    [batch, _, rows, cols] = multimask.shape
    # batch = 1
    conn = torch.zeros([batch, class_num*8, rows, cols]).cuda()
    for i in range(class_num):
        mask = multimask[:, i, :, :]
        # print(mask.shape)
        up = torch.zeros([batch, rows, cols]).cuda()  # move the orignal mask to up
        down = torch.zeros([batch, rows, cols]).cuda()
        left = torch.zeros([batch, rows, cols]).cuda()
        right = torch.zeros([batch, rows, cols]).cuda()
        up_left = torch.zeros([batch, rows, cols]).cuda()
        up_right = torch.zeros([batch, rows, cols]).cuda()
        down_left = torch.zeros([batch, rows, cols]).cuda()
        down_right = torch.zeros([batch, rows, cols]).cuda()

        up[:, :rows-1, :] = mask[:, 1:rows, :]
        down[:, 1:rows, :] = mask[:, 0:rows-1, :]
        left[:, :, :cols-1] = mask[:, :, 1:cols]
        right[:, :, 1:cols] = mask[:, :, :cols-1]
        up_left[:, 0:rows-1, 0:cols-1] = mask[:, 1:rows, 1:cols]
        up_right[:, 0:rows-1, 1:cols] = mask[:, 1:rows, 0:cols-1]
        down_left[:, 1:rows, 0:cols-1] = mask[:, 0:rows-1, 1:cols]
        down_right[:, 1:rows, 1:cols] = mask[:, 0:rows-1, 0:cols-1]

        conn[:, (i*8)+0, :, :] = mask*down_right
        conn[:, (i*8)+1, :, :] = mask*down
        conn[:, (i*8)+2, :, :] = mask*down_left
        conn[:, (i*8)+3, :, :] = mask*right
        conn[:, (i*8)+4, :, :] = mask*left
        conn[:, (i*8)+5, :, :] = mask*up_right
        conn[:, (i*8)+6, :, :] = mask*up
        conn[:, (i*8)+7, :, :] = mask*up_left

    conn = conn.float()
    conn = conn.squeeze()
    # keep batch axis for batch-size 1 to avoid shape mismatch in downstream BCE
    if conn.dim() == 3:
        conn = conn.unsqueeze(0)
    # print(conn.shape)
    return conn


def connectivity_matrix_5x5(multimask, class_num):

    ##### converting segmentation masks to directional affinity maps ####

    [batch, _, rows, cols] = multimask.shape
    # batch = 1
    conn = torch.zeros([batch, class_num*25, rows, cols]).cuda()
    for i in range(class_num):
        mask = multimask[:, i, :, :]
        # fill all 25 neighbors from a 5x5 window using offsets (dr, dc) in [2..-2]
        channel_idx = 0
        for dr in range(2, -3, -1):
            for dc in range(2, -3, -1):
                shifted = torch.zeros([batch, rows, cols]).cuda()

                if dr >= 0:
                    src_r0, src_r1 = 0, rows - dr
                    dst_r0, dst_r1 = dr, rows
                else:
                    src_r0, src_r1 = -dr, rows
                    dst_r0, dst_r1 = 0, rows + dr

                if dc >= 0:
                    src_c0, src_c1 = 0, cols - dc
                    dst_c0, dst_c1 = dc, cols
                else:
                    src_c0, src_c1 = -dc, cols
                    dst_c0, dst_c1 = 0, cols + dc

                if src_r1 > src_r0 and src_c1 > src_c0:
                    shifted[:, dst_r0:dst_r1, dst_c0:dst_c1] = mask[:, src_r0:src_r1, src_c0:src_c1]

                conn[:, (i*25)+channel_idx, :, :] = mask * shifted
                channel_idx += 1

    conn = conn.float()
    conn = conn.squeeze()
    # keep batch axis for batch-size 1 to avoid shape mismatch in downstream BCE
    if conn.dim() == 3:
        conn = conn.unsqueeze(0)
    # print(conn.shape)
    return conn


def shift_8_directions(x):
    # x: (B, 1, H, W)
    up = torch.zeros_like(x)
    down = torch.zeros_like(x)
    left = torch.zeros_like(x)
    right = torch.zeros_like(x)
    up_left = torch.zeros_like(x)
    up_right = torch.zeros_like(x)
    down_left = torch.zeros_like(x)
    down_right = torch.zeros_like(x)

    up[:, :, :-1, :] = x[:, :, 1:, :]
    down[:, :, 1:, :] = x[:, :, :-1, :]
    left[:, :, :, :-1] = x[:, :, :, 1:]
    right[:, :, :, 1:] = x[:, :, :, :-1]

    up_left[:, :, :-1, :-1] = x[:, :, 1:, 1:]
    up_right[:, :, :-1, 1:] = x[:, :, 1:, :-1]
    down_left[:, :, 1:, :-1] = x[:, :, :-1, 1:]
    down_right[:, :, 1:, 1:] = x[:, :, :-1, :-1]

    return [
        down_right, down, down_left,
        right, left,
        up_right, up, up_left
    ]


def distance_affinity_matrix(dist_map, sigma=2.0):
    """
    dist_map : (B, H, W) distance map for each class
    return: (B, 8*C, H, W) directional affinity map for each class

    우선 single-class만 구현
    """

    if dist_map.ndim == 4:
        if dist_map.shape[1] != 1:
            raise ValueError(f"single-class dist_map must have shape (B,1,H,W), got {dist_map.shape}")
        dist_map = dist_map[:, 0, :, :]
    elif dist_map.ndim != 3:
        raise ValueError(f"dist_map must have shape (B,H,W) or (B,1,H,W), got {dist_map.shape}")

    D = dist_map.unsqueeze(1)  # (B,1,H,W)

    neighbors_D = shift_8_directions(D)

    conn_list = []
    for D_k in neighbors_D:
        Ck = D * D_k * torch.exp(-torch.abs(D - D_k) / sigma)
        conn_list.append(Ck)

    conn = torch.cat(conn_list, dim=1)   # (B,8,H,W)
    return conn


def Bilateral_voting(c_map, hori_translation, verti_translation):

    #### bilateral voting and convert directional affinity into a distance score map. ####

    batch, class_num, channel, row, column = c_map.size()
    vote_out = torch.zeros([batch, class_num, channel, row, column]).cuda()
    # print(vote_out.shape,affinity_map.shape,hori_translation.shape)
    right = (torch.bmm(c_map[:, :, 4].contiguous().view(-1, row, column), hori_translation.view(-1, column, column))).view(batch, class_num, row, column)

    left = (torch.bmm(c_map[:, :, 3].contiguous().view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)

    left_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), c_map[:, :, 5].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    left_bottom = (torch.bmm(left_bottom.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)
    right_above = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 2].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_above = (torch.bmm(right_above.view(-1, row, column), hori_translation.view(-1, column, column))).view(batch, class_num, row, column)
    left_above = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 0].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    left_above = (torch.bmm(left_above.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)
    bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), c_map[:, :, 6].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    up = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 1].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), c_map[:, :, 7].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_bottom = (torch.bmm(right_bottom.view(-1, row, column), hori_translation.view(-1, column, column))).view(batch, class_num, row, column)

    vote_out[:, :, 0] = (c_map[:, :, 0]) * (right_bottom)
    vote_out[:, :, 1] = (c_map[:, :, 1]) * (bottom)
    vote_out[:, :, 2] = (c_map[:, :, 2]) * (left_bottom)
    vote_out[:, :, 3] = (c_map[:, :, 3]) * (right)
    vote_out[:, :, 4] = (c_map[:, :, 4]) * (left)
    vote_out[:, :, 5] = (c_map[:, :, 5]) * (right_above)
    vote_out[:, :, 6] = (c_map[:, :, 6]) * (up)
    vote_out[:, :, 7] = (c_map[:, :, 7]) * (left_above)

    pred_mask, _ = torch.max(vote_out, dim=2)
    ###
    # vote_out = vote_out.view(batch,-1, row, column)
    return pred_mask, vote_out


def shift_map(x, hori_translation, verti_translation, dy, dx):
    # x: (N, H, W)

    out = x

    if hori_translation.dim() == 4:
        hori_translation = hori_translation.squeeze(1)
    if verti_translation.dim() == 4:
        verti_translation = verti_translation.squeeze(1)

    if hori_translation.dim() != 3 or verti_translation.dim() != 3:
        raise ValueError("translation tensors must be 3D or 4D")

    n = out.shape[0]
    if hori_translation.shape[0] != n:
        if hori_translation.shape[0] == 1:
            hori_translation = hori_translation.repeat(n, 1, 1)
        elif n % hori_translation.shape[0] == 0:
            hori_translation = hori_translation.repeat_interleave(n // hori_translation.shape[0], dim=0)
        else:
            raise ValueError(f"hori_translation batch mismatch: got {hori_translation.shape[0]}, expected {n}")

    if verti_translation.shape[0] != n:
        if verti_translation.shape[0] == 1:
            verti_translation = verti_translation.repeat(n, 1, 1)
        elif n % verti_translation.shape[0] == 0:
            verti_translation = verti_translation.repeat_interleave(n // verti_translation.shape[0], dim=0)
        else:
            raise ValueError(f"verti_translation batch mismatch: got {verti_translation.shape[0]}, expected {n}")

    # vertical shift
    if dy > 0:
        for _ in range(dy):
            out = torch.bmm(verti_translation.transpose(2, 1), out)
    elif dy < 0:
        for _ in range(-dy):
            out = torch.bmm(verti_translation, out)

    # horizontal shift
    if dx > 0:
        for _ in range(dx):
            out = torch.bmm(out, hori_translation)
    elif dx < 0:
        for _ in range(-dx):
            out = torch.bmm(out, hori_translation.transpose(2, 1))

    return out


def Bilateral_voting_kxk(affinity_map, hori_translation, verti_translation, conn_num=5, second_weight=1.0):
    radius = conn_num // 2
    # affinity_map: (B, C, K, H, W), K can be either (2r+1)^2-1 or (2r+1)^2

    B, C, K, H, W = affinity_map.shape

    full_k = (2 * radius + 1) ** 2
    without_center_k = full_k - 1
    if K == full_k:
        include_center = True
    elif K == without_center_k:
        include_center = False
    else:
        raise ValueError(
            f"affinity_map channel mismatch for conn_num={conn_num}: "
            f"got K={K}, expected {without_center_k} or {full_k}"
        )

    # Channel order is aligned with connectivity_matrix_5x5 loop order:
    # dr from +radius to -radius, dc from +radius to -radius.
    offsets = []
    for dy in range(radius, -radius - 1, -1):
        for dx in range(radius, -radius - 1, -1):
            if not include_center and dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))

    if K != len(offsets):
        raise ValueError(f"Internal offset size mismatch: K={K}, offsets={len(offsets)}")

    offset_to_idx = {o: i for i, o in enumerate(offsets)}

    vote_out = torch.zeros_like(affinity_map)

    for dy, dx in offsets:
        idx = offset_to_idx[(dy, dx)]
        rev_idx = offset_to_idx[(-dy, -dx)]

        rev_map = affinity_map[:, :, rev_idx].contiguous().view(-1, H, W)
        shifted_rev = shift_map(rev_map, hori_translation, verti_translation, dy, dx)
        shifted_rev = shifted_rev.view(B, C, H, W)

        vote_out[:, :, idx] = affinity_map[:, :, idx] * shifted_rev

    pred_mask, _ = torch.max(vote_out, dim=2)
    return pred_mask, vote_out


class dice_loss(nn.Module):
    def __init__(self, bin_wide, density):
        super(dice_loss, self).__init__()
        self.bin_wide = bin_wide
        self.density = density

    def soft_dice_coeff(self, y_pred, y_true, class_i=None):
        smooth = 0.0001  # may change

        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))

        score = (2. * intersection + smooth) / (i + j + smooth)

        if self.bin_wide:
            weight = density_weight(self.bin_wide[class_i], i, self.density[class_i])
            return (1-score)*weight
        else:
            return (1-score)

    def soft_dice_loss(self, y_pred, y_true, class_i=None):
        loss = self.soft_dice_coeff(y_true, y_pred, class_i)
        return loss.mean()

    def __call__(self, y_pred, y_true, class_i=None):

        b = self.soft_dice_loss(y_true, y_pred, class_i)
        return b


def weighted_log_loss(output, weight, target):

    # print(weight.min())
    # target = target.type(torch.LongTensor).cuda()
    log_out = F.binary_cross_entropy(output, target, reduction='none')
    # log_out = F.binary_cross_entropy(output,target,reduction='none')
    loss = torch.mean(log_out*weight)
    # print(log_out.shape,weight.shape)

    return loss


def density_weight(bin_wide, gt_cnt, density):

    index = gt_cnt//bin_wide

    selected_density = [density[index[i].long()] for i in range(gt_cnt.shape[0])]
    selected_density = torch.tensor(selected_density).cuda()
    log_inv_density = torch.log(1/(selected_density+0.0001))

    return log_inv_density


class connect_loss(nn.Module):
    def __init__(self, args, hori_translation, verti_translation, density=None, bin_wide=None, label_mode=None, conn_num=8, sigma=2.0):
        super(connect_loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.BCEloss = nn.BCELoss(reduction='none')
        self.dice_loss = dice_loss(bin_wide=bin_wide, density=density)
        self.bin_wide = bin_wide
        self.density = density
        self.args = args

        self.verti_translation = verti_translation
        self.hori_translation = hori_translation

        self.label_mode = label_mode
        self.conn_num = conn_num
        self.sigma = sigma
        self.sml1_loss = nn.SmoothL1Loss(reduction='none')
        self.dist_aux_loss_name = getattr(self.args, 'dist_aux_loss', 'smooth_l1')
        self.dist_sf_l1_gamma = getattr(self.args, 'dist_sf_l1_gamma', 1.0)
        self._collect_dist_edge_stats = False
        self._dist_edge_stats = {
            'edge_mean_sum': 0.0,
            'edge_nonzero_sum': 0.0,
            'count': 0,
        }

    def set_dist_edge_stat_collection(self, enabled):
        self._collect_dist_edge_stats = bool(enabled)

    def reset_dist_edge_stats(self):
        self._dist_edge_stats = {
            'edge_mean_sum': 0.0,
            'edge_nonzero_sum': 0.0,
            'count': 0,
        }

    def get_dist_edge_stats(self):
        count = self._dist_edge_stats['count']
        if count == 0:
            return None
        return {
            'edge_mean': self._dist_edge_stats['edge_mean_sum'] / count,
            'edge_nonzero_ratio': self._dist_edge_stats['edge_nonzero_sum'] / count,
            'count': count,
        }

    def dist_target_to_mask(self, target):
        # CHASE distance labels are positive on vessel pixels and zero on background.
        return (target > 0).float()

    def gjml_loss(self, pred, target, eps=1e-8):
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        sum_norm = torch.sum(torch.abs(pred_flat + target_flat), dim=1)
        diff_norm = torch.sum(torch.abs(pred_flat - target_flat), dim=1)
        jaccard_term = (sum_norm - diff_norm) / (sum_norm + diff_norm + eps)
        return (1 - jaccard_term).mean()

    def stable_focal_l1_loss(self, pred, target, gamma):
        diff = torch.abs(target - pred)
        indicator = (target * pred >= 0).float()
        return (diff * torch.pow(diff, gamma) * indicator).mean()

    def dist_aux_regression_loss(self, pred, target):
        if self.dist_aux_loss_name == 'smooth_l1':
            return self.sml1_loss(pred, target).mean()
        if self.dist_aux_loss_name == 'gjml_sf_l1':
            return self.gjml_loss(pred, target) + self.stable_focal_l1_loss(
                pred, target, gamma=self.dist_sf_l1_gamma
            )
        raise ValueError(
            f"Unsupported dist_aux_loss {self.dist_aux_loss_name}, expected 'smooth_l1' or 'gjml_sf_l1'"
        )

    def binary_edge_target_from_affinity(self, affinity_target):
        # `affinity_target` here is expected to be a binary-style connectivity map.
        # For distance mode, pass connectivity rebuilt from `mask_target`, not the
        # raw `distance_affinity_matrix(...)` output.
        class_conn = affinity_target.view(
            [
                affinity_target.shape[0],
                self.args.num_class,
                self.conn_num,
                affinity_target.shape[2],
                affinity_target.shape[3],
            ]
        )
        sum_conn = torch.sum(class_conn, dim=2)
        return torch.where(
            (sum_conn < self.conn_num) & (sum_conn > 0),
            torch.full_like(sum_conn, 1),
            torch.full_like(sum_conn, 0),
        )

    def edge_loss(self, vote_out, edge):
        pred_mask_min, _ = torch.min(vote_out.cuda(), dim=2)
        pred_mask_min = pred_mask_min*edge
        minloss = self.BCEloss(pred_mask_min, torch.full_like(pred_mask_min, 0))
        return (minloss.sum()/(pred_mask_min.sum().clamp(min=1e-8)))  # +maxloss

    def dist_edge_loss(self, vote_out, mask_target):
        # Distance mode still optimizes continuous directional affinity, but the
        # final segmentation target is binary. Build edge supervision from the
        # binary vessel mask instead of the weak continuous 4n(1-n) target.
        # Rebuild binary connectivity from the derived vessel mask before
        # computing the edge target. This keeps the edge definition aligned with
        # the binary path instead of reinterpreting raw distance affinity values.
        binary_affinity_target = connectivity_matrix(mask_target, self.args.num_class)
        edge = self.binary_edge_target_from_affinity(binary_affinity_target)
        return self.edge_loss(vote_out, edge), edge

    def forward(self, affinity_map, target, return_details=False):
        if self.args.num_class == 1:
            loss, loss_dict = self.single_class_forward(affinity_map, target)
        else:
            loss, loss_dict = self.multi_class_forward(affinity_map, target)

        if return_details:
            return loss, loss_dict
        return loss

    def multi_class_forward(self, affinity_map, target):
        #######
        # affinity_map: (B, 8*C, H, W), B: batch, C: class number
        # target: (B, H, W)
        #######
        target = target.type(torch.LongTensor).cuda()
        batch_num = affinity_map.shape[0]
        onehotmask = F.one_hot(target.long(), self.args.num_class)  # change it to your class number if needed
        onehotmask = onehotmask.permute(0, 3, 1, 2)
        onehotmask = onehotmask.float()

        ### build directional affinity target ###
        affinity_target = connectivity_matrix(onehotmask, self.args.num_class)  # (B, 8*C, H, W)

        # matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num, 1, 1, 1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num, 1, 1, 1).cuda()
        # shape for hori_translation: [Batch, NumClass, H, H]
        # shape for verti_translation: [Batch, NumClass, W, W]

        dice_l = 0

        ### get edges gt###
        class_conn = affinity_target.view([affinity_map.shape[0], self.args.num_class, 8, affinity_map.shape[2], affinity_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)
        # edge: (B, C, H, W)
        edge = torch.where((sum_conn < 8) & (sum_conn > 0), torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))

        ### bilateral voting #####
        # final pred: (B, C, H, W), vote_out: (B, C, 8, H, W), bicon_map: (B, 8*C, H, W)
        class_pred = affinity_map.view([affinity_map.shape[0], self.args.num_class, 8, affinity_map.shape[2], affinity_map.shape[3]])
        final_pred, vote_out = Bilateral_voting(class_pred, hori_translation, verti_translation)
        _, bicon_map = Bilateral_voting(F.sigmoid(class_pred), hori_translation, verti_translation)
        bicon_map = bicon_map.view(affinity_target.shape)

        edge_l = self.edge_loss(F.sigmoid(vote_out), edge)

        pred = F.softmax(final_pred, dim=1)

        for j in range(1, self.args.num_class):
            dice_l += self.dice_loss(pred[:, j, :, :], onehotmask[:, j], j-1)

        ce_loss = F.cross_entropy(final_pred, target)
        affinity_l = self.BCEloss(F.sigmoid(affinity_map), affinity_target).mean()
        bicon_l = self.BCEloss(bicon_map, affinity_target).mean()
        loss = ce_loss + affinity_l + edge_l + 0.2 * bicon_l + dice_l  # + bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice
        loss_dict = {
            'total': loss,
            'ce': ce_loss,
            'affinity': affinity_l,
            'edge': edge_l,
            'bicon': bicon_l,
            'dice': dice_l,
        }

        return loss, loss_dict

    def single_class_forward(self, c_map, target):
        #######
        # affinity_map: (B, 8, H, W), B: batch, C: class number
        # target: (B, 1, H, W)
        #######

        batch_num = c_map.shape[0]
        target = target.float()

        ### build directional affinity target ###
        if self.label_mode == 'binary':
            if self.conn_num == 8:
                con_target = connectivity_matrix(target, self.args.num_class)  # (B, 8, H, W)
            elif self.conn_num == 25:
                con_target = connectivity_matrix_5x5(target, self.args.num_class)  # (B, 25, H, W)
            else:
                raise ValueError(f"Unsupported conn_num {self.conn_num}, only 8 and 25 are supported")
            affinity_target = con_target
        else:
            if self.conn_num != 8:
                raise ValueError("distance label modes currently support only conn_num=8")
            affinity_target = distance_affinity_matrix(target, sigma=self.sigma)  # (B, 8, H, W)

        # matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num, 1, 1, 1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num, 1, 1, 1).cuda()

        c_map = F.sigmoid(c_map)

        ### get edges gt###
        class_conn = affinity_target.view([c_map.shape[0], self.args.num_class, self.conn_num, c_map.shape[2], c_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)

        # edge: (B, 1, H, W)

        ### bilateral voting #####
        # pred: (B, 1, H, W) score map (distance score map in distance label_mode),
        # bicon_map: (B, 1, 8, H, W) directional affinity map
        class_pred = c_map.view([c_map.shape[0], self.args.num_class, self.conn_num, c_map.shape[2], c_map.shape[3]])
        if self.conn_num == 8:
            pred, bicon_map = Bilateral_voting(class_pred, hori_translation, verti_translation)
        elif self.conn_num == 25:
            kxk_size = int(math.sqrt(self.conn_num))
            if kxk_size * kxk_size != self.conn_num:
                raise ValueError(f"conn_num={self.conn_num} is not a perfect square for kxk voting")
            pred, bicon_map = Bilateral_voting_kxk(class_pred, hori_translation, verti_translation, conn_num=kxk_size)
        else:
            raise ValueError(f"Unsupported conn_num {self.conn_num}, only 8 and 25 are supported")

        if self.label_mode == 'binary':
            edge = torch.where((sum_conn < self.conn_num) & (sum_conn > 0), torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))
            edge_l = self.edge_loss(bicon_map, edge)
            dice_l = self.dice_loss(pred[:, 0], target[:, 0])

            bce_loss = self.BCEloss(pred, target).mean()
            conn_l = self.BCEloss(c_map, con_target).mean()

            if self.args.dataset == 'chase':
                loss = bce_loss + conn_l + edge_l + dice_l
                bicon_l = torch.zeros_like(loss)
            else:
                bicon_l = self.BCEloss(bicon_map.squeeze(1), con_target).mean()
                loss = bce_loss + conn_l + edge_l + 0.2 * bicon_l + dice_l  # + bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice
            loss_dict = {
                'total': loss,
                'vote': bce_loss,
                'affinity': conn_l,
                'edge': edge_l,
                'bicon': bicon_l,
                'dice': dice_l,
            }

        elif self.label_mode in ['dist_signed', 'dist_inverted']:
            mask_target = self.dist_target_to_mask(target)
            edge_l, edge = self.dist_edge_loss(bicon_map, mask_target)

            if self._collect_dist_edge_stats:
                with torch.no_grad():
                    edge_detached = edge.detach()
                    self._dist_edge_stats['edge_mean_sum'] += edge_detached.mean().item()
                    self._dist_edge_stats['edge_nonzero_sum'] += (edge_detached > 1e-6).float().mean().item()
                    self._dist_edge_stats['count'] += 1

            # Binary path predicts the final vessel mask; keep that property for
            # distance mode as well, while using the distance map as an affinity
            # regression target.
            vote_loss = self.BCEloss(pred, mask_target).mean()
            dice_l = self.dice_loss(pred[:, 0], mask_target[:, 0])
            affinity_l = self.dist_aux_regression_loss(c_map, affinity_target)

            bicon_map = bicon_map.view(affinity_target.shape)   # (B,8,H,W)
            if self.args.dataset == 'chase':
                bicon_l = torch.zeros_like(vote_loss)
                loss = vote_loss + affinity_l + edge_l + dice_l
            else:
                bicon_l = self.dist_aux_regression_loss(bicon_map, affinity_target)
                loss = vote_loss + affinity_l + edge_l + 0.2 * bicon_l + dice_l
            loss_dict = {
                'total': loss,
                'vote': vote_loss,
                'affinity': affinity_l,
                'edge': edge_l,
                'bicon': bicon_l,
                'dice': dice_l,
            }

        return loss, loss_dict
