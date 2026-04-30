# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
import math

import autorootcwd
import torch
import torch.nn as nn
import torchsummary
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

import model.gap as gap
from model.attention import CAM_Module, PAM_Module
from model.resnet import resnet34

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

LEGACY_CONN_FUSION_ALIASES = {
    'decoder_guided': 'dg',
}


def normalize_conn_fusion_mode(mode):
    normalized = str(mode if mode is not None else 'none')
    return LEGACY_CONN_FUSION_ALIASES.get(normalized, normalized)

OUTER_8_NATIVE_ORDER = [
    (-2, -2),
    (-2, 0),
    (-2, 2),
    (0, -2),
    (0, 2),
    (2, -2),
    (2, 0),
    (2, 2),
]

OUTER_8_STANDARD_ORDER = [
    (2, 2),
    (2, 0),
    (2, -2),
    (0, 2),
    (0, -2),
    (-2, 2),
    (-2, 0),
    (-2, -2),
]

_OUTER_8_NATIVE_POS = {offset: idx for idx, offset in enumerate(OUTER_8_NATIVE_ORDER)}
OUTER_8_TO_STANDARD8_INDEX = [_OUTER_8_NATIVE_POS[offset] for offset in OUTER_8_STANDARD_ORDER]


def reorder_outer8_to_standard8(logits, index=OUTER_8_TO_STANDARD8_INDEX):
    if logits.dim() != 4:
        raise ValueError(f"Expected 4D logits tensor (B,C,H,W), got {logits.shape}")
    if logits.shape[1] % 8 != 0:
        raise ValueError(f"Channel dimension must be divisible by 8, got {logits.shape[1]}")

    group_count = logits.shape[1] // 8
    if group_count == 1:
        return logits[:, index, :, :]

    view = logits.view(logits.shape[0], group_count, 8, logits.shape[2], logits.shape[3])
    gathered = view[:, :, index, :, :]
    return gathered.reshape(logits.shape[0], logits.shape[1], logits.shape[2], logits.shape[3])


def fuse_directional_logits(
    c3_logits,
    c5_aligned_logits,
    mode,
    residual_scale=0.2,
    gate_conv=None,
    residual_conv=None,
):
    if c3_logits.shape != c5_aligned_logits.shape:
        raise ValueError(
            f"Shape mismatch for fusion: c3={tuple(c3_logits.shape)}, c5={tuple(c5_aligned_logits.shape)}"
        )

    if mode == 'gate':
        if gate_conv is None:
            raise ValueError("gate_conv is required for mode='gate'")
        alpha = torch.sigmoid(gate_conv(torch.cat([c3_logits, c5_aligned_logits], dim=1)))
        return alpha * c3_logits + (1.0 - alpha) * c5_aligned_logits
    if mode == 'scaled_sum':
        return c3_logits + float(residual_scale) * c5_aligned_logits
    if mode == 'conv_residual':
        if residual_conv is None:
            raise ValueError("residual_conv is required for mode='conv_residual'")
        return c3_logits + residual_conv(c5_aligned_logits)
    raise ValueError(f"Unsupported conn_fusion mode: {mode}")


class DecoderGuidedResidualFusion(nn.Module):
    def __init__(self, dec_ch, conn_ch=8, hidden_ch=64):
        super().__init__()
        in_ch = dec_ch + conn_ch + conn_ch

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(hidden_ch, conn_ch, kernel_size=1),
            nn.Sigmoid(),
        )

        self.residual = nn.Conv2d(hidden_ch, conn_ch, kernel_size=1)

    def forward(self, D, C3, C5):
        if C3.shape != C5.shape:
            raise ValueError(
                f"C3 and C5 must have the same shape for decoder-guided residual fusion, "
                f"got C3={C3.shape}, C5={C5.shape}"
            )

        if D.shape[-2:] != C3.shape[-2:]:
            raise ValueError(
                f"Decoder feature and connectivity maps must have the same spatial size, "
                f"got D={D.shape}, C3={C3.shape}"
            )

        x = torch.cat([D, C3, C5], dim=1)
        h = self.shared(x)
        beta = self.gate(h)
        residual = self.residual(h)
        c_fused = C3 + beta * residual
        return c_fused, beta


class ConnectivitySegHead(nn.Module):
    def __init__(self, dec_ch, conn_ch=8, hidden_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(dec_ch + conn_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, D, C_fused):
        if D.shape[-2:] != C_fused.shape[-2:]:
            raise ValueError(
                f"Decoder feature and fused connectivity must have the same spatial size, "
                f"got D={D.shape}, C_fused={C_fused.shape}"
            )
        x = torch.cat([D, C_fused], dim=1)
        return self.head(x)


class DecoderOnlySegHead(nn.Module):
    def __init__(self, dec_ch, hidden_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(dec_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, D):
        return self.head(D)


class DconnNet(nn.Module):
    def __init__(
        self,
        num_class=1,
        conn_num=8,
        conn_layout=None,
        conn_fusion='none',
        fusion_residual_scale=0.2,
        use_seg_aux=False,
    ):
        super(DconnNet, self).__init__()
        from connect_loss import resolve_connectivity_layout

        self.num_class = num_class
        self.connectivity_layout = resolve_connectivity_layout(conn_num, conn_layout)
        self.conn_num = self.connectivity_layout['channel_count']
        self.conn_fusion = normalize_conn_fusion_mode(conn_fusion)
        self.fusion_residual_scale = float(fusion_residual_scale)
        self.use_seg_aux = use_seg_aux

        allowed_fusion_modes = {'none', 'gate', 'scaled_sum', 'conv_residual', 'dg', 'dg_direct'}
        if self.conn_fusion not in allowed_fusion_modes:
            raise ValueError(
                f"Unsupported conn_fusion mode: {self.conn_fusion} "
                f"(supported: {sorted(allowed_fusion_modes)})"
            )
        self.seg_aux_mode = 'decoder_only' if self.conn_fusion == 'dg_direct' else 'connectivity'

        out_planes = num_class * self.conn_num
        self.backbone = resnet34(pretrained=True)
        self.sde_module = SDE_module(512, 512, out_planes, self.conn_num)
        self.fb5 = FeatureBlock(512, 256, relu=False, last=True)  # 256
        self.fb4 = FeatureBlock(256, 128, relu=False)  # 128
        self.fb3 = FeatureBlock(128, 64, relu=False)  # 64
        self.fb2 = FeatureBlock(64, 64)

        self.gap = gap.GlobalAvgPool2D()

        self.sb1 = SpaceBlock(512, 512, 512)
        self.sb2 = SpaceBlock(512, 256, 256)
        self.sb3 = SpaceBlock(256, 128, 128)
        self.sb4 = SpaceBlock(128, 64, 64)
        # self.sb5 = SpaceBlock(64,64,32)

        self.relu = nn.ReLU()

        self.final_decoder = LWdecoder(in_channels=[64, 64, 128, 256], out_channels=32, in_feat_output_strides=(4, 8, 16, 32), out_feat_output_stride=4,
                                       norm_fn=nn.BatchNorm2d, num_groups_gn=None)

        self.cls_pred_conv = nn.Conv2d(64, 32, 3, 1, 1)
        self.cls_pred_conv_2 = nn.Conv2d(32, out_planes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(512, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

        self.direc_reencode = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1),
            # nn.BatchNorm2d(out_planes),
            # nn.ReLU(True)
        )

        if self.conn_fusion != 'none':
            if self.num_class != 1:
                raise ValueError("conn_fusion currently supports only num_class=1")
            if self.conn_num != 8:
                raise ValueError("conn_fusion currently supports only conn_num=8")
            if self.connectivity_layout['name'] != 'standard8':
                raise ValueError("conn_fusion currently supports only conn_layout='standard8'")

            self.cls_pred_inner_conv = nn.Conv2d(32, out_planes, 1)
            self.cls_pred_outer_conv = nn.Conv2d(32, out_planes, 1)

            if self.conn_fusion == 'gate':
                self.fusion_gate_conv = nn.Conv2d(out_planes * 2, out_planes, 1)
            elif self.conn_fusion == 'conv_residual':
                self.fusion_residual_conv = nn.Conv2d(out_planes, out_planes, 1)
            elif self.conn_fusion in {'dg', 'dg_direct'}:
                self.decoder_guided_fusion = DecoderGuidedResidualFusion(dec_ch=32, conn_ch=out_planes, hidden_ch=64)

        if self.use_seg_aux:
            if self.seg_aux_mode == 'decoder_only':
                self.seg_head = DecoderOnlySegHead(dec_ch=32, hidden_ch=64)
            else:
                self.seg_head = ConnectivitySegHead(dec_ch=32, conn_ch=out_planes, hidden_ch=64)

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64

        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512

        #### directional Prior ####
        directional_c5 = self.channel_mapping(c5)
        mapped_c5 = F.interpolate(
            directional_c5, scale_factor=32, mode='bilinear', align_corners=True)
        mapped_c5 = self.direc_reencode(mapped_c5)

        d_prior = self.gap(mapped_c5)

        c5 = self.sde_module(c5, d_prior)

        c6 = self.gap(c5)

        r5 = self.sb1(c6, c5)

        d4 = self.relu(self.fb5(r5)+c4)  # 256
        r4 = self.sb2(self.gap(r5), d4)

        d3 = self.relu(self.fb4(r4)+c3)  # 128
        r3 = self.sb3(self.gap(r4), d3)

        d2 = self.relu(self.fb3(r3)+c2)  # 64
        r2 = self.sb4(self.gap(r3), d2)

        d1 = self.fb2(r2)+c1  # 32
        # d1 = self.sr5(c6,d1)

        feat_list = [d1, d2, d3, d4, c5]

        final_feat = self.final_decoder(feat_list)

        if self.conn_fusion != 'none':
            c3_logits = self.cls_pred_inner_conv(final_feat)
            c3_logits = self.upsample4x_op(c3_logits)

            c5_logits_native = self.cls_pred_outer_conv(final_feat)
            c5_logits_native = self.upsample4x_op(c5_logits_native)
            c5_logits_aligned = reorder_outer8_to_standard8(c5_logits_native)

            if self.conn_fusion in {'dg', 'dg_direct'}:
                final_feat_up = self.upsample4x_op(final_feat)
                c_fused_logits, beta = self.decoder_guided_fusion(final_feat_up, c3_logits, c5_logits_aligned)
            else:
                c_fused_logits = fuse_directional_logits(
                    c3_logits=c3_logits,
                    c5_aligned_logits=c5_logits_aligned,
                    mode=self.conn_fusion,
                    residual_scale=self.fusion_residual_scale,
                    gate_conv=getattr(self, 'fusion_gate_conv', None),
                    residual_conv=getattr(self, 'fusion_residual_conv', None),
                )

            output_dict = {
                'fused': c_fused_logits,
                'inner': c3_logits,
                'outer': c5_logits_native,
                'outer_aligned': c5_logits_aligned,
                'aux': mapped_c5,
            }
            if self.conn_fusion in {'dg', 'dg_direct'}:
                output_dict['fusion_gate'] = beta

            if getattr(self, 'use_seg_aux', False):
                final_feat_up = self.upsample4x_op(final_feat)
                if self.seg_aux_mode == 'decoder_only':
                    output_dict['mask_logit'] = self.seg_head(final_feat_up)
                else:
                    output_dict['mask_logit'] = self.seg_head(final_feat_up, c_fused_logits)

            return output_dict

        cls_pred = self.cls_pred_conv_2(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)

        output_dict = {
            'fused': cls_pred,
            'aux': mapped_c5,
        }

        if getattr(self, 'use_seg_aux', False):
            final_feat_up = self.upsample4x_op(final_feat)
            output_dict['mask_logit'] = self.seg_head(final_feat_up, cls_pred)

        return output_dict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
#        return F.logsigmoid(main_out,dim=1)


class SDE_module(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, conn_num):
        super(SDE_module, self).__init__()
        self.conn_num = conn_num
        self.inter_channels = in_channels // conn_num

        for i in range(conn_num):
            setattr(self, 'att'+str(i+1), DANetHead(self.inter_channels, self.inter_channels))
        # self.att1 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att2 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att3 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att4 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att5 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att6 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att7 = DANetHead(self.inter_channels, self.inter_channels)
        # self.att8 = DANetHead(self.inter_channels, self.inter_channels)

        # For conn_num values that do not divide in_channels (e.g., 24 for 512),
        # concatenated directional features have conn_num * inter_channels channels.
        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(self.inter_channels * self.conn_num, out_channels, 1))
        # self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))

        if num_class < 32:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, num_class*conn_num, 1),
                nn.ReLU(True),
                nn.Conv2d(num_class*conn_num, in_channels, 1))
        else:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, in_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 1))

    def forward(self, x, d_prior):

        ### re-order encoded_c5 ###
        # new_order = [0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31]
        # # print(encoded_c5.shape)
        # re_order_d_prior = d_prior[:,new_order,:,:]
        # print(d_prior)
        enc_feat = self.reencoder(d_prior)

        feats = []
        for i in range(self.conn_num):
            feats.append(getattr(self, 'att'+str(i+1))(x[:, i*self.inter_channels:(i+1)*self.inter_channels], enc_feat[:, i*self.inter_channels:(i+1)*self.inter_channels]))
        feat = torch.cat(feats, dim=1)

        # feat1 = self.att1(x[:, :self.inter_channels], enc_feat[:, 0:self.inter_channels])
        # feat2 = self.att2(x[:, self.inter_channels:2*self.inter_channels], enc_feat[:, self.inter_channels:2*self.inter_channels])
        # feat3 = self.att3(x[:, 2*self.inter_channels:3*self.inter_channels], enc_feat[:, 2*self.inter_channels:3*self.inter_channels])
        # feat4 = self.att4(x[:, 3*self.inter_channels:4*self.inter_channels], enc_feat[:, 3*self.inter_channels:4*self.inter_channels])
        # feat5 = self.att5(x[:, 4*self.inter_channels:5*self.inter_channels], enc_feat[:, 4*self.inter_channels:5*self.inter_channels])
        # feat6 = self.att6(x[:, 5*self.inter_channels:6*self.inter_channels], enc_feat[:, 5*self.inter_channels:6*self.inter_channels])
        # feat7 = self.att7(x[:, 6*self.inter_channels:7*self.inter_channels], enc_feat[:, 6*self.inter_channels:7*self.inter_channels])
        # feat8 = self.att8(x[:, 7*self.inter_channels:8*self.inter_channels], enc_feat[:, 7*self.inter_channels:8*self.inter_channels])

        # feat = torch.cat([feat1, feat2, feat3, feat4,
        #                  feat5, feat6, feat7, feat8], dim=1)

        sasc_output = self.final_conv(feat)
        sasc_output = sasc_output+x

        return sasc_output


class DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv

        feat_sum = feat_sum*F.sigmoid(enc_feat)

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class SpaceBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_in,
                 out_channels,
                 scale_aware_proj=False):
        super(SpaceBlock, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

        self.content_encoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.feature_reencoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features):
        content_feats = self.content_encoders(features)

        scene_feat = self.scene_encoder(scene_feature)
        relations = self.normalizer(
            (scene_feat * content_feats).sum(dim=1, keepdim=True))

        p_feats = self.feature_reencoders(features)

        refined_feats = relations * p_feats

        return refined_feats


class LWdecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(LWdecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError(
                    'When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn,
                                num_channels=out_channels)
        else:
            raise ValueError(
                'Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        dec_level = 0
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - \
                int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels[dec_level] if idx ==
                              0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(
                        **norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(
                        scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))
            dec_level += 1

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


class FeatureBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(FeatureBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


# if __name__ == '__main__':


#    model = DconnNet()
#    torchsummary.summary(model, (3, 512, 512))
