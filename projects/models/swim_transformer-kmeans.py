# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES

from mmcv.runner import BaseModule
from sklearn.cluster import KMeans


#stage 1

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class DronePatchExtractor(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, num_classes=80):
        super().__init__()
        
        self.patch_size = self.to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 卷积操作，用于将图像划分为不重叠的patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # 位置编码：简单的learnable位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_size[0], self.patch_size[1]))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def to_2tuple(self, x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, x):
        # 获取图像的patch
        patches = self.proj(x)  # [B, embed_dim, H/p, W/p]

        # 加入位置编码
        patches += self.pos_embed

        # 展平每个patch，并加上位置编码
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # 对patch进行归一化（如果需要）
        if self.norm:
            patches = self.norm(patches)
        
        return patches

    def extract_drone_patches_with_bbox(self, images, bboxes):
        """
        根据目标范围提取包含Drone的patch。

        Args:
            images (Tensor): 输入的图像数据，形状为 [B, C, H, W]
            bboxes (List[Tuple[int, int, int, int]]): 每张图的目标边界框列表，
                格式为 [(xmin, ymin, w, h), ...]

        Returns:
            List[Tensor]: 含有目标Drone的patch列表
        """
        drone_patches = []
        patch_size = self.patch_size[0]

        for img, bbox in zip(images, bboxes):
            # 转换 bbox 到 (xmin, ymin, xmax, ymax)
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h

            patches = self.forward(img.unsqueeze(0))  # 获取patch特征

            h_img, w_img = img.shape[1:]  # 获取图像尺寸

            # 遍历每个patch
            for i in range(0, h_img - patch_size + 1, patch_size):
                for j in range(0, w_img - patch_size + 1, patch_size):
                    # 计算当前patch的范围
                    patch_x_min = j
                    patch_y_min = i
                    patch_x_max = j + patch_size
                    patch_y_max = i + patch_size

                    # 判断是否与目标范围有交集
                    if not (patch_x_max <= xmin or patch_x_min >= xmax or
                            patch_y_max <= ymin or patch_y_min >= ymax):
                        # 如果有交集，将patch保存
                        drone_patches.append(patches[0, :, i // patch_size, j // patch_size])

        return drone_patches


    def cluster_patches(self, drone_patches, num_clusters=80):
        """
        对含有目标Drone的patch进行聚类。
        
        Args:
            drone_patches (List[Tensor]): 含有目标Drone的patch列表
            num_clusters (int): 聚类数量
            
        Returns:
            List[Tensor]: 聚类后的patch
        """
        # 将所有patch展平为二维数据，以便进行聚类
        patch_features = [patch.flatten().cpu().numpy() for patch in drone_patches]
        patch_features = np.stack(patch_features)

        # 使用K-means进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(patch_features)

        # 获取聚类结果
        clustered_patches = []
        for cluster_idx in range(num_clusters):
            # 获取属于当前聚类的所有patch
            cluster_patches = [drone_patches[i] for i in range(len(drone_patches)) if kmeans.labels_[i] == cluster_idx]
            clustered_patches.append(cluster_patches)

        return clustered_patches

@BACKBONES.register_module()
class SwinTransformerV1(BaseModule):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pretrained=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(pretrained, str):
    #         self.apply(_init_weights)
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         self.apply(_init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def init_weights(self):
        """Initialize the weights in backbone."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformerV1, self).train(mode)
        self._freeze_stages()