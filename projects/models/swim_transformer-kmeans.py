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
        
        # 位置编码：
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

        # 对patch进行归一化
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



# Kmeans batchsize=1000

class DronePatchExtractor(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, num_classes=80):
        super().__init__()
        self.patch_size = self.to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 卷积操作，用于将图像划分为不重叠的patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_size[0], self.patch_size[1]))

        # 如果需要归一化
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def to_2tuple(self, x):
        return (x, x) if isinstance(x, int) else x

    def extract_drone_patches_with_bbox(self, images, labels):
        """
        根据标签中的边界框提取包含Drone的patch。

        Args:
            images (Tensor): 输入的图像数据，形状为 [B, C, H, W]
            labels (List[List[Tuple[int, int, int, int]]]): 每张图的目标边界框列表，
                格式为 [(xmin, ymin, w, h), ...]，每个图像的标签是一个边界框的列表。

        Returns:
            List[Tensor]: 含有目标Drone的patch列表
        """
        drone_patches = []
        patch_size = self.patch_size[0]

        for img, bbox_list in zip(images, labels):
            patches = self.forward(img.unsqueeze(0))  # 获取patch特征
            h_img, w_img = img.shape[1:]  # 获取图像尺寸

            # 遍历每个边界框（每张图可能有多个目标Drone）
            for bbox in bbox_list:
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h

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

    def cluster_patches(self, drone_patches, num_clusters=80, batch_size=1000):
        """
        对Drone patches进行两阶段聚类，首先对patch进行批次聚类，然后对所有聚类结果再次聚类。

        Args:
            drone_patches (List[Tensor]): 待聚类的Drone patches列表
            num_clusters (int): 聚类的数量
            batch_size (int): 每次聚类处理的patch数量

        Returns:
            List[List[Tensor]]: 最终聚类后的patch列表，每个元素是一个集群中所有patch的列表
        """
        # 第一阶段聚类结果
        clustered_patches = []  # 存储第一阶段聚类后的patch列表
        patch_batch = []  # 用于存储当前批次的patch

        # 遍历所有的patch
        for i, patch in enumerate(drone_patches):
            patch_batch.append(patch.flatten().cpu().numpy())  # 将每个patch展平并存储

            # 一旦积累的patch数量达到batch_size，就进行一次聚类
            if len(patch_batch) == batch_size or i == len(drone_patches) - 1:
                # 将当前批次的所有patch特征堆叠成二维数组（每个patch的特征是一维的）
                patch_features = np.stack(patch_batch)

                # 执行KMeans聚类
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(patch_features)

                # 根据聚类结果将patch分组
                for cluster_idx in range(num_clusters):
                    cluster_patches = [drone_patches[j] for j in range(len(patch_batch)) if kmeans.labels_[j] == cluster_idx]
                    clustered_patches.append(cluster_patches)

                # 处理完当前批次后，清空patch_batch，释放内存
                patch_batch.clear()

        # 第二阶段聚类：对第一阶段得到的所有聚类结果进行二次聚类
        # 将所有的集群patch展平为特征向量
        all_cluster_patches = []
        for cluster in clustered_patches:
            for patch in cluster:
                all_cluster_patches.append(patch.flatten().cpu().numpy())

        # 进行第二阶段KMeans聚类
        patch_features_final = np.stack(all_cluster_patches)
        kmeans_final = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans_final.fit(patch_features_final)

        # 将最终聚类的结果根据标签重新分组
        final_clustered_patches = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(kmeans_final.labels_):
            final_clustered_patches[label].append(all_cluster_patches[idx])

        return final_clustered_patches

    def forward(self, images, labels, num_clusters=80, batch_size=1000):
        """
        计算整个前向过程，包括提取Drone patches、聚类操作。

        Args:
            images (Tensor): 输入的图像数据，形状为 [B, C, H, W]
            labels (List[List[Tuple[int, int, int, int]]]): 每张图的目标边界框列表，
                格式为 [(xmin, ymin, w, h), ...]，每个图像的标签是一个边界框的列表。
            num_clusters (int): 聚类的数量
            batch_size (int): 每次聚类处理的patch数量

        Returns:
            List[List[Tensor]]: 最终聚类后的patch列表，每个元素是一个集群中所有patch的列表
        """
        # 步骤 1：根据标签提取Drone patches
        drone_patches = self.extract_drone_patches_with_bbox(images, labels)

        # 步骤 2：对提取的Drone patches进行聚类
        clustered_patches = self.cluster_patches(drone_patches, num_clusters, batch_size)

        return clustered_patches
    

class Reliabilty_Aware_Block(nn.Module):
    def __init__(self, input_dim, dropout, num_heads=8, dim_feedforward=128):
        super(Reliabilty_Aware_Block, self).__init__()
        self.conv_query = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)

        self.self_atten = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=0.1)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, attn_mask=None):
        src = features.permute(2, 0, 1)  # [T, B, F]
        q = k = src
        q = self.conv_query(features).permute(2, 0, 1)
        k = self.conv_key(features).permute(2, 0, 1)

        src2, attn = self.self_atten(q, k, src, attn_mask=attn_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.permute(1, 2, 0)  # [B, F, T]
        return src, attn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.feature_dim = args.feature_dim  # 输入特征维度

        # RAB args
        RAB_args = args.RAB_args
        self.RAB = nn.ModuleList([
            Reliabilty_Aware_Block(
                input_dim=self.feature_dim,
                dropout=RAB_args['drop_out'],
                num_heads=RAB_args['num_heads'],
                dim_feedforward=RAB_args['dim_feedforward'])
            for _ in range(RAB_args['layer_num'])
        ])

        # 特征嵌入层，通常用于映射特征
        self.feature_embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, input_features, prototypes=None):
        '''
        input_features: [B, T, F] -> 每个patch的特征，来自DronePatchExtractor的聚类结果
        prototypes：[C,1,F] -> 可学习的原型向量
        '''
        B, T, F = input_features.shape

        
        input_features = input_features.permute(0, 2, 1)                        #[B,F,T]
        prototypes = prototypes.to(input_features.device)                       #[C,1,F]
        prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]

        # 多层RAB处理
        for layer in self.RAB:
            input_features, _ = layer(input_features)

        # 最终特征嵌入
        embeded_features = self.feature_embedding(input_features)  # [B, F, T]

        return embeded_features