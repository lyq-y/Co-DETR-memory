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

from swin_transformer-original import *

class SwinTransformerBlockWithMemory(nn.Module):
    """ Swin Transformer Block with Memory Tokens.
    """
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer):
        super().__init__()
        # Existing components of SwinTransformerBlock
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

        # Memory tokens initialization (80 tokens as per your requirement)
        self.num_memory_tokens = 80
        self.memory_tokens = nn.Parameter(torch.zeros(1, self.num_memory_tokens, dim))  # 80 memory tokens

    def forward(self, x, H, W):
        """ Forward pass with image patches and memory tokens."""
        B, L, C = x.shape

        # Step 1: Concatenate the image patches and memory tokens
        memory_tokens = self.memory_tokens.expand(B, self.num_memory_tokens, C)  # B, 80, C
        x = torch.cat([x, memory_tokens], dim=1)  # B, L + 80, C (Concatenating patches and memory tokens)

        # Step 2: Perform attention with the combined tokens (image patches + memory tokens)
        x = self.norm1(x)
        attn_output = self.attn(x, H, W)  # Shape: B, L + 80, C

        # Step 3: Apply DropPath
        x = x + self.drop_path(attn_output)  # Residual connection

        # Step 4: MLP and another residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Step 5: Only keep the image patches (discard memory tokens)
        x = x[:, :-self.num_memory_tokens, :]  # Keep only the patches, discard the memory tokens

        return x, H, W  # Return only the image patches


# Modify your BasicLayer to use SwinTransformerBlockWithMemory
class BasicLayerWithMemory(nn.Module):
    """ A basic Swin Transformer layer for one stage with Memory Tokens."""
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlockWithMemory(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        
        # Downsampling layer (optional)
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function."""
        for blk in self.blocks:
            x, H, W = blk(x, H, W)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class SwinTransformerV1WithMemory(BaseModule):
    """ Swin Transformer backbone with Memory Tokens."""
    def __init__(self, ...):
        super().__init__()

        # Existing code to initialize patch embedding, position embedding, etc.

        # Create layers with memory tokens support
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerWithMemory(
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
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

    def forward(self, x):
        """ Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)
