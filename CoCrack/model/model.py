import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from typing import Callable
import torch.nn.functional as F


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]
        SHAPE_FIX[0] = H // 2
        SHAPE_FIX[1] = W // 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        if norm_layer is not None:
            self.norm = norm_layer(dim // dim_scale)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale*dim_scale*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)
        return x


class MsCK(nn.Module):
    def __init__(self,
                 in_channels=0,
                 groups=1,
                 conv_bias=True,
                 small_kernel=3,
                 large_kernel=7,
                 middle_kernel=5):
        super().__init__()
        self.large_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            bias=conv_bias,
            kernel_size=large_kernel,
            padding=(large_kernel - 1) // 2,
        )
        self.middle_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            bias=conv_bias,
            kernel_size=middle_kernel,
            padding=(middle_kernel - 1) // 2,
        )
        self.small_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            bias=conv_bias,
            kernel_size=small_kernel,
            padding=(small_kernel - 1) // 2,
        )

    def forward(self, x):
        return self.large_conv(x) + self.small_conv(x) + self.middle_conv(x) + x


class RepGiB(nn.Module):
    def __init__(
            self,
            dim=96,
            expand_ratio=2.0,
            act_layer=nn.SiLU,
            small_kernel=3,
            large_kernel=7,
            middle_kernel=5,
            conv_bias=True,
            bias=False,
    ):
        super().__init__()
        dim_expand = int(expand_ratio * dim)
        self.in_proj = nn.Linear(dim, dim_expand * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.ms_conv = MsCK(in_channels=dim_expand,
                            groups=dim_expand,
                            conv_bias=conv_bias,
                            small_kernel=small_kernel,
                            large_kernel=large_kernel,
                            middle_kernel=middle_kernel)
        self.out_proj = nn.Linear(dim_expand, dim, bias=bias)
        self.out_norm = nn.LayerNorm(dim_expand)

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        z = self.act(z)
        res = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        x = self.ms_conv(x)
        x = self.act(x)
        x = self.out_norm(x.permute(0, 2, 3, 1)).view(B, H, W, -1)
        x = (x + res) * z
        x = self.out_proj(x)
        return x


class RepGiB_Block(nn.Module):
    def __init__(
        self,
        hidden_dim=0,
        drop_path=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        expand_ratio=2,
        small_kernel=3,
        large_kernel=7,
        middle_kernel=5,
        att_name=RepGiB,
        act=nn.SiLU,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = att_name(
            dim=hidden_dim,
            expand_ratio=expand_ratio,
            small_kernel=small_kernel,
            large_kernel=large_kernel,
            middle_kernel=middle_kernel,
            act_layer=act,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class RepGiB_Layer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        expand_ratio=2,
        small_kernel=3,
        large_kernel=7,
        middle_kernel=5,
        youhua_kernel=0,
        block_name=RepGiB_Block,
        att_name=RepGiB,
        multi_out=False,
        act=nn.SiLU,
    ):
        super().__init__()
        self.dim = dim
        self.multi_out = multi_out
        self.blocks = nn.ModuleList([
            block_name(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                expand_ratio=expand_ratio,
                small_kernel=small_kernel,
                large_kernel=large_kernel,
                middle_kernel=middle_kernel,
                att_name=att_name,
                act=act,
            )
            for i in range(depth)
        ])
        
    def forward(self, x):
        out = []
        for blk in self.blocks:
            x = blk(x)
            out.append(x)
        if self.multi_out:
            return out
        else:
            return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SiLKAN(nn.Module):
    def __init__(self, dim=0, depth1=0, depth2=0, expand_ratio=0, drop_path=0,
                 large_kernel=0, small_kernel=0, middle_kernel=0,
                 att_name=0, att_name1=0, act=nn.SiLU):
        super().__init__()
        self.large = RepGiB_Layer(
            dim=dim,
            depth=depth1,
            expand_ratio=expand_ratio,
            drop_path=drop_path,
            large_kernel=large_kernel,
            small_kernel=small_kernel,
            middle_kernel=middle_kernel,
            att_name=att_name1,
            norm_layer=nn.LayerNorm,
            act=act,
        )
        self.middle = RepGiB_Layer(
            dim=dim,
            depth=depth1,
            expand_ratio=expand_ratio,
            drop_path=drop_path,
            large_kernel=large_kernel - 6,
            small_kernel=small_kernel - 6,
            middle_kernel=middle_kernel - 6,
            att_name=att_name,
            multi_out=False,
            norm_layer=nn.LayerNorm,
            act=act,
        )
        self.samll = RepGiB_Layer(
            dim=dim,
            depth=depth1,
            expand_ratio=expand_ratio,
            drop_path=drop_path,
            large_kernel=large_kernel - 12,
            small_kernel=small_kernel - 12,
            middle_kernel=middle_kernel - 12,
            att_name=att_name,
            multi_out=False,
            norm_layer=nn.LayerNorm,
            act=act,
        )
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.ATT1 = nn.Sequential(FRAM(dim, 16))
        self.ATT2 = nn.Sequential(FRAM(dim, 16))
        self.ATT3 = nn.Sequential(FRAM(dim, 16))

    def forward(self, x):
        xs = self.samll(x)
        xm = self.middle(xs)
        xl = self.large(xm)
        x = self.ATT1(xm) + self.ATT2(xs) + self.ATT3(xl)
        x = self.linear(x)
        return x


class FRAM(nn.Module):
    def __init__(self, input_channels, ratio):
        super(FRAM, self).__init__()
        if input_channels // ratio < 1:
            self.down = nn.Linear(input_channels, 1, bias=True)
            self.nonlinear = nn.SiLU(inplace=True)
            self.up = nn.Linear(1, input_channels, bias=True)
            self.LIP = FRAM_block(input_channels, 11, 5)  
        else:
            self.down = nn.Linear(input_channels, input_channels // ratio, bias=True)
            self.nonlinear = nn.SiLU(inplace=True)
            self.up = nn.Linear(input_channels // ratio, input_channels, bias=True)
            self.LIP = FRAM_block(input_channels, 11, 5)

    def forward(self, inputs):
        x = self.LIP(inputs)
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = inputs * x 
        return x


class SpatialAttention(nn.Module):
    def __init__(self, inch, kernel_size=3, padding=1):
        super(SpatialAttention, self).__init__()
        self.conv11 = nn.Conv2d(1, 1, kernel_size + 4, padding=padding + 2, bias=False)

    def forward(self, x):
        shape = x.shape
        x1 = torch.mean(x, dim=1, keepdim=True)
        x1 = self.conv11(x1)
        return x1


class FRAM_block(nn.Module):
    def __init__(self, in_chans=3, sa_ks=11, sa_pa=5):
        super().__init__()
        self.logits = SpatialAttention(in_chans, sa_ks, sa_pa)
        self.logitsra = SpatialAttention(in_chans, sa_ks, sa_pa)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        weights = self.logits(x)
        weights = torch.sigmoid(weights).mul(12)
        weights = weights.exp()
        weightsra = self.logitsra(x)
        weightsra = (1 - torch.sigmoid(weightsra)).mul(12)
        weightsra = weightsra.exp()
        out1 = torch.mean(x * weights, dim=[2, 3], keepdim=True) / torch.mean(weights, dim=[2, 3], keepdim=True)
        out2 = torch.mean(x * weightsra, dim=[2, 3], keepdim=True) / torch.mean(weightsra, dim=[2, 3], keepdim=True)
        return (out1 + out2).permute(0, 2, 3, 1)


class CoCrack(nn.Module):
    def __init__(self,
                 dim: list = [96, 192, 384, 192, 96],
                 depth=[[1, 2], [1, 2, 1, 2, 1, 2], [1, 2], [1, 2,], [1, 2]],
                 expand_ratio=[2, 2, 2, 2, 2],
                 large_kernel=[7, 21, 21, 7, 7],
                 middle_kernel=[5, 19, 19, 5, 5],
                 small_kernel=[3, 17, 17, 3, 3],
                 drop_path=0.1,
                 att_name=RepGiB,
                 att_name1=RepGiB,
                 att_name2=RepGiB,
                 act=nn.SiLU,
                 ):
        super(CoCrack, self).__init__()
        self.channel = dim
        self.norm = nn.LayerNorm
        self.stem = PatchEmbed2D(4, 3, dim[0], self.norm)
        self.block0 = nn.Sequential(
            RepGiB_Layer(dim[0], depth=depth[0][1], expand_ratio=expand_ratio[0], drop_path=drop_path,
                                   small_kernel=small_kernel[0], large_kernel=large_kernel[0],
                                   middle_kernel=middle_kernel[0], att_name=att_name1, act=act),
        )
        self.down0 = PatchMerging2D(dim[0], self.norm)
        self.block1 = nn.Sequential(
            SiLKAN(dim[1], depth1=depth[1][0], depth2=depth[1][1],
                                               expand_ratio=expand_ratio[1], drop_path=drop_path,
                                               small_kernel=small_kernel[1], large_kernel=large_kernel[1],
                                               middle_kernel=middle_kernel[1], att_name=att_name, att_name1=att_name2,
                                               act=act),
            SiLKAN(dim[1], depth1=depth[1][2], depth2=depth[1][3],
                                               expand_ratio=expand_ratio[1], drop_path=drop_path,
                                               small_kernel=small_kernel[1], large_kernel=large_kernel[1],
                                               middle_kernel=middle_kernel[1], att_name=att_name, att_name1=att_name2,
                                               act=act),
            SiLKAN(dim[1], depth1=depth[1][4], depth2=depth[1][5],
                                               expand_ratio=expand_ratio[1], drop_path=drop_path,
                                               small_kernel=small_kernel[1], large_kernel=large_kernel[1],
                                               middle_kernel=middle_kernel[1], att_name=att_name, att_name1=att_name2,
                                               act=act),
        )
        self.down1 = PatchMerging2D(dim[1], self.norm)
        self.block2 = nn.Sequential(
            SiLKAN(dim[2], depth1=depth[2][0], depth2=depth[2][1],
                                               expand_ratio=expand_ratio[2], drop_path=drop_path,
                                               small_kernel=small_kernel[2], large_kernel=large_kernel[2],
                                               middle_kernel=middle_kernel[2], att_name=att_name, att_name1=att_name2,
                                               act=act),
        )
        self.patchup3 = PatchExpand(dim[2], 2, norm_layer=self.norm)
        self.patchup4 = PatchExpand(dim[1], 2, norm_layer=self.norm)
        self.upconv3 = nn.Linear(dim[2], dim[1])
        self.upconv4 = nn.Linear(dim[1], dim[0])
        self.up3 = nn.Sequential(
            RepGiB_Layer(dim[1], depth=depth[3][1], expand_ratio=expand_ratio[3], drop_path=drop_path,
                                   small_kernel=small_kernel[3], large_kernel=large_kernel[3],
                                   middle_kernel=middle_kernel[3], att_name=att_name1, act=act),
        )
        self.up4 = nn.Sequential(
            RepGiB_Layer(dim[0], depth=depth[4][1], expand_ratio=expand_ratio[4], drop_path=drop_path,
                                   small_kernel=small_kernel[4], large_kernel=large_kernel[4],
                                   middle_kernel=middle_kernel[4], att_name=att_name1, act=act),
        )
        self.finalup = FinalPatchExpand_X4(dim[0], dim_scale=4, norm_layer=self.norm)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.block0(x1)
        x2 = self.down0(x1)
        x2 = self.block1(x2)  
        x3 = self.down1(x2)
        x3 = self.block2(x3)
        d3 = self.patchup3(x3)
        d3 = self.up3(self.upconv3(torch.cat((d3, x2), dim=-1)))
        d4 = self.patchup4(d3)
        d4 = self.up4(self.upconv4(torch.cat((d4, x1), dim=-1)))
        out = self.finalup(d4)
        out = out.permute(0, 3, 1, 2)
        return out,