"""Self-contained pure-PyTorch RRDBNet (the Real-ESRGAN ``RealESRGAN_x4plus`` generator).

This is a re-implementation of the RRDBNet architecture from ESRGAN /
BasicSR / Real-ESRGAN so that `escher` does not need to import ``basicsr`` or
``realesrgan`` (neither builds on recent Python versions). Module and
parameter names match the upstream implementation exactly
(``conv_first``, ``body.N.rdb{1,2,3}.conv{1..5}``, ``conv_body``,
``conv_up1/2``, ``conv_hr``, ``conv_last``), so the official
``RealESRGAN_x4plus.pth`` state dict loads with ``strict=True``.

Credit / licence:
    * Architecture: ESRGAN (Wang et al., 2018), as implemented in BasicSR
      (https://github.com/XPixelGroup/BasicSR, Apache-2.0) and Real-ESRGAN
      (https://github.com/xinntao/Real-ESRGAN, BSD-3-Clause).
    * This file is an independent minimal re-write of that architecture; the
      pretrained weights are NOT bundled (see `escher.sr`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """(b, c, h*s, w*s) -> (b, c*s^2, h, w); used only for scale < 4 variants."""
    return F.pixel_unshuffle(x, scale)


class ResidualDenseBlock(nn.Module):
    """Five densely connected 3x3 convs with a 0.2-scaled residual."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: three RDBs with a 0.2-scaled residual."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet generator. ``RealESRGAN_x4plus`` uses the defaults below.

    Input/output are RGB in [0, 1] (nominally), shapes (b,3,h,w) -> (b,3,h*s,w*s).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError(f"RRDBNet: scale must be 1, 2 or 4, got {scale}")
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsampling: two nearest-neighbour x2 stages + conv (=> x4 total).
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = _pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = _pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
