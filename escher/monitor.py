"""escher.monitor -- optional, purely observational diagnostic monitor.

The braided sampler can be given a :class:`Monitor` to record the decoded clean
prediction ``x0`` (a pixel image) at chosen points of a run, each tagged with
its step index, sigma and a free-form label, and to assemble the frames into one
labelled montage PNG. It exists so a run that "looks wrong" can be inspected
from the inside -- above all, the ``pre``/``post`` frames around each warp show
what the conformal operator actually did.

The monitor never mutates the tensors it is shown (it converts to a small PIL
tile immediately), never touches an RNG, and is only constructed when the
caller opts in, so with monitoring off the sampler is unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont

TILE = 256          # max tile edge in the montage, in pixels
_LABEL_H = 26       # two text lines under each tile
_PAD = 4
_MAX_COLS = 6
_BG = (24, 24, 24)
_FG = (235, 235, 235)


@dataclass
class Frame:
    """One captured x0 frame: a downscaled tile plus its tags."""

    step: int
    sigma: float
    label: str
    tile: Image.Image


def _tile_of(x0_pixels: torch.Tensor) -> Image.Image:
    """``(1, 3, H, W)`` pixels in ``[-1, 1]`` -> a PIL tile of edge <= ``TILE``."""
    x = x0_pixels.detach().float().clamp(-1.0, 1.0)
    x = ((x + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    img = Image.fromarray(x[0].permute(1, 2, 0).cpu().numpy())
    img.thumbnail((TILE, TILE), Image.LANCZOS)
    return img


class Monitor:
    """Collects tagged x0 frames and renders them as a labelled montage."""

    def __init__(self, path: str, every: int) -> None:
        self.path = path
        self.every = every
        self.frames: List[Frame] = []

    def due(self, i: int) -> bool:
        """True when ordinary denoise step ``i`` should be captured."""
        return self.every > 0 and i % self.every == 0

    def capture(self, x0_pixels: torch.Tensor, step: int, sigma, label: str) -> None:
        self.frames.append(Frame(int(step), float(sigma), label, _tile_of(x0_pixels)))

    def save(self) -> str:
        """Lay the frames out on a grid with a text label per tile and save."""
        n = len(self.frames)
        if n == 0:
            raise ValueError("Monitor.save() called with no captured frames")
        cols = min(n, _MAX_COLS)
        rows = math.ceil(n / cols)
        tw = max(f.tile.width for f in self.frames)
        th = max(f.tile.height for f in self.frames)
        cw = tw + 2 * _PAD
        ch = th + _LABEL_H + 2 * _PAD
        canvas = Image.new("RGB", (cols * cw, rows * ch), _BG)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for k, fr in enumerate(self.frames):
            r, c = divmod(k, cols)
            x, y = c * cw + _PAD, r * ch + _PAD
            canvas.paste(fr.tile, (x, y))
            draw.text((x, y + fr.tile.height + 2), f"step {fr.step}  sigma {fr.sigma:.3f}", fill=_FG, font=font)
            draw.text((x, y + fr.tile.height + 14), fr.label, fill=_FG, font=font)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        canvas.save(self.path, format="PNG")
        return self.path
