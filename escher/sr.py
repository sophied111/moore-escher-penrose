"""Real-ESRGAN x4 super-resolution for the sampler's `--upres` hook.

`upscale4(x, mode)` is the sampler's upres entry point (see
`escher/sampler.py`, which lazily imports this module only when
`mode != "off"`). This module has NO optional dependencies: the RRDBNet
generator is vendored in pure PyTorch (`escher/_rrdbnet.py`), so neither
`basicsr` nor `realesrgan` is needed (both are unbuildable on Python 3.13).
The old `escher[sr]` extra is kept only as an empty, no-op alias.

Modes:
    "off"     -- identity, returns `x` unchanged.
    "bicubic" -- `F.interpolate(x, scale_factor=4, mode="bicubic")`.
    "sr"      -- Real-ESRGAN x4plus (RRDBNet). Deterministic (no RNG).

Tensor convention: input/output are `(1,3,H,W)` / `(1,3,4H,4W)` pixels in
`[-1,1]`, in ANY float dtype (fp16/bf16/fp32). For `"sr"` the bridge is
    x01 = (x + 1) / 2                      # [-1,1] -> [0,1]
    y01 = RRDBNet(x01.float()).clamp(0,1)  # GAN always runs in float32
    y   = (2 * y01 - 1).to(x.dtype)        # back to [-1,1], original dtype
and it runs on the input tensor's device (GPU if `x` is on GPU). To bound
memory on large inputs (e.g. 1024^2 -> 4096^2) inference is tiled
(`_TILE=400`, `_TILE_PAD=10`), with the padding cropped away when stitching.

Weights: `RealESRGAN_x4plus.pth` (Real-ESRGAN v0.1.0 release; state dict
stored under key `params_ema`). Resolved in this order:
    1. env `ESCHER_SR_WEIGHTS` (path to a local .pth; must exist),
    2. cache `~/.cache/escher/RealESRGAN_x4plus.pth`,
    3. otherwise downloaded (urllib) from
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
       into the cache (atomically). If the download fails (e.g. offline
       compute node) a RuntimeError tells you to fetch the file elsewhere and
       set `ESCHER_SR_WEIGHTS` to it.
No weights are committed to this repo. The built model is cached
module-globally across calls.

Architecture credit: ESRGAN / BasicSR (Apache-2.0) / Real-ESRGAN (BSD-3);
see `escher/_rrdbnet.py`.
"""
import os
import urllib.request

import torch
import torch.nn.functional as F

from escher._rrdbnet import RRDBNet

_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
_WEIGHTS_NAME = "RealESRGAN_x4plus.pth"
_ENV_VAR = "ESCHER_SR_WEIGHTS"

_TILE = 400  # tile side in input pixels (0 disables tiling)
_TILE_PAD = 10  # context pixels on each tile side, cropped after inference
_SCALE = 4

_MODEL = None  # cached RRDBNet (float32, eval), built lazily on first "sr" call


def _cache_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "escher", _WEIGHTS_NAME)


def _no_weights_error(detail: str) -> RuntimeError:
    return RuntimeError(
        f"Real-ESRGAN weights are not available ({detail}). Download "
        f"{_WEIGHTS_URL} on a machine with internet access and point the "
        f"{_ENV_VAR} environment variable at the local .pth file, e.g.\n"
        f"  export {_ENV_VAR}=/path/to/{_WEIGHTS_NAME}"
    )


def _resolve_weights() -> str:
    """Return a local path to RealESRGAN_x4plus.pth (env -> cache -> download)."""
    env = os.environ.get(_ENV_VAR)
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"{_ENV_VAR} is set to {env!r}, which is not a file.")
        return env

    path = _cache_path()
    if os.path.isfile(path):
        return path

    tmp = path + ".part"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(_WEIGHTS_URL, timeout=60) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, path)
    except Exception as exc:  # offline node, proxy, DNS, permissions, ...
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise _no_weights_error(f"download failed: {exc!r}") from exc
    return path


def _get_model() -> RRDBNet:
    """Lazily build and cache the RRDBNet x4plus with pretrained weights.

    Reference parity (``debug_conformal.upscale._get_vendored``): on CUDA the
    model is moved to the GPU and cast to **fp16** (``model.half()``); on CPU it
    stays float32. SR then runs in that dtype on the whole image (no tiling)."""
    global _MODEL
    if _MODEL is None:
        path = _resolve_weights()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for key in ("params_ema", "params"):
                if key in state:
                    state = state[key]
                    break
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_grow_ch=32, num_block=23, scale=_SCALE
        )
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        if torch.cuda.is_available():
            model = model.to("cuda").half()
        _MODEL = model
    return _MODEL


def _tiled_forward(model, x: torch.Tensor, tile: int, pad: int, scale: int = _SCALE):
    """Run `model` over (1,3,H,W) `x` in tiles; returns (1,3,H*scale,W*scale).

    Each tile is extended by up to `pad` context pixels per side (clamped to the
    image); the extra `pad*scale` output pixels are cropped before stitching, so
    tile borders do not show seams. `tile <= 0` (or a tile that covers the whole
    image) is a plain forward pass.
    """
    _, c, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return model(x)

    out = x.new_zeros((1, c, h * scale, w * scale))
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        py0, py1 = max(y0 - pad, 0), min(y1 + pad, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            px0, px1 = max(x0 - pad, 0), min(x1 + pad, w)
            y_t = model(x[:, :, py0:py1, px0:px1])
            oy0, ox0 = (y0 - py0) * scale, (x0 - px0) * scale
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = y_t[
                :, :, oy0 : oy0 + (y1 - y0) * scale, ox0 : ox0 + (x1 - x0) * scale
            ]
    return out


def sr_x4(img: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) [0,1] RGB -> (1,3,4H,4W) [0,1] RGB via Real-ESRGAN x4plus.

    Byte-for-byte matches the reference vendored path
    (``debug_conformal.upscale._sr_vendored``): clamp to [0,1], run on the model's
    device in the model's dtype (**fp16 on CUDA**, whole image -- no tiling),
    reflect-pad H/W up to an even size before the two x2 upsamples and crop the
    output back, then ``clamp(0,1).float()``.
    """
    assert img.dim() == 4 and img.shape[0] == 1 and img.shape[1] == 3, (
        f"sr_x4 expects (1,3,H,W), got {tuple(img.shape)}"
    )
    model = _get_model()
    p = next(model.parameters())
    x = img.clamp(0, 1).to(p.device)
    if p.dtype == torch.float16:
        x = x.half()
    _, _, h, w = x.shape
    ph, pw = (h % 2), (w % 2)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        out = model(x)
    if ph or pw:
        out = out[:, :, : h * _SCALE, : w * _SCALE]
    return out.clamp(0, 1).float()


def upscale4(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Upscale a (1,3,H,W) tensor in [-1,1] to (1,3,4H,4W) in [-1,1].

    mode:
        "off"     -- identity.
        "bicubic" -- bicubic interpolation, scale_factor=4.
        "sr"      -- Real-ESRGAN x4 (see module docstring for the [-1,1]/[0,1]
                     bridge, weights resolution and `ESCHER_SR_WEIGHTS`). The
                     result has the same dtype as `x`.
    """
    if mode == "off":
        return x
    if mode == "bicubic":
        return F.interpolate(x, scale_factor=4, mode="bicubic")
    if mode == "sr":
        # Clamp to [-1, 1] before the [0, 1] bridge, matching the reference SR
        # wrapper (step_modifiers.modify_x0: ``(out.clamp(-1, 1) + 1) / 2``).
        # The VAE decode overshoots [-1, 1]; without this clamp the RRDBNet
        # input differs on those pixels and the SR-warp op diverges bit-wise.
        # (bicubic mode intentionally has no clamp -- neither does the reference.)
        x01 = (x.float().clamp(-1, 1) + 1) / 2
        y01 = sr_x4(x01)
        return (2 * y01 - 1).to(x.dtype)
    raise ValueError(f"upscale4: unknown mode {mode!r}, expected one of 'off', 'bicubic', 'sr'")
