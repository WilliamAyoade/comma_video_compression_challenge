"""Corridor-weighted upscale used by tune and inflate."""
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

_CORRIDOR = [
  (0, 299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)]),
  (300, 599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)]),
  (600, 899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)]),
  (900, 1199, [(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)]),
]
_FALLBACK = [(0.15, 0.52), (0.85, 0.48), (1.00, 1.00), (0.00, 1.00)]


def corridor_mask(idx, w, h):
  poly = _FALLBACK
  for lo, hi, pts in _CORRIDOR:
    if lo <= idx <= hi:
      poly = pts
      break
  img = Image.new("L", (w, h), 0)
  ImageDraw.Draw(img).polygon([(x * w, y * h) for x, y in poly], fill=255)
  img = img.filter(ImageFilter.GaussianBlur(radius=16))
  m = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).clone()
  return (m.view(h, w).float() / 255.0)


def upscale_rgb(t: torch.Tensor, idx: int, target_h: int, target_w: int) -> torch.Tensor:
  H, W, _ = t.shape
  if H == target_h and W == target_w:
    return t
  x = t.permute(2, 0, 1).unsqueeze(0).float()
  bicubic = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
  bilinear = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
  mask = corridor_mask(idx, target_w, target_h).unsqueeze(0).unsqueeze(0)
  mixed = bicubic * mask + bilinear * (1.0 - mask)
  return mixed.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
