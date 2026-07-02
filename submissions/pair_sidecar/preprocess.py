#!/usr/bin/env python3
"""YUV corridor denoise with pair-asymmetric outside blend."""
import argparse
import sys
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
from frame_utils import yuv420_to_rgb

_CORRIDOR = [
  (0, 299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)]),
  (300, 599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)]),
  (600, 899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)]),
  (900, 1199, [(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)]),
]
_FALLBACK = [(0.15, 0.52), (0.85, 0.48), (1.00, 1.00), (0.00, 1.00)]


def corridor_mask(idx, w, h, feather=22):
  poly = _FALLBACK
  for lo, hi, pts in _CORRIDOR:
    if lo <= idx <= hi:
      poly = pts
      break
  img = Image.new("L", (w, h), 0)
  ImageDraw.Draw(img).polygon([(x * w, y * h) for x, y in poly], fill=255)
  if feather > 0:
    img = img.filter(ImageFilter.GaussianBlur(radius=feather))
  m = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).clone()
  return (m.view(h, w).float() / 255.0).unsqueeze(0).unsqueeze(0)


def rgb_to_yuv(x):
  r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
  y = 0.299 * r + 0.587 * g + 0.114 * b
  u = (b - y) / 1.772 + 128.0
  v = (r - y) / 1.402 + 128.0
  return torch.cat([y, u, v], dim=1)


def yuv_to_rgb(yuv):
  y, u, v = yuv[:, 0:1], yuv[:, 1:2] - 128.0, yuv[:, 2:3] - 128.0
  return torch.cat([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], dim=1)


def luma_blur(yuv, strength):
  if strength <= 0:
    return yuv
  ks = 3 if strength <= 2.0 else 5
  sigma = max(0.1, strength * 0.35)
  coords = torch.arange(ks).float() - ks // 2
  g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
  k1 = g / g.sum()
  k2 = torch.outer(k1, k1).view(1, 1, ks, ks)
  y = yuv[:, 0:1]
  y_blur = F.conv2d(y, k2, padding=ks // 2)
  mix = min(0.9, strength / 3.0)
  yuv = yuv.clone()
  yuv[:, 0:1] = (1 - mix) * y + mix * y_blur
  return yuv


def chroma_pool(yuv, mode):
  if mode == "normal":
    return yuv
  k = {"soft": 1, "medium": 2, "strong": 4}[mode]
  uv = yuv[:, 1:3]
  uv = F.avg_pool2d(uv, kernel_size=k * 2 + 1, stride=1, padding=k)
  yuv = yuv.clone()
  yuv[:, 1:3] = uv
  return yuv


def process_frame(rgb_u8, idx):
  x = rgb_u8.permute(2, 0, 1).float().unsqueeze(0)
  mask = corridor_mask(idx, x.shape[-1], x.shape[-2])
  luma_s = 2.3 if idx % 2 == 0 else 1.8
  chroma_m = "strong" if idx % 2 == 0 else "medium"
  outside = 0.78 if idx % 2 == 0 else 0.38
  yuv = rgb_to_yuv(x)
  yuv = luma_blur(yuv, luma_s)
  yuv = chroma_pool(yuv, chroma_m)
  smooth = yuv_to_rgb(yuv)
  alpha = (1.0 - mask) * outside
  mixed = x * (1.0 - alpha) + smooth * alpha
  return mixed.clamp(0, 255).round().to(torch.uint8).squeeze(0).permute(1, 2, 0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", type=Path, required=True)
  ap.add_argument("--output", type=Path, required=True)
  args = ap.parse_args()
  src = av.open(str(args.input))
  st_in = src.streams.video[0]
  dst = av.open(str(args.output), mode="w")
  st_out = dst.add_stream("ffv1", rate=20)
  st_out.width, st_out.height, st_out.pix_fmt = st_in.width, st_in.height, "yuv420p"
  for i, frame in enumerate(src.decode(st_in)):
    out = process_frame(yuv420_to_rgb(frame), i)
    vf = av.VideoFrame.from_ndarray(out.cpu().numpy(), format="rgb24")
    for pkt in st_out.encode(vf):
      dst.mux(pkt)
  for pkt in st_out.encode():
    dst.mux(pkt)
  dst.close()
  src.close()


if __name__ == "__main__":
  main()
