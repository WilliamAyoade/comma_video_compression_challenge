#!/usr/bin/env python3
"""Compact pair-asymmetric sidecar: per-frame luma bias + dy/dx shift."""
from __future__ import annotations

import struct
from pathlib import Path

import torch

MAGIC = b"PASC"
HEADER = struct.Struct("<4sBH")  # magic, version, num_frames
ENTRY = struct.Struct("<bbb")  # luma_bias, dy, dx as int8


def encode_sidecar(params: list[tuple[int, int, int]], path: Path) -> None:
  """params: list of (luma_bias, dy, dx) per frame."""
  buf = bytearray()
  buf.extend(HEADER.pack(MAGIC, 1, len(params)))
  for luma, dy, dx in params:
    buf.extend(ENTRY.pack(
      max(-128, min(127, int(luma))),
      max(-128, min(127, int(dy))),
      max(-128, min(127, int(dx))),
    ))
  path.write_bytes(buf)


def decode_sidecar(path: Path) -> list[tuple[int, int, int]]:
  data = path.read_bytes()
  magic, version, n = HEADER.unpack_from(data, 0)
  if magic != MAGIC:
    raise ValueError(f"bad sidecar magic: {magic!r}")
  if version != 1:
    raise ValueError(f"unsupported sidecar version: {version}")
  off = HEADER.size
  params = []
  for _ in range(n):
    luma, dy, dx = ENTRY.unpack_from(data, off)
    if luma >= 128:
      luma -= 256
    if dy >= 128:
      dy -= 256
    if dx >= 128:
      dx -= 256
    params.append((luma, dy, dx))
    off += ENTRY.size
  return params


def shift_rgb(frame: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
  """frame: (H, W, 3) float."""
  if dy == 0 and dx == 0:
    return frame
  h, w, _ = frame.shape
  out = torch.zeros_like(frame)
  sy0 = max(0, dy)
  sy1 = min(h, h + dy)
  sx0 = max(0, dx)
  sx1 = min(w, w + dx)
  ty0 = max(0, -dy)
  ty1 = ty0 + (sy1 - sy0)
  tx0 = max(0, -dx)
  tx1 = tx0 + (sx1 - sx0)
  if sy1 > sy0 and sx1 > sx0:
    out[ty0:ty1, tx0:tx1] = frame[sy0:sy1, sx0:sx1]
  return out


def apply_correction(frame: torch.Tensor, luma: int, dy: int, dx: int) -> torch.Tensor:
  """frame: (H,W,3) uint8 -> uint8."""
  x = frame.float()
  if dy or dx:
    x = shift_rgb(x, dy, dx)
  if luma:
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    y = (y + luma).clamp(0, 255)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    old_y = 0.299 * r + 0.587 * g + 0.114 * b
    scale = (y / (old_y + 1e-3)).unsqueeze(-1)
    x = (x * scale).clamp(0, 255)
  return x.round().to(torch.uint8)
