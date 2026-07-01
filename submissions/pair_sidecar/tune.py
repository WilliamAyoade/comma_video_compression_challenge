#!/usr/bin/env python3
"""Fast pair-asymmetric sidecar search with second-pass even-frame refine."""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import av
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb, seq_len
from modules import DistortionNet, segnet_sd_path, posenet_sd_path
from submissions.pair_sidecar.sidecar import apply_correction, encode_sidecar
from submissions.pair_sidecar.upscale import upscale_rgb


def load_video_rgb(path: Path, target_hw=None, upscale=False) -> list[torch.Tensor]:
  container = av.open(str(path))
  stream = container.streams.video[0]
  frames = []
  for i, frame in enumerate(container.decode(stream)):
    t = yuv420_to_rgb(frame)
    if target_hw and upscale:
      t = upscale_rgb(t, i, target_hw[0], target_hw[1])
    frames.append(t)
  container.close()
  return frames


def pair_score(net, gt_pair, comp_pair, device):
  g = gt_pair.unsqueeze(0).to(device)
  c = comp_pair.unsqueeze(0).to(device)
  pd, sd = net.compute_distortion(g, c)
  return 100 * sd.item() + math.sqrt(max(0.0, 10 * pd.item()))


def search_frame(net, gt0, gt1, c0, c1, device, fix_first, dy_cands, luma_cands, dx_cands=None):
  best = (0, 0, 0)
  best_score = float("inf")
  for dy in dy_cands:
    for luma in luma_cands:
      t0 = c0 if fix_first else apply_correction(c0, luma, dy, 0)
      t1 = apply_correction(c1, luma, dy, 0) if fix_first else c1
      score = pair_score(net, torch.stack([gt0, gt1]), torch.stack([t0, t1]), device)
      if score < best_score:
        best_score, best = score, (luma, dy, 0)
  if dx_cands:
    l, dy, _ = best
    for dx in dx_cands:
      if dx == 0:
        continue
      t0 = c0 if fix_first else apply_correction(c0, l, dy, dx)
      t1 = apply_correction(c1, l, dy, dx) if fix_first else c1
      score = pair_score(net, torch.stack([gt0, gt1]), torch.stack([t0, t1]), device)
      if score < best_score:
        best_score, best = score, (l, dy, dx)
  return best


def checkpoint_path(sidecar_out: Path) -> Path:
  return sidecar_out.with_name(f"{sidecar_out.stem}.tune.pkl")


def save_checkpoint(path: Path, params: list[tuple[int, int, int]], refine_from: int) -> None:
  path.write_bytes(pickle.dumps({"params": params, "refine_from": refine_from}))


def load_checkpoint(path: Path) -> tuple[list[tuple[int, int, int]], int] | None:
  if not path.is_file():
    return None
  data = pickle.loads(path.read_bytes())
  return data["params"], int(data.get("refine_from", 0))


def refine_local(net, gt0, gt1, c0, c1, device, fix_first, init):
  l, dy, dx = init
  best, best_s = init, float("inf")
  for dl in (-2, -1, 0, 1, 2):
    for ddy in (-2, -1, 0, 1, 2):
      for ddx in (-2, -1, 0, 1, 2):
        lu, y, x = l + dl, dy + ddy, dx + ddx
        t0 = c0 if fix_first else apply_correction(c0, lu, y, x)
        t1 = apply_correction(c1, lu, y, x) if fix_first else c1
        s = pair_score(net, torch.stack([gt0, gt1]), torch.stack([t0, t1]), device)
        if s < best_s:
          best_s, best = s, (lu, y, x)
  return best


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--source", type=Path, required=True)
  ap.add_argument("--encoded", type=Path, required=True)
  ap.add_argument("--sidecar-out", type=Path, required=True)
  ap.add_argument("--fast", action="store_true", help="coarse grid, no local refine or second pass (~5x faster)")
  ap.add_argument(
    "--resume-refine", action="store_true",
    help="skip pass-1 if checkpoint exists; resume second-pass refine",
  )
  ap.add_argument("--device", default="auto")
  args = ap.parse_args()

  if args.device == "auto":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(args.device)

  th, tw = camera_size[1], camera_size[0]
  gt_frames = load_video_rgb(args.source, (th, tw), upscale=False)
  dec_frames = load_video_rgb(args.encoded, (th, tw), upscale=True)

  net = DistortionNet().eval().to(device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

  ckpt = checkpoint_path(args.sidecar_out)
  loaded = load_checkpoint(ckpt) if args.resume_refine else None
  params = loaded[0] if loaded else [(0, 0, 0)] * len(dec_frames)
  refine_from = loaded[1] if loaded else 0
  n_pairs = len(dec_frames) // seq_len

  if args.fast:
    even_dy = [-6, -3, 0, 3, 6]
    even_lu = [-8, -4, 0, 4, 8]
    even_dx = [-2, 0, 2]
    odd_dy = [-2, 0, 2]
    odd_lu = [-4, 0, 4]
    odd_dx = None
  else:
    even_dy = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    even_lu = [-12, -8, -4, 0, 4, 8, 12]
    even_dx = [-4, -2, 0, 2, 4]
    odd_dy = [-3, -2, -1, 0, 1, 2, 3]
    odd_lu = [-6, -3, 0, 3, 6]
    odd_dx = [-1, 0, 1]

  if not (args.resume_refine and loaded):
    for p in tqdm(range(n_pairs), desc="tune pairs"):
      i0, i1 = p * 2, p * 2 + 1
      gt0, gt1 = gt_frames[i0], gt_frames[i1]
      c0, c1 = dec_frames[i0], dec_frames[i1]
      l0, dy0, dx0 = search_frame(net, gt0, gt1, c0, c1, device, False, even_dy, even_lu, even_dx)
      if not args.fast:
        l0, dy0, dx0 = refine_local(net, gt0, gt1, c0, c1, device, False, (l0, dy0, dx0))
      params[i0] = (l0, dy0, dx0)
      c0 = apply_correction(c0, l0, dy0, dx0)
      l1, dy1, dx1 = search_frame(net, gt0, gt1, c0, c1, device, True, odd_dy, odd_lu, odd_dx)
      if not args.fast:
        l1, dy1, dx1 = refine_local(net, gt0, gt1, c0, c1, device, True, (l1, dy1, dx1))
      params[i1] = (l1, dy1, dx1)
    if not args.fast:
      save_checkpoint(ckpt, params, 0)
      print(f"checkpoint pass-1 -> {ckpt}", flush=True)
  elif refine_from > 0:
    print(f"resuming refine from pair {refine_from}/{n_pairs}", flush=True)

  if not args.fast:
    # Second pass: re-refine even frames with odd correction fixed.
    for p in tqdm(range(refine_from, n_pairs), desc="refine even", initial=refine_from, total=n_pairs):
      i0, i1 = p * 2, p * 2 + 1
      gt0, gt1 = gt_frames[i0], gt_frames[i1]
      c0, c1 = dec_frames[i0], dec_frames[i1]
      l1, dy1, dx1 = params[i1]
      c1 = apply_correction(c1, l1, dy1, dx1)
      l0, dy0, dx0 = refine_local(net, gt0, gt1, c0, c1, device, False, params[i0])
      params[i0] = (l0, dy0, dx0)
      if (p + 1) % 25 == 0 or p + 1 == n_pairs:
        save_checkpoint(ckpt, params, p + 1)

  encode_sidecar(params, args.sidecar_out)
  ckpt.unlink(missing_ok=True)
  print(f"wrote sidecar ({len(params)} frames) -> {args.sidecar_out}")


if __name__ == "__main__":
  main()
