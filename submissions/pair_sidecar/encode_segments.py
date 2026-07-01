#!/usr/bin/env python3
"""Six-segment AV1 encode with per-segment CRF tuned for motion/content."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Frame cuts at 20fps matching ~0,24,30,36,42,48 seconds of driving footage.
SEGMENTS = [
  (0, 480, 34, 0.92),
  (480, 600, 36, 0.88),
  (600, 720, 35, 0.86),
  (720, 840, 36, 0.86),
  (840, 960, 35, 0.88),
  (960, 1200, 34, 0.90),
]


def run(cmd):
  subprocess.run(cmd, check=True)


def encode_segment(pre: Path, work: Path, start: int, end: int, crf: int, sat: float) -> Path:
  seg = work / f"seg_{start}_{end}.mkv"
  run([
    "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
    "-r", "20", "-i", str(pre),
    "-vf", (
      f"select='between(n\\,{start}\\,{end - 1})',setpts=N/FRAME_RATE/TB,"
      f"scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos,"
      f"eq=saturation={sat:.2f}"
    ),
    "-pix_fmt", "yuv420p", "-c:v", "libsvtav1", "-preset", "0", "-crf", str(crf),
    "-svtav1-params", "film-grain=20:keyint=180:scd=0",
    "-r", "20", str(seg),
  ])
  return seg


def concat_segments(segs: list[Path], out: Path, work: Path):
  lst = work / "concat.txt"
  lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
  run([
    "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
    "-f", "concat", "-safe", "0", "-i", str(lst),
    "-c", "copy", str(out),
  ])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--pre", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--work", type=Path, required=True)
  args = ap.parse_args()
  args.work.mkdir(parents=True, exist_ok=True)
  segs = [encode_segment(args.pre, args.work, s, e, crf, sat) for s, e, crf, sat in SEGMENTS]
  concat_segments(segs, args.out, args.work)


if __name__ == "__main__":
  main()
