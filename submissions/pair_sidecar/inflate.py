#!/usr/bin/env python
import sys
from pathlib import Path

import av

from frame_utils import camera_size, yuv420_to_rgb
from submissions.pair_sidecar.sidecar import apply_correction, decode_sidecar
from submissions.pair_sidecar.upscale import upscale_rgb


def decode_with_sidecar(video_path: str, sidecar_path: str, dst: str):
  target_w, target_h = camera_size
  params = decode_sidecar(Path(sidecar_path))
  container = av.open(video_path)
  stream = container.streams.video[0]
  with open(dst, "wb") as f:
    for i, frame in enumerate(container.decode(stream)):
      t = yuv420_to_rgb(frame)
      t = upscale_rgb(t, i, target_h, target_w)
      if i < len(params):
        t = apply_correction(t, *params[i])
      f.write(t.contiguous().numpy().tobytes())
  container.close()


if __name__ == "__main__":
  decode_with_sidecar(sys.argv[1], sys.argv[2], sys.argv[3])
