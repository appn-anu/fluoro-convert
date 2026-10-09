#!/usr/bin/env python3
"""Debug tool: dump every frame of a .DAT file as a 16-bit TIFF.

Usage: dat2tiff.py FILE.DAT [OUTPUT_DIR]

Frames are written unmodified (no scaling, junk pixels left in) as
<name>_f0.tiff, <name>_f1.tiff, ... next to the input, or in OUTPUT_DIR.
Requires numpy and Pillow.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# .DAT layout: headerless uint16 LE, frames of 2048 x 1500, each stored column-major.
W, H = 2048, 1500
FRAME_BYTES = W * H * 2


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else src.parent

    size = src.stat().st_size
    if size == 0 or size % FRAME_BYTES:
        sys.exit(f"{src}: {size} bytes is not a whole number of {W} x {H} 16-bit frames")

    frames = np.fromfile(src, dtype="<u2").reshape(-1, W, H).transpose(0, 2, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        dst = out_dir / f"{src.stem}_f{i}.tiff"
        Image.fromarray(np.ascontiguousarray(frame)).save(dst)
        print(f"{dst}  min={frame.min()} max={frame.max()} mean={frame.mean():.1f}")


if __name__ == "__main__":
    main()
