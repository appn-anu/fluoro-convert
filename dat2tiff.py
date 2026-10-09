#!/usr/bin/env python3
"""Debug tool: dump every frame of a .DAT file as a 16-bit TIFF.

Usage: dat2tiff.py FILE.DAT [OUTPUT_DIR]

Frames are written unmodified (no scaling, junk pixels left in) as
<name>_f0.tiff, <name>_f1.tiff, ... next to the input, or in OUTPUT_DIR.
Frame size comes from the matching HDR_*.INF if it is next to the input,
and is inferred from the data otherwise.
Requires numpy and Pillow.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# .DAT layout: headerless uint16 LE, frames concatenated, each stored column-major.
# Frame size depends on sensor binning. Smallest first - see infer_size().
SIZES = [(2048, 1500), (4096, 3000)]
# Neighbouring columns of a correctly sized frame correlate at about 0.99;
# a too-small guess gives 0.4 or less.
MIN_CORRELATION = 0.7


def frame_bytes(size):
    return size[0] * size[1] * 2


def header_size(src):
    """Frame size from the capture's header file, or None if there isn't one."""
    hdr = src.with_name("HDR_" + src.stem.partition("_")[2] + ".INF")
    if not hdr.is_file():
        return None
    keys = dict(line.strip().split("=", 1) for line in hdr.read_text(errors="replace").splitlines() if "=" in line)
    try:
        return int(keys["ImageCols"]), int(keys["ImageRows"])
    except (KeyError, ValueError):
        return None


def column_correlation(data, size):
    """Best correlation between neighbouring columns over the first few frames.

    Each column has its own mean removed first, so this compares the profiles down the
    columns. Otherwise a wrong size can score well just because both halves of the
    image are bright in the same columns.
    """
    w, h = size
    best = 0.0
    for i in range(min(4, data.size // (w * h))):
        cols = data[i * w * h:(i + 1) * w * h].reshape(w, h)
        # every 8th column against its immediate right-hand neighbour; row 0 has junk pixels
        a = cols[0:-1:8, 1:].astype(np.float64)
        b = cols[1::8, 1:].astype(np.float64)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den:
            best = max(best, (a * b).sum() / den)
    return best


def infer_size(data):
    """Pick the frame size the data fits, or None.

    The same byte count can be several small frames or fewer large ones. Small frames
    read as large still look smooth, but not the other way round, so the smallest size
    whose columns line up is the right one.
    """
    fits = [s for s in SIZES if data.nbytes and data.nbytes % frame_bytes(s) == 0]
    for size in fits[:-1]:
        if column_correlation(data, size) >= MIN_CORRELATION:
            return size
    return fits[-1] if fits else None


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else src.parent

    nbytes = src.stat().st_size
    if nbytes == 0 or nbytes % 2:
        sys.exit(f"{src}: {nbytes} bytes is not 16-bit frame data")
    data = np.memmap(src, dtype="<u2", mode="r")

    size, source = header_size(src), "header"
    if size is None:
        size, source = infer_size(data), "inferred"
        if size is None:
            known = ", ".join(f"{w} x {h}" for w, h in SIZES)
            sys.exit(f"{src}: {nbytes} bytes is not a whole number of 16-bit frames ({known})")
    w, h = size
    if nbytes % frame_bytes(size):
        sys.exit(f"{src}: {nbytes} bytes is not a whole number of {w} x {h} 16-bit frames (size from {source})")

    frames = data.reshape(-1, w, h).transpose(0, 2, 1)
    print(f"{src.name}: {len(frames)} frames of {w} x {h} ({source})")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        dst = out_dir / f"{src.stem}_f{i}.tiff"
        Image.fromarray(np.ascontiguousarray(frame)).save(dst)
        print(f"{dst}  min={frame.min()} max={frame.max()} mean={frame.mean():.1f}")


if __name__ == "__main__":
    main()
