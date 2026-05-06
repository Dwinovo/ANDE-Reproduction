"""Parallel extractor for darknet-datasets.zip — Tor subset only.

The ANDE paper uses only the ``tor/`` portion of darknet-2020. We extract
those 60 pcaps with a multiprocessing pool because zip entries are
independently compressed (unlike .xz which is one stream).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import zipfile
from pathlib import Path


def _extract_one(args: tuple[str, str, str]) -> tuple[str, int]:
    src, dst, name = args
    with zipfile.ZipFile(src) as z:
        info = z.getinfo(name)
        z.extract(info, dst)
    return name, info.file_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"C:\Users\dwin\Downloads\darknet-datasets.zip")
    parser.add_argument("--dst", default="data/raw/darknet2020")
    parser.add_argument("--prefix", default="tor/", help="only extract entries starting with this")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    Path(args.dst).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.src) as z:
        members = [m for m in z.infolist() if m.filename.startswith(args.prefix) and not m.is_dir()]
    total = sum(m.file_size for m in members)
    print(f"will extract {len(members)} files, {total / 1e9:.2f} GB uncompressed")

    jobs = [(args.src, args.dst, m.filename) for m in members]
    t0 = time.time()
    done_size = 0
    done_n = 0
    with mp.Pool(args.workers) as pool:
        for name, size in pool.imap_unordered(_extract_one, jobs):
            done_n += 1
            done_size += size
            elapsed = time.time() - t0
            rate = done_size / 1e6 / max(elapsed, 1e-9)
            print(
                f"[{done_n:3d}/{len(members)}] "
                f"{done_size / total * 100:5.1f}%  "
                f"{done_size / 1e9:5.2f}/{total / 1e9:.2f} GB  "
                f"{rate:.1f} MB/s  {name}",
                flush=True,
            )
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
