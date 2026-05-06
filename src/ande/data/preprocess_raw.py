"""Algorithm 1 (paper Section III-B-1): pcap -> session bytes -> grayscale image.

Pipeline per pcap file:
  1. Stream-parse packets with scapy.
  2. Group into bidirectional sessions keyed by an unordered 5-tuple
     (ip_a, ip_b, port_a, port_b, protocol).
  3. Drop sessions with fewer than 3 packets.
  4. Anonymise each packet by zero-ing MAC, IP and L4-port fields.
  5. Concatenate the raw bytes of every packet in arrival order, truncate or
     zero-pad to ``size`` bytes (784 / 4096 / 8100), and reshape to a square
     uint8 image.

The resulting tensors are written to ``out_dir/images/<size>/<label>/<sid>.npy``
together with a parquet manifest used by the dataset loader.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import Ether, PcapReader, raw  # type: ignore[import-untyped]
from scapy.layers.inet import IP, TCP, UDP  # type: ignore[import-untyped]
from tqdm import tqdm

from ande.data.labels import Label, label_from_path

LOG = logging.getLogger(__name__)

VALID_SIZES: tuple[int, ...] = (784, 4096, 8100)
SIZE_TO_HW: dict[int, tuple[int, int]] = {784: (28, 28), 4096: (64, 64), 8100: (90, 90)}

MIN_PACKETS_PER_SESSION = 3
SessionKey = tuple[str, str, int, int, int]


@dataclass
class SessionRecord:
    sid: str
    pcap_src: str
    label: Label
    packets: list[bytes]

    def num_packets(self) -> int:
        return len(self.packets)


def _session_key(pkt) -> SessionKey | None:
    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = int(ip.proto)
    sport = dport = 0
    if TCP in pkt:
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif UDP in pkt:
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
    a, b = sorted([(ip.src, sport), (ip.dst, dport)])
    return (a[0], b[0], a[1], b[1], proto)


def _anonymize(pkt) -> bytes:
    """Zero MAC, IP and L4-port fields, then return raw bytes."""
    if Ether in pkt:
        pkt[Ether].src = "00:00:00:00:00:00"
        pkt[Ether].dst = "00:00:00:00:00:00"
    if IP in pkt:
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"
    if TCP in pkt:
        pkt[TCP].sport = 0
        pkt[TCP].dport = 0
    elif UDP in pkt:
        pkt[UDP].sport = 0
        pkt[UDP].dport = 0
    return bytes(raw(pkt))


def _iter_sessions(pcap: Path, label: Label) -> Iterator[SessionRecord]:
    sessions: dict[SessionKey, list[bytes]] = {}
    try:
        with PcapReader(str(pcap)) as reader:
            for i, pkt in enumerate(reader):
                key = _session_key(pkt)
                if key is None:
                    continue
                try:
                    sessions.setdefault(key, []).append(_anonymize(pkt))
                except Exception as exc:  # pragma: no cover - rare malformed packets
                    LOG.debug("skip pkt %d in %s: %s", i, pcap.name, exc)
    except Exception as exc:
        LOG.warning("scapy failed on %s: %s", pcap, exc)
        return
    for key, packets in sessions.items():
        if len(packets) < MIN_PACKETS_PER_SESSION:
            continue
        sid = f"{pcap.stem}__{key[0]}_{key[2]}_{key[1]}_{key[3]}_{key[4]}"
        yield SessionRecord(sid=sid, pcap_src=str(pcap), label=label, packets=packets)


def session_to_image(packets: list[bytes], size: int) -> np.ndarray:
    """Concatenate packet bytes, truncate or zero-pad to ``size``, reshape to HxW."""
    if size not in SIZE_TO_HW:
        raise ValueError(f"size must be one of {VALID_SIZES}, got {size}")
    h, w = SIZE_TO_HW[size]
    flat = b"".join(packets)
    if len(flat) >= size:
        flat = flat[:size]
    else:
        flat = flat + b"\x00" * (size - len(flat))
    arr = np.frombuffer(flat, dtype=np.uint8).copy()
    return arr.reshape(h, w)


def _process_pcap(args: tuple[Path, Path, tuple[int, ...]]) -> list[dict]:
    pcap, out_root, sizes = args
    label = label_from_path(pcap)
    if label is None:
        LOG.info("no label match, skipping %s", pcap.name)
        return []
    rows: list[dict] = []
    for rec in _iter_sessions(pcap, label):
        row: dict = {
            "session_id": rec.sid,
            "pcap_src": rec.pcap_src,
            "activity": label.activity,
            "is_tor": label.is_tor,
            "label_2cls": label.binary_id,
            "label_14cls": label.behavior_id,
            "num_packets": rec.num_packets(),
        }
        for size in sizes:
            img = session_to_image(rec.packets, size)
            target = out_root / "images" / str(size) / label.behavior_name / f"{rec.sid}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, img)
            row[f"image_{size}"] = str(target.relative_to(out_root))
        rows.append(row)
    return rows


def preprocess(
    raw_root: Path,
    out_root: Path,
    sizes: tuple[int, ...] = VALID_SIZES,
    workers: int = 1,
) -> Path:
    """Process every pcap under ``raw_root`` and write a manifest parquet."""
    pcaps = sorted(p for p in raw_root.rglob("*") if p.suffix.lower() in {".pcap", ".pcapng"})
    if not pcaps:
        raise FileNotFoundError(f"no pcap/pcapng under {raw_root}")
    jobs = [(p, out_root, sizes) for p in pcaps]

    rows: list[dict] = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for chunk in tqdm(pool.imap_unordered(_process_pcap, jobs), total=len(jobs)):
                rows.extend(chunk)
    else:
        for job in tqdm(jobs):
            rows.extend(_process_pcap(job))

    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest_raw.parquet"
    df.to_parquet(manifest, index=False)
    LOG.info("wrote %d sessions -> %s", len(df), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(VALID_SIZES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    invalid = [s for s in args.sizes if s not in VALID_SIZES]
    if invalid:
        parser.error(f"invalid sizes {invalid}; allowed: {VALID_SIZES}")
    preprocess(args.raw_root, args.out_root, tuple(args.sizes), workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
