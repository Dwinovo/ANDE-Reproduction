"""Algorithm 2 (paper Section III-B-2): per-SESSION 26-d statistical features.

Important reading of the paper:
    Algorithm 2's pseudocode says ``for *.pcap in folder do; computefeatures(*.pcap)``,
    but Table II describes every feature as "within a packet window" and the last
    feature is explicitly "Number of packets in *one session*". Combined with the
    fact that the paper reports 50,905 *samples* (sessions, not pcaps), the
    correct interpretation is that ``*.pcap`` in Algorithm 2 refers to the
    *per-session* pcap files emitted by Algorithm 1 line 5 ("save session to
    folders"), not to the raw capture files. We therefore compute one 26-d
    vector **per session** here, with session boundaries identical to those
    used by ``preprocess_raw.py``.

For each session we compute:
  TCP flag means (SYN, URG, FIN, ACK, PSH, RST)        ->  6
  Protocol counts (DNS, TCP, UDP, ICMP)                ->  4
  Window duration                                       ->  1
  Inter-packet delta times (avg/min/max/stdev)          ->  4
  Packet lengths      (avg/min/max/stdev)               ->  4
  Small-payload (<32B) packet ratio                     ->  1
  Payload sizes       (avg/min/max/stdev)               ->  4
  DNS / TCP ratio                                       ->  1
  Total packet count                                    ->  1
                                                       ----
                                                         26

Threshold: keep sessions with >= MIN_PACKETS=10 packets (Algorithm 2 line 13).
Algorithm 1 already drops sessions with < 3 packets; sessions with 3-9 packets
exist in the image manifest but not the stats manifest. The Dataset's
``load_joined_manifest`` joins on ``session_id`` so those few sessions are
dropped from the joined training set.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import PcapReader  # type: ignore[import-untyped]
from scapy.layers.dns import DNS  # type: ignore[import-untyped]
from scapy.layers.inet import ICMP, IP, TCP, UDP  # type: ignore[import-untyped]
from tqdm import tqdm

from ande.data.labels import label_from_path
from ande.data.preprocess_raw import _session_key

LOG = logging.getLogger(__name__)

MIN_PACKETS = 10
SMALL_PAYLOAD_THRESHOLD = 32

FEATURE_ORDER: tuple[str, ...] = (
    "Avg_syn_flag",
    "Avg_urg_flag",
    "Avg_fin_flag",
    "Avg_ack_flag",
    "Avg_psh_flag",
    "Avg_rst_flag",
    "Avg_DNS_pkt",
    "Avg_TCP_pkt",
    "Avg_UDP_pkt",
    "Avg_ICMP_pkt",
    "Duration_window_flow",
    "Avg_deltas_time",
    "Min_deltas_time",
    "Max_deltas_time",
    "StDev_deltas_time",
    "Avg_Pkts_length",
    "Min_Pkts_length",
    "Max_Pkts_length",
    "StDev_Pkts_length",
    "Avg_small_loadings_pkt",
    "Avg_payload",
    "Min_payload",
    "Max_payload",
    "StDev_payload",
    "Avg_DNS_over_TCP",
    "num_packets",
)


def _safe_stat(values: list[float], fn) -> float:
    return float(fn(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def _payload_size(pkt) -> int:
    if TCP in pkt:
        return len(bytes(pkt[TCP].payload))
    if UDP in pkt:
        return len(bytes(pkt[UDP].payload))
    return 0


def compute_features(packets: list) -> dict | None:
    """Compute 26-d features over the list of scapy packets belonging to ONE session."""
    n = len(packets)
    if n <= MIN_PACKETS:
        return None

    flags = {k: 0 for k in ("S", "U", "F", "A", "P", "R")}
    n_dns = n_tcp = n_udp = n_icmp = 0
    n_small = 0
    pkt_lens: list[int] = []
    payload_lens: list[int] = []
    timestamps: list[float] = []

    for pkt in packets:
        pkt_lens.append(len(pkt))
        timestamps.append(float(pkt.time))
        if TCP in pkt:
            n_tcp += 1
            f = int(pkt[TCP].flags)
            if f & 0x02:
                flags["S"] += 1
            if f & 0x20:
                flags["U"] += 1
            if f & 0x01:
                flags["F"] += 1
            if f & 0x10:
                flags["A"] += 1
            if f & 0x08:
                flags["P"] += 1
            if f & 0x04:
                flags["R"] += 1
        if UDP in pkt:
            n_udp += 1
        if ICMP in pkt:
            n_icmp += 1
        if DNS in pkt:
            n_dns += 1
        pl = _payload_size(pkt)
        payload_lens.append(pl)
        if pl < SMALL_PAYLOAD_THRESHOLD:
            n_small += 1

    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, n)] if n > 1 else []
    duration = (timestamps[-1] - timestamps[0]) if n > 1 else 0.0

    feats = {
        "Avg_syn_flag": flags["S"] / n,
        "Avg_urg_flag": flags["U"] / n,
        "Avg_fin_flag": flags["F"] / n,
        "Avg_ack_flag": flags["A"] / n,
        "Avg_psh_flag": flags["P"] / n,
        "Avg_rst_flag": flags["R"] / n,
        "Avg_DNS_pkt": n_dns / n,
        "Avg_TCP_pkt": n_tcp / n,
        "Avg_UDP_pkt": n_udp / n,
        "Avg_ICMP_pkt": n_icmp / n,
        "Duration_window_flow": duration,
        "Avg_deltas_time": _safe_stat(deltas, statistics.mean),
        "Min_deltas_time": _safe_stat(deltas, min),
        "Max_deltas_time": _safe_stat(deltas, max),
        "StDev_deltas_time": _stdev(deltas),
        "Avg_Pkts_length": _safe_stat(pkt_lens, statistics.mean),
        "Min_Pkts_length": _safe_stat(pkt_lens, min),
        "Max_Pkts_length": _safe_stat(pkt_lens, max),
        "StDev_Pkts_length": _stdev([float(x) for x in pkt_lens]),
        "Avg_small_loadings_pkt": n_small / n,
        "Avg_payload": _safe_stat(payload_lens, statistics.mean),
        "Min_payload": _safe_stat(payload_lens, min),
        "Max_payload": _safe_stat(payload_lens, max),
        "StDev_payload": _stdev([float(x) for x in payload_lens]),
        "Avg_DNS_over_TCP": (n_dns / n_tcp) if n_tcp else 0.0,
        "num_packets": float(n),
    }
    assert tuple(feats) == FEATURE_ORDER, "feature ordering drifted"
    return feats


def _process_pcap(pcap: Path) -> list[dict]:
    """Walk one raw pcap, split into sessions, compute 26-d features per session.

    Per-packet exception handling (matching preprocess_raw) so that one
    malformed packet does not discard the whole pcap.
    """
    label = label_from_path(pcap)
    if label is None:
        return []
    sessions: dict[tuple, list] = {}
    try:
        reader = PcapReader(str(pcap))
    except Exception as exc:
        LOG.warning("could not open %s: %s", pcap, exc)
        return []
    try:
        for i, pkt in enumerate(reader):
            try:
                key = _session_key(pkt)
                if key is None:
                    continue
                sessions.setdefault(key, []).append(pkt)
            except Exception as exc:  # pragma: no cover - rare malformed packets
                LOG.debug("skip pkt %d in %s: %s", i, pcap.name, exc)
    finally:
        reader.close()
    rows: list[dict] = []
    for key, packets in sessions.items():
        feats = compute_features(packets)
        if feats is None:
            continue
        sid = f"{pcap.stem}__{key[0]}_{key[2]}_{key[1]}_{key[3]}_{key[4]}"
        rows.append({
            "session_id": sid,
            "pcap_src": str(pcap),
            **feats,
        })
    return rows


def to_array(rows: list[dict]) -> np.ndarray:
    return np.array([[r[k] for k in FEATURE_ORDER] for r in rows], dtype=np.float64)


def zscore_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma


def zscore_apply(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (arr - mu) / sigma


def preprocess(raw_root: Path, out_root: Path, workers: int = 1) -> Path:
    pcaps = sorted(p for p in raw_root.rglob("*") if p.suffix.lower() in {".pcap", ".pcapng"})
    if not pcaps:
        raise FileNotFoundError(f"no pcap/pcapng under {raw_root}")

    rows: list[dict] = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for chunk in tqdm(pool.imap_unordered(_process_pcap, pcaps), total=len(pcaps)):
                rows.extend(chunk)
    else:
        for pcap in tqdm(pcaps):
            rows.extend(_process_pcap(pcap))

    if not rows:
        raise RuntimeError(f"no usable sessions under {raw_root}")

    feat_only = [{k: r[k] for k in FEATURE_ORDER} for r in rows]
    arr = to_array(feat_only)
    mu, sigma = zscore_fit(arr)
    arr_norm = zscore_apply(arr, mu, sigma)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats").mkdir(exist_ok=True)
    (out_root / "stats" / "stats_moments.json").write_text(
        json.dumps({"mu": mu.tolist(), "sigma": sigma.tolist()})
    )

    df = pd.DataFrame(arr_norm, columns=list(FEATURE_ORDER))
    df.insert(0, "session_id", [r["session_id"] for r in rows])
    df.insert(1, "pcap_src", [r["pcap_src"] for r in rows])
    manifest = out_root / "manifest_stats.parquet"
    df.to_parquet(manifest, index=False)
    LOG.info("wrote %d session-level feature vectors -> %s", len(df), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    preprocess(args.raw_root, args.out_root, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
