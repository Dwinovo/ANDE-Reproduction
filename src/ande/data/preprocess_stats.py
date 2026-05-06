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

Implementation note (streaming):
    Pcaps in the dataset can hold tens of millions of packets in a single
    session (e.g. a 2.4 GB FILE-TRANSFER capture is essentially one TCP flow).
    Buffering all scapy ``Packet`` objects per session blows up memory and
    spends most of its CPU time copying payload bytes. We instead use a
    ``SessionAcc`` accumulator that updates running counters and Welford's
    online variance per packet, so memory is O(num_sessions) and we never
    materialise per-packet payload bytes.

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

Threshold: keep sessions with > MIN_PACKETS=10 packets (Algorithm 2 line 13).
Algorithm 1 already drops sessions with < 3 packets; sessions with 3-10 packets
exist in the image manifest but not the stats manifest. The Dataset's
``load_joined_manifest`` joins on ``session_id`` so those few sessions are
dropped from the joined training set.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import PcapReader  # type: ignore[import-untyped]
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


class SessionAcc:
    """Streaming accumulator for one session.

    Memory is O(1) regardless of packet count; Welford's online algorithm is
    used for running variance so we never store per-packet values.
    """

    __slots__ = (
        "n", "n_tcp", "n_udp", "n_icmp", "n_dns", "n_small",
        "syn", "urg", "fin", "ack", "psh", "rst",
        "first_ts", "last_ts",
        "d_n", "d_mean", "d_M2", "d_min", "d_max",
        "pl_n", "pl_mean", "pl_M2", "pl_min", "pl_max",
        "py_n", "py_mean", "py_M2", "py_min", "py_max",
    )

    def __init__(self) -> None:
        self.n = 0
        self.n_tcp = self.n_udp = self.n_icmp = self.n_dns = self.n_small = 0
        self.syn = self.urg = self.fin = self.ack = self.psh = self.rst = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        # delta-times (n-1 inter-packet intervals)
        self.d_n = 0
        self.d_mean = 0.0
        self.d_M2 = 0.0
        self.d_min = math.inf
        self.d_max = -math.inf
        # packet lengths
        self.pl_n = 0
        self.pl_mean = 0.0
        self.pl_M2 = 0.0
        self.pl_min = math.inf
        self.pl_max = -math.inf
        # payload sizes
        self.py_n = 0
        self.py_mean = 0.0
        self.py_M2 = 0.0
        self.py_min = math.inf
        self.py_max = -math.inf

    @staticmethod
    def _payload_size_fast(pkt) -> int:
        """O(1) payload size via header arithmetic - avoids materialising bytes."""
        if TCP in pkt:
            ip = pkt[IP] if IP in pkt else None
            if ip is None:
                return 0
            tcp = pkt[TCP]
            ip_payload = int(ip.len) - (int(ip.ihl) * 4)
            return max(0, ip_payload - (int(tcp.dataofs) * 4))
        if UDP in pkt:
            return max(0, int(pkt[UDP].len) - 8)
        return 0

    def update(self, pkt) -> None:
        self.n += 1

        # timestamp + delta
        ts = float(pkt.time)
        if self.first_ts is None:
            self.first_ts = ts
        else:
            d = ts - (self.last_ts or ts)
            self.d_n += 1
            delta = d - self.d_mean
            self.d_mean += delta / self.d_n
            self.d_M2 += delta * (d - self.d_mean)
            if d < self.d_min:
                self.d_min = d
            if d > self.d_max:
                self.d_max = d
        self.last_ts = ts

        # packet length
        plen = len(pkt)
        self.pl_n += 1
        delta = plen - self.pl_mean
        self.pl_mean += delta / self.pl_n
        self.pl_M2 += delta * (plen - self.pl_mean)
        if plen < self.pl_min:
            self.pl_min = plen
        if plen > self.pl_max:
            self.pl_max = plen

        # protocol counts + flags
        if TCP in pkt:
            self.n_tcp += 1
            f = int(pkt[TCP].flags)
            if f & 0x02:
                self.syn += 1
            if f & 0x20:
                self.urg += 1
            if f & 0x01:
                self.fin += 1
            if f & 0x10:
                self.ack += 1
            if f & 0x08:
                self.psh += 1
            if f & 0x04:
                self.rst += 1
        if UDP in pkt:
            self.n_udp += 1
            # DNS heuristic: UDP and either port == 53 (avoids slow DNS-layer parse)
            try:
                udp = pkt[UDP]
                if int(udp.sport) == 53 or int(udp.dport) == 53:
                    self.n_dns += 1
            except Exception:  # pragma: no cover
                pass
        if ICMP in pkt:
            self.n_icmp += 1

        # payload size (O(1) header math)
        py = self._payload_size_fast(pkt)
        self.py_n += 1
        delta = py - self.py_mean
        self.py_mean += delta / self.py_n
        self.py_M2 += delta * (py - self.py_mean)
        if py < self.py_min:
            self.py_min = py
        if py > self.py_max:
            self.py_max = py
        if py < SMALL_PAYLOAD_THRESHOLD:
            self.n_small += 1

    def to_dict(self) -> dict | None:
        if self.n <= MIN_PACKETS:
            return None
        n = self.n
        pl_stdev = math.sqrt(self.pl_M2 / self.pl_n) if self.pl_n > 1 else 0.0
        py_stdev = math.sqrt(self.py_M2 / self.py_n) if self.py_n > 1 else 0.0
        d_stdev = math.sqrt(self.d_M2 / self.d_n) if self.d_n > 1 else 0.0
        duration = (self.last_ts - self.first_ts) if self.first_ts is not None else 0.0
        d_min = float(self.d_min) if self.d_n > 0 else 0.0
        d_max = float(self.d_max) if self.d_n > 0 else 0.0
        d_mean = float(self.d_mean) if self.d_n > 0 else 0.0
        feats = {
            "Avg_syn_flag": self.syn / n,
            "Avg_urg_flag": self.urg / n,
            "Avg_fin_flag": self.fin / n,
            "Avg_ack_flag": self.ack / n,
            "Avg_psh_flag": self.psh / n,
            "Avg_rst_flag": self.rst / n,
            "Avg_DNS_pkt": self.n_dns / n,
            "Avg_TCP_pkt": self.n_tcp / n,
            "Avg_UDP_pkt": self.n_udp / n,
            "Avg_ICMP_pkt": self.n_icmp / n,
            "Duration_window_flow": duration,
            "Avg_deltas_time": d_mean,
            "Min_deltas_time": d_min,
            "Max_deltas_time": d_max,
            "StDev_deltas_time": d_stdev,
            "Avg_Pkts_length": float(self.pl_mean),
            "Min_Pkts_length": float(self.pl_min) if self.pl_n > 0 else 0.0,
            "Max_Pkts_length": float(self.pl_max) if self.pl_n > 0 else 0.0,
            "StDev_Pkts_length": pl_stdev,
            "Avg_small_loadings_pkt": self.n_small / n,
            "Avg_payload": float(self.py_mean),
            "Min_payload": float(self.py_min) if self.py_n > 0 else 0.0,
            "Max_payload": float(self.py_max) if self.py_n > 0 else 0.0,
            "StDev_payload": py_stdev,
            "Avg_DNS_over_TCP": (self.n_dns / self.n_tcp) if self.n_tcp else 0.0,
            "num_packets": float(n),
        }
        assert tuple(feats) == FEATURE_ORDER, "feature ordering drifted"
        return feats


def _process_pcap(pcap: Path) -> list[dict]:
    """Walk one raw pcap, split into sessions, accumulate features per session."""
    label = label_from_path(pcap)
    if label is None:
        return []
    sessions: dict[tuple, SessionAcc] = {}
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
                acc = sessions.get(key)
                if acc is None:
                    acc = SessionAcc()
                    sessions[key] = acc
                acc.update(pkt)
            except Exception as exc:  # pragma: no cover
                LOG.debug("skip pkt %d in %s: %s", i, pcap.name, exc)
    finally:
        reader.close()
    rows: list[dict] = []
    for key, acc in sessions.items():
        feats = acc.to_dict()
        if feats is None:
            continue
        sid = f"{pcap.stem}__{key[0]}_{key[2]}_{key[1]}_{key[3]}_{key[4]}"
        rows.append({"session_id": sid, "pcap_src": str(pcap), **feats})
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


def compute_features(packets: list) -> dict | None:
    """Backwards-compatible entry point used by tests: feed a list of scapy
    packets through a fresh accumulator and return the 26-d dict.
    """
    acc = SessionAcc()
    for pkt in packets:
        acc.update(pkt)
    return acc.to_dict()


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
