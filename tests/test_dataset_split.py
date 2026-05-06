"""Tests for the pcap-level vs session-level split logic.

The pcap-level split is the default because session-level causes the
26-d statistical features (computed per pcap) to be duplicated across
train and test, allowing tree models to memorise pcap -> label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ande.data.dataset import VALID_SPLIT_GRANULARITY, stratified_split


def _make_synthetic_manifest(
    n_pcaps_per_class: dict[int, int],
    sessions_per_pcap: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a fake manifest where every pcap has multiple sessions and a
    single label. Used to verify split semantics without touching disk.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    pcap_idx = 0
    for label, n in n_pcaps_per_class.items():
        for _ in range(n):
            pcap_src = f"pcap_{pcap_idx}_{label}.pcap"
            pcap_idx += 1
            for sess in range(sessions_per_pcap):
                rows.append({
                    "session_id": f"{pcap_src}__sess{sess}",
                    "pcap_src": pcap_src,
                    "label_14cls": label,
                    "feature_x": rng.random(),  # surrogate for stats
                })
    return pd.DataFrame(rows)


def test_split_at_constants():
    assert "pcap" in VALID_SPLIT_GRANULARITY
    assert "session" in VALID_SPLIT_GRANULARITY


def test_invalid_split_at_raises():
    df = _make_synthetic_manifest({0: 4, 1: 4})
    with pytest.raises(ValueError):
        stratified_split(df, "label_14cls", 0.8, seed=0, split_at="bogus")


def test_pcap_level_split_no_pcap_overlap():
    df = _make_synthetic_manifest({0: 8, 1: 8, 2: 8})
    s = stratified_split(df, "label_14cls", train_ratio=0.75, seed=0, split_at="pcap")
    train_pcaps = set(s.train["pcap_src"])
    test_pcaps = set(s.test["pcap_src"])
    assert train_pcaps & test_pcaps == set(), "pcaps must not appear in both splits"


def test_pcap_level_split_keeps_all_classes_in_test():
    # 14 classes, each with 4 pcaps -> 80% train (3 pcaps) / 20% test (1 pcap)
    df = _make_synthetic_manifest({c: 4 for c in range(14)})
    s = stratified_split(df, "label_14cls", train_ratio=0.75, seed=0, split_at="pcap")
    test_classes = set(s.test["label_14cls"].unique())
    assert test_classes == set(range(14)), "every class must appear in test"


def test_pcap_level_drops_singleton_classes():
    # class 0 has 1 pcap (cannot be split) -> dropped silently with the rest preserved
    df = _make_synthetic_manifest({0: 1, 1: 6, 2: 6})
    s = stratified_split(df, "label_14cls", train_ratio=0.5, seed=0, split_at="pcap")
    assert 0 not in set(s.train["label_14cls"])
    assert 0 not in set(s.test["label_14cls"])
    assert {1, 2}.issubset(set(s.train["label_14cls"]))
    assert {1, 2}.issubset(set(s.test["label_14cls"]))


def test_pcap_level_session_count_proportional():
    # Each pcap has 5 sessions; with pcap-level 80/20 split we expect
    # roughly 80/20 sessions too (slight slack because we split pcaps not sessions).
    df = _make_synthetic_manifest({c: 10 for c in range(14)}, sessions_per_pcap=5)
    s = stratified_split(df, "label_14cls", train_ratio=0.8, seed=42, split_at="pcap")
    total = len(s.train) + len(s.test)
    train_frac = len(s.train) / total
    assert 0.7 < train_frac < 0.9, train_frac


def test_session_level_split_ignores_pcap_boundary():
    # Sanity check the legacy mode still works.
    df = _make_synthetic_manifest({c: 4 for c in range(14)}, sessions_per_pcap=5)
    s = stratified_split(df, "label_14cls", 0.8, seed=0, split_at="session")
    # Same pcap can appear in both splits (that's the whole reason pcap-level exists)
    train_pcaps = set(s.train["pcap_src"])
    test_pcaps = set(s.test["pcap_src"])
    assert train_pcaps & test_pcaps  # overlap is expected here


def test_pcap_level_split_is_deterministic_per_seed():
    df = _make_synthetic_manifest({c: 6 for c in range(14)}, sessions_per_pcap=4)
    s1 = stratified_split(df, "label_14cls", 0.75, seed=42, split_at="pcap")
    s2 = stratified_split(df, "label_14cls", 0.75, seed=42, split_at="pcap")
    assert set(s1.train["pcap_src"]) == set(s2.train["pcap_src"])
    assert set(s1.test["pcap_src"]) == set(s2.test["pcap_src"])
