import torch

from ande.attacks import AttackSpec, perturb_image, perturb_stat
from ande.data.preprocess_stats import FEATURE_ORDER


def test_padding_attack_shifts_bytes() -> None:
    image = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    out = perturb_image(image, AttackSpec("zero_padding", "medium"))
    assert out.shape == image.shape
    assert torch.count_nonzero(out.flatten()[:2]) == 0
    assert not torch.equal(out, image)


def test_delay_attack_changes_timing_stats_only() -> None:
    stat = torch.zeros(len(FEATURE_ORDER))
    out = perturb_stat(stat, AttackSpec("random_delay", "high"))
    changed = torch.nonzero(out).flatten().tolist()
    changed_names = {FEATURE_ORDER[i] for i in changed}
    assert "Duration_window_flow" in changed_names
    assert "Avg_deltas_time" in changed_names
    assert "Avg_payload" not in changed_names


def test_traffic_shaping_shrinks_size_features() -> None:
    stat = torch.ones(len(FEATURE_ORDER))
    out = perturb_stat(stat, AttackSpec("traffic_shaping", "medium"))
    assert out[FEATURE_ORDER.index("Avg_payload")] < stat[FEATURE_ORDER.index("Avg_payload")]
    assert out[FEATURE_ORDER.index("Avg_TCP_pkt")] == stat[FEATURE_ORDER.index("Avg_TCP_pkt")]
