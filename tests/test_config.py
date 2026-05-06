from pathlib import Path

import pytest

from ande.utils.config import load_config


@pytest.fixture
def configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def test_load_main_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "ande_8100_14cls.yaml")
    assert cfg.data.size == 8100
    assert cfg.data.task == "behavior14"
    assert cfg.model.use_se is True


def test_no_se_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "ande_8100_14cls_no_se.yaml")
    assert cfg.model.use_se is False
