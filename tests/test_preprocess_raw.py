"""Unit tests for the raw-byte image conversion. Pcap parsing is not exercised
here (it requires real captures); these tests focus on the deterministic
byte-to-image transformation.
"""

import numpy as np

from ande.data.preprocess_raw import VALID_SIZES, session_to_image


def test_session_to_image_shapes():
    for size in VALID_SIZES:
        packets = [b"\x00\x01\x02\x03"] * 5
        img = session_to_image(packets, size=size)
        assert img.dtype == np.uint8
        assert img.size == size


def test_padding_with_zero():
    img = session_to_image([b"\xff" * 10], size=784)
    # first 10 bytes should be 0xff; the rest must be 0 padding
    assert (img.flatten()[:10] == 0xFF).all()
    assert (img.flatten()[10:] == 0x00).all()


def test_truncation_when_oversized():
    big = [b"\xab" * 10000]
    img = session_to_image(big, size=784)
    assert img.size == 784
    assert (img == 0xAB).all()
