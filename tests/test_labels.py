from pathlib import Path

from ande.data.labels import (
    ACTIVITIES,
    Label,
    activity_from_filename,
    label_from_path,
)


def test_seven_activities_present():
    assert set(ACTIVITIES) == {"browsing", "chat", "email", "ft", "p2p", "streaming", "voip"}


def test_keyword_matching():
    assert activity_from_filename("BROWSING_gateway.pcap") == "browsing"
    assert activity_from_filename("CHAT_aim.pcap") == "chat"
    assert activity_from_filename("MAIL_thunderbird.pcap") == "email"
    assert activity_from_filename("FTP_filezilla.pcap") == "ft"
    assert activity_from_filename("P2P_vuze.pcap") == "p2p"
    assert activity_from_filename("YOUTUBE_traffic.pcap") == "streaming"
    assert activity_from_filename("VOIPBUSTER_call.pcap") == "voip"


def test_voip_prefix_wins_over_audio_substring():
    # In ISCXTor2016 voice-call captures always start with VOIP_ even when
    # the rest of the name contains "audio" or "voice".
    assert activity_from_filename("VOIP_gate_facebook_Audio.pcap") == "voip"
    assert activity_from_filename("VOIP_Skype_Voice_Gateway.pcap") == "voip"


def test_audio_prefix_is_streaming():
    # ISCXTor2016's AUDIO_ prefix is always Spotify (music streaming),
    # not voice traffic.
    assert activity_from_filename("AUDIO_spotifygateway.pcap") == "streaming"
    assert activity_from_filename("AUDIO_tor_spotify.pcap") == "streaming"


def test_unknown_returns_none():
    assert activity_from_filename("random_capture.pcap") is None


def test_label_ids_consistent():
    label = Label(activity="browsing", is_tor=True)
    assert label.binary_id == 1
    assert label.behavior_id == 1  # browsing index 0 * 2 + 1


def test_label_from_path_iscxtor_layout():
    # Both legacy (Pcaps/) and the actual zip top-level (Tor/) work.
    for sub in ("Pcaps", "Tor"):
        p = Path(f"data/raw/iscxtor2016/{sub}/BROWSING_gateway.pcap")
        label = label_from_path(p)
        assert label is not None
        assert label.activity == "browsing"
        assert label.is_tor is True


def test_label_from_path_nontor_layout():
    p = Path("data/raw/iscxtor2016/NonTor-PCAPs/CHAT_facebook.pcap")
    label = label_from_path(p)
    assert label is not None
    assert label.activity == "chat"
    assert label.is_tor is False
