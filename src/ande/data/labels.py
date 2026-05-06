"""Label mapping for ISCXTor2016 + darknet-2020.

ISCXTor2016 pcap filenames carry the activity type. The paper groups the raw
filenames into 7 activity classes (Section III-A): browsing, chat, email, file
transfer (FT), peer-to-peer (P2P), streaming, VoIP. Combined with the binary
{Tor, non-Tor} dimension this yields the 14 user-behavior classes used in the
14-class experiment.

The mapping below is built from the official ISCXTor2016 archive layout
(``Pcaps`` and ``NonTor-PCAPs`` folders) and the keyword list provided by
Lashkari et al. [19] in the dataset documentation. It is implemented as a
keyword match on lower-cased filenames so that minor naming variations across
the two source archives are handled uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ACTIVITIES: tuple[str, ...] = (
    "browsing",
    "chat",
    "email",
    "ft",
    "p2p",
    "streaming",
    "voip",
)

# Filename keyword -> canonical activity. Order matters: the first match wins.
#
# ISCXTor2016 uses a categorical UPPERCASE prefix on every "main" pcap
# (``AUDIO_*`` / ``VIDEO_*`` / ``VOIP_*`` / ``CHAT_*`` / ``MAIL_*`` /
# ``BROWSING_*`` / ``P2P_*`` / ``FILE-TRANSFER_*``), plus a handful of
# lowercase ``tor<Site>.pcap`` and ``tor_<thing>.pcap`` captures.
#
# Strategy: list explicit lowercase exceptions first, then prefix-anchored
# rules (e.g. ``voip_``), then loose substring keywords for darknet-2020 and
# other variations.
_KEYWORD_TO_ACTIVITY: tuple[tuple[str, str], ...] = (
    # ----- Lowercase tor<Site>.pcap exceptions (handled before generic rules)
    ("torfacebook", "browsing"),  # plain Facebook web browsing through Tor
    ("torgoogle", "browsing"),
    ("tortwitter", "browsing"),
    # tor_p2p_*.pcap -> matched later via "p2p"
    # tor_spotify*.pcap -> matched later via "spotify"
    # torVimeo* / torYoutube* -> matched later via "vimeo" / "youtube"
    # ----- VoIP must come before any "audio" / "skype" rule.
    ("voipbuster", "voip"),
    ("voip_", "voip"),
    ("voip", "voip"),
    ("hangout_voice", "voip"),
    ("hangouts_voice", "voip"),
    ("skype_voice", "voip"),
    # NonTor side has voice-call captures without VOIP_ prefix:
    # Skype_Audio.pcap, Facebook_Voice_Workstation.pcap, facebook_Audio.pcap.
    # Match the suffix patterns *_voice and *_audio. The streaming AUDIO_*
    # files have the underscore on the OTHER side (audio_*) so they're safe.
    ("_voice", "voip"),
    ("_audio", "voip"),
    # Streaming (incl. AUDIO_ prefix which in this dataset is always Spotify)
    ("audio_", "streaming"),
    ("video_", "streaming"),
    ("streaming", "streaming"),
    ("spotify", "streaming"),
    ("vimeo", "streaming"),
    ("youtube", "streaming"),
    ("netflix", "streaming"),
    # Email — must come before "file-transfer"/"filetransfer" because
    # ISCXTor2016 has files like MAIL_gate_POP_filetransfer.pcap.
    ("mail_", "email"),
    ("email", "email"),
    ("mail", "email"),
    ("smtp", "email"),
    ("pop_", "email"),
    ("imap", "email"),
    ("thunderbird", "email"),
    # File transfer
    ("file-transfer", "ft"),
    ("filetransfer", "ft"),
    ("file_transfer", "ft"),
    ("sftp", "ft"),
    ("ftp", "ft"),
    ("scp", "ft"),
    # P2P
    ("p2p", "p2p"),
    ("vuze", "p2p"),
    ("bittorrent", "p2p"),
    ("utorrent", "p2p"),
    ("torrent", "p2p"),
    # Chat
    ("chat", "chat"),
    ("icq", "chat"),
    ("aim", "chat"),
    ("messenger", "chat"),
    ("hangout", "chat"),
    ("skype", "chat"),
    ("facebook", "chat"),
    # Browsing - last because many categories tunnel through browsers.
    ("browsing", "browsing"),
    ("browser", "browsing"),
    ("ssl_browsing", "browsing"),
    ("ssl", "browsing"),  # NonTor has a bare ssl.pcap (HTTPS browsing)
    ("http", "browsing"),
    ("https", "browsing"),
)


@dataclass(frozen=True)
class Label:
    """Composite label for a session."""

    activity: str  # one of ACTIVITIES
    is_tor: bool  # True for Tor / dark-web traffic, False for normal

    @property
    def binary_id(self) -> int:
        """0 = normal, 1 = Tor."""
        return 1 if self.is_tor else 0

    @property
    def behavior_id(self) -> int:
        """14-class id: activity index * 2 + is_tor.

        Order: browsing-NonTor=0, browsing-Tor=1, chat-NonTor=2, ... voip-Tor=13.
        """
        return ACTIVITIES.index(self.activity) * 2 + int(self.is_tor)

    @property
    def behavior_name(self) -> str:
        return f"{self.activity}-{'tor' if self.is_tor else 'nontor'}"


def activity_from_filename(name: str) -> str | None:
    """Best-effort match of pcap filename to one of ACTIVITIES.

    Returns None when nothing in the keyword table matches; callers should
    decide whether to skip such files or assign a fallback label.
    """
    lowered = name.lower()
    for keyword, activity in _KEYWORD_TO_ACTIVITY:
        if keyword in lowered:
            return activity
    return None


def is_tor_from_path(path: Path) -> bool:
    """Heuristic: ISCXTor2016 separates Tor and non-Tor at the top folder level
    (``Pcaps/`` vs ``NonTor-PCAPs/``); darknet-2020 has a ``tor`` subfolder.
    """
    parts = {p.lower() for p in path.parts}
    if "nontor-pcaps" in parts or "nontor" in parts:
        return False
    if "pcaps" in parts and "tor" not in parts:
        # ISCXTor2016 ``Pcaps`` folder is the Tor side
        return "nontor-pcaps" not in parts
    return "tor" in parts


def label_from_path(path: Path) -> Label | None:
    activity = activity_from_filename(path.name)
    if activity is None:
        return None
    return Label(activity=activity, is_tor=is_tor_from_path(path))
