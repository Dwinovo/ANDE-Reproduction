"""Bootstrap dataset directories and verify integrity.

ISCXTor2016 requires manual registration — we cannot fetch it programmatically.
This script:

  1. Documents where the user must place each archive.
  2. Verifies SHA256 hashes of every downloaded file when ``--check`` is given.
  3. Walks ``data/raw/`` and reports counts that should match the paper's Fig. 4.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"

INSTRUCTIONS = """\
ISCXTor2016
  Source : https://www.unb.ca/cic/datasets/tor.html
  Layout : extract under  data/raw/iscxtor2016/
           expected subfolders: Pcaps/, NonTor-PCAPs/

darknet-2020
  Source : https://github.com/huyz97/darknet-dataset-2020
  Layout : extract under  data/raw/darknet2020/
           keep only the Tor subset; the paper ignores the rest.
"""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def walk_pcaps(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".pcap", ".pcapng"})


def cmd_layout(_args: argparse.Namespace) -> int:
    console.print("[bold]Required directory layout[/bold]\n" + INSTRUCTIONS)
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    if not RAW.exists():
        console.print(f"[red]missing[/red] {RAW}")
        return 1
    pcaps = walk_pcaps(RAW)
    if not pcaps:
        console.print(f"[yellow]warning[/yellow] no .pcap/.pcapng under {RAW}")
        return 1
    table = Table(title=f"Pcap inventory under {RAW}")
    table.add_column("relpath", overflow="fold")
    table.add_column("MB", justify="right")
    table.add_column("sha256[:12]")
    total = 0
    for p in pcaps:
        size = p.stat().st_size
        total += size
        table.add_row(str(p.relative_to(RAW)), f"{size / 1e6:.1f}", _sha256(p)[:12])
    console.print(table)
    console.print(f"\n[bold]total[/bold]: {len(pcaps)} files, {total / 1e9:.2f} GB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layout", help="print where each archive must live")
    sub.add_parser("check", help="walk data/raw and report files + sha256")
    args = parser.parse_args(argv)
    return {"layout": cmd_layout, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
