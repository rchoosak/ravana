"""Killable worker that snapshots a non-git project without following symlinks."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(2)

    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2])

    def ignore_root(path: str, names: list[str]) -> set[str]:
        return {".ravana"} if Path(path).resolve() == source else set()

    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=ignore_root,
    )


if __name__ == "__main__":
    main()
