"""Killable worker that writes one unpublished code-interpreter temp file."""

from __future__ import annotations

import os
import sys

_COPY_CHUNK_BYTES = 64 * 1024


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(2)

    stage_fd = int(sys.argv[1])
    with os.fdopen(stage_fd, "wb") as output:
        while chunk := sys.stdin.buffer.read(_COPY_CHUNK_BYTES):
            output.write(chunk)


if __name__ == "__main__":
    main()
