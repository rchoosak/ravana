"""Killable workspace quota scanner used by the sandbox boundary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ravana.runtime.toolkits.sandbox import _workspace_violation


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("max_bytes", type=int)
    parser.add_argument("max_files", type=int)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args()

    while True:
        violation = _workspace_violation(
            args.workspace,
            max_bytes=args.max_bytes,
            max_files=args.max_files,
        )
        if violation is not None or not args.watch:
            payload = (
                None
                if violation is None
                else {
                    "message": violation.message,
                    "measurement_failed": violation.measurement_failed,
                }
            )
            print(json.dumps(payload, separators=(",", ":")), flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
