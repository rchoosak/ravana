"""Shared git/subprocess execution for the §10.1 workspace modules.

Both workspace provisioning (`git_workspace`) and the terminal handoff
(`git_handoff`) shell out to git with the same requirements: run in a separate
process group so a timeout kills git AND any helper/hook it spawned, stay
cancellable inside the engine's event loop, and surface failures as `GitError`.

This lives in one module on purpose. The provisioning slice previously carried
two copies of this runner (a sync twin and an async one) and the copies drifted
out of test coverage; a second copy here would reintroduce the same problem.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess

GIT_TIMEOUT_SECONDS = 120
_CLEANUP_SECONDS = 5


class GitError(Exception):
    """A git operation needed by the run workspace lifecycle failed."""


async def run_git(
    args: list[str], *, git: str = "git", check: bool
) -> subprocess.CompletedProcess[str]:
    try:
        return await run_subprocess(
            [git, *args],
            operation=f"git {args[0]}",
            timeout_seconds=GIT_TIMEOUT_SECONDS,
            check=check,
        )
    except FileNotFoundError as exc:
        raise GitError(
            f"'{git}' not found on PATH — the Local tier needs git for workspace isolation (§10.1)"
        ) from exc


async def run_subprocess(
    argv: list[str],
    *,
    operation: str,
    timeout_seconds: float,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    # `start_new_session=True` puts the child in its own process group so a
    # timeout can kill it AND any helper / hook subprocess it spawned — killing
    # only the direct child would orphan its descendants.
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise GitError(f"{operation} could not start ({type(exc).__name__})") from exc

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        await stop_process_group(proc)
        raise GitError(f"{operation} timed out after {timeout_seconds:g}s") from exc
    except BaseException:
        await stop_process_group(proc)
        raise

    stdout = stdout_bytes.decode("utf-8", "replace")
    stderr = stderr_bytes.decode("utf-8", "replace")
    returncode = proc.returncode or 0
    if check and returncode != 0:
        # git's stderr names paths/refs, not credentials; it's the actionable part.
        raise GitError(f"{operation} failed (exit {returncode}): {stderr.strip()}")
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


async def stop_process_group(proc: asyncio.subprocess.Process) -> None:
    # Best-effort: killing the group (the child + any helper/hook) must never
    # mask the timeout or cancellation that triggered it.
    with contextlib.suppress(ProcessLookupError):
        # start_new_session=True makes the child's pid its process-group id.
        os.killpg(proc.pid, signal.SIGKILL)

    async def reap() -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.communicate(), timeout=_CLEANUP_SECONDS)

    # Shielded so cancellation during cleanup can't leave a zombie behind.
    cleanup = asyncio.create_task(reap())
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
