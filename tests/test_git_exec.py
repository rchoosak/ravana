"""Shared git/subprocess execution helpers (§10.1).

`remove_tree` exists instead of a bare `shutil.rmtree` or a bare
`asyncio.to_thread` because the workspace modules delete *staging clones* from
`except BaseException:` blocks. That places two requirements in tension: the
delete must not stall the engine's event loop, and it must still finish when
the caller is being cancelled — otherwise a cancelled provision leaks a whole
repository. Both are asserted here.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import threading

from ravana.runtime import git_exec
from ravana.runtime.git_exec import remove_tree


def _populate(path, files=40):
    path.mkdir(parents=True)
    for i in range(files):
        (path / f"f{i}.txt").write_text("x" * 128)
    return path


async def test_remove_tree_runs_off_the_event_loop_thread(tmp_path, monkeypatch):
    # A sync rmtree of a real clone measured ~185ms of dead event loop, which
    # freezes every concurrent node. Asserting the thread identity rather than a
    # duration keeps this deterministic instead of timing-flaky.
    victim = _populate(tmp_path / "victim")
    seen: dict[str, int] = {}
    real_rmtree = shutil.rmtree

    def spy(path, ignore_errors=False):
        seen["thread"] = threading.get_ident()
        return real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(git_exec.shutil, "rmtree", spy)

    await remove_tree(victim)

    assert seen["thread"] != threading.get_ident()
    assert not victim.exists()


async def test_remove_tree_finishes_even_when_the_caller_is_cancelled(tmp_path):
    # The property a plain `await asyncio.to_thread(...)` would NOT have: under
    # cancellation it would re-raise at the await and abandon the delete,
    # leaking a full staging clone.
    victim = _populate(tmp_path / "victim", files=200)
    entered = asyncio.Event()

    async def cleanup_under_cancellation():
        entered.set()
        await remove_tree(victim)

    task = asyncio.create_task(cleanup_under_cancellation())
    await entered.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert not victim.exists()  # cleanup completed despite the cancellation


async def test_remove_tree_ignores_a_missing_path(tmp_path):
    # Cleanup runs on failure paths where the tree may never have been created;
    # it must not raise a second error over the one being handled.
    await remove_tree(tmp_path / "never-existed")
