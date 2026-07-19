"""Per-run git workspace provisioning (§10.1).

The Local tier installs `.ravana/` onto a project path, and agents need real
filesystem access to that project — but must NEVER touch the developer's actual
checkout. §10.1 resolves this: each run gets an **isolated clone** of the base
repo at `runs/<run_id>/workspace`, on its own branch `ravana/run-<run_id>`
created from the requested base ref.

Critically the clone is `--no-hardlinks`, NOT the plain `--local` hardlinking
clone. The sandbox bind-mounts the workspace — INCLUDING its `.git` — read-write,
so if the clone's objects were hardlinked to the source's, agent code writing
through a workspace object inode would corrupt the *same inode* in the
developer's repo (reproduced: `git fsck` on the source fails). `--no-hardlinks`
copies the objects, so the clone is a genuinely independent repository (its own
`.git`, refs, index, HEAD) — a bad `git reset --hard`/`rm -rf` in the sandbox
still can't reach the source. Provisioning is atomic (clone into a staging dir,
then `rename` into place) so an interrupted attempt never leaves a half-clone
that a retry would mistake for a finished workspace.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import uuid
from pathlib import Path

RUN_BRANCH_PREFIX = "ravana/run-"
DEFAULT_BASE_REF = "HEAD"
_GIT_TIMEOUT_SECONDS = 120


class GitError(Exception):
    """A git operation needed to provision a run workspace failed."""


def run_branch_name(run_id: str) -> str:
    return f"{RUN_BRANCH_PREFIX}{run_id}"


def git_toplevel(path: Path, *, git: str = "git") -> Path | None:
    """The root of the git working tree containing `path`, or None if `path`
    isn't inside one. Cloning must use the toplevel, not `path` itself — when
    `.ravana/` sits in a subdirectory of a monorepo, `path` is inside a work
    tree but `git clone <subdir>` fails; the toplevel is the real repo."""
    if not path.is_dir():
        return None
    result = _git(["-C", str(path), "rev-parse", "--show-toplevel"], git=git, check=False)
    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top) if top else None


def is_git_repo(path: Path, *, git: str = "git") -> bool:
    """True if `path` is inside a git working tree (the precondition for using
    it — or its toplevel — as a run's base repo)."""
    return git_toplevel(path, git=git) is not None


def provision_run_workspace(
    *,
    base_repo: Path,
    runs_dir: Path,
    run_id: str,
    base_ref: str = DEFAULT_BASE_REF,
    git: str = "git",
) -> Path:
    """Clone `base_repo`'s toplevel into `runs/<run_id>/workspace` on a fresh
    `ravana/run-<run_id>` branch from `base_ref`, and return the workspace path.

    Idempotent AND crash-safe: the clone lands in a staging directory and is
    `rename`d into place only once complete, so `workspace` exists iff it was
    fully provisioned. A resumed run / retry that finds a valid workspace keeps
    it (re-cloning would wipe the run's work); one that finds a *half*-written
    workspace (which the atomic publish makes near-impossible, but verify rather
    than trust) fails loudly rather than running against a broken repo.
    """
    runs_dir = runs_dir.resolve()
    run_dir = runs_dir / run_id
    workspace = (run_dir / "workspace").resolve()
    # §10.1: the workspace must stay under the runs dir. run_id is engine-minted
    # (a UUID), but verify rather than trust so a bad id can't escape it.
    if not workspace.is_relative_to(runs_dir):
        raise GitError(f"refusing a workspace path outside the runs dir ({run_id!r})")

    top = git_toplevel(base_repo, git=git)
    if top is None:
        raise GitError(f"base repo is not a git working tree: {base_repo.resolve()}")
    top = top.resolve()

    if workspace.exists():
        _validate_existing_workspace(
            workspace,
            source_toplevel=top,
            expected_branch=run_branch_name(run_id),
            git=git,
        )
        return workspace  # fully provisioned before — keep the run's work

    base_commit = _resolve_commit(top, base_ref, git=git)

    run_dir.mkdir(parents=True, exist_ok=True)
    staging = run_dir / f".workspace-staging-{uuid.uuid4().hex}"
    try:
        # `--no-hardlinks`: copy objects so the clone shares NO inodes with the
        # source (see module docstring) — the writable sandbox mount makes
        # hardlinks a source-corruption vector.
        _git(["clone", "--no-hardlinks", str(top), str(staging)], git=git, check=True)
        _git(
            ["-C", str(staging), "checkout", "-b", run_branch_name(run_id), base_commit],
            git=git,
            check=True,
        )
        os.rename(staging, workspace)  # atomic publish (same filesystem)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return workspace


def _resolve_commit(repo: Path, base_ref: str, *, git: str) -> str:
    if not base_ref:
        raise GitError("base ref must not be empty")
    result = _git(
        ["-C", str(repo), "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        git=git,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise GitError(f"base ref does not resolve to a commit: {base_ref!r}")
    return commit


def _validate_existing_workspace(
    workspace: Path,
    *,
    source_toplevel: Path,
    expected_branch: str,
    git: str,
) -> None:
    workspace_toplevel = git_toplevel(workspace, git=git)
    if workspace_toplevel is None or workspace_toplevel.resolve() != workspace:
        raise GitError(
            f"existing run workspace is not an independent git repo (partial/corrupt): {workspace}"
        )

    branch = _git(
        ["-C", str(workspace), "symbolic-ref", "--quiet", "--short", "HEAD"],
        git=git,
        check=False,
    )
    if branch.returncode != 0 or branch.stdout.strip() != expected_branch:
        raise GitError(
            f"existing run workspace is not on expected branch {expected_branch!r}: {workspace}"
        )

    origin = _git(
        ["-C", str(workspace), "config", "--get", "remote.origin.url"],
        git=git,
        check=False,
    )
    origin_path = origin.stdout.strip()
    if (
        origin.returncode != 0
        or not origin_path
        or Path(origin_path).expanduser().resolve() != source_toplevel
    ):
        raise GitError(
            f"existing run workspace does not belong to source repo {source_toplevel}: {workspace}"
        )


def _git(args: list[str], *, git: str, check: bool) -> subprocess.CompletedProcess[str]:
    # Run in its own process group so a timeout can kill git AND any helper /
    # hook subprocess it spawned — `subprocess.run(timeout=)` only kills the
    # direct child, orphaning descendants.
    try:
        proc = subprocess.Popen(
            [git, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise GitError(f"'{git}' not found on PATH — the Local tier needs git for workspace isolation (§10.1)") from exc
    try:
        stdout, stderr = proc.communicate(timeout=_GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)  # reap
        raise GitError(f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s") from exc
    except BaseException:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)
        raise
    if check and proc.returncode != 0:
        # git's stderr names paths/refs, not credentials; it's the actionable part.
        raise GitError(f"git {args[0]} failed (exit {proc.returncode}): {(stderr or '').strip()}")
    return subprocess.CompletedProcess(args, proc.returncode or 0, stdout, stderr)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    # Best-effort: killing the group (git + any helper/hook) must never mask
    # the timeout that triggered it.
    with contextlib.suppress(Exception):
        # start_new_session=True makes the child's pid its process-group id.
        # Using the saved id still works if the direct child has just exited
        # while one of its descendants keeps a pipe open.
        os.killpg(proc.pid, signal.SIGKILL)
