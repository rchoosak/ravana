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

Every entry point here is async: provisioning runs inside the engine's event
loop, and these are the functions the product actually calls. There is
deliberately no sync twin — a second copy of these validations is a second
place for a security check to drift.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

RUN_BRANCH_PREFIX = "ravana/run-"
DEFAULT_BASE_REF = "HEAD"
_GIT_TIMEOUT_SECONDS = 120
_GIT_CLEANUP_SECONDS = 5
_PROVENANCE_FILENAME = ".workspace-provenance.json"
_PROVENANCE_VERSION = 1
_GIT_CLONE_MODE = "git-clone-no-hardlinks"
_SHADOW_COPY_MODE = "shadow-copy"
_MAX_PROVENANCE_BYTES = 4096


class GitError(Exception):
    """A git operation needed to provision a run workspace failed."""


@dataclass(frozen=True)
class WorkspaceProvenance:
    """Trusted run metadata stored beside, never inside, the sandbox mount."""

    version: int
    isolation_mode: str
    source_toplevel: str
    branch: str
    base_commit: str


def run_branch_name(run_id: str) -> str:
    return f"{RUN_BRANCH_PREFIX}{run_id}"


async def git_toplevel(path: Path, *, git: str = "git") -> Path | None:
    """The root of the git working tree containing `path`, or None if `path`
    isn't inside one. Cloning must use the toplevel, not `path` itself — when
    `.ravana/` sits in a subdirectory of a monorepo, `path` is inside a work
    tree but `git clone <subdir>` fails; the toplevel is the real repo."""
    if not path.is_dir():
        return None
    result = await _git(
        ["-C", str(path), "rev-parse", "--show-toplevel"],
        git=git,
        check=False,
    )
    top = result.stdout.strip()
    return Path(top) if result.returncode == 0 and top else None


async def provision_run_workspace(
    *,
    base_repo: Path,
    runs_dir: Path,
    run_id: str,
    base_ref: str | None = DEFAULT_BASE_REF,
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
    _, run_dir, workspace = _workspace_paths(runs_dir, run_id)
    top = await git_toplevel(base_repo, git=git)
    if top is None:
        raise GitError(f"base repo is not a git working tree: {base_repo.resolve()}")
    top = top.resolve()

    if workspace.exists():
        requested_base_commit = (
            await _resolve_commit(top, base_ref, git=git)
            if base_ref is not None
            else None
        )
        await _validate_existing_workspace(
            workspace,
            run_dir=run_dir,
            source_toplevel=top,
            expected_branch=run_branch_name(run_id),
            expected_mode=_GIT_CLONE_MODE,
            requested_base_commit=requested_base_commit,
            git=git,
        )
        return workspace  # fully provisioned before — keep the run's work

    if _provenance_path(run_dir).exists():
        raise GitError(
            f"workspace provenance exists without a workspace (partial/corrupt): {run_dir}"
        )
    if base_ref is None:
        raise GitError("a base ref is required when provisioning a new git workspace")
    base_commit = await _resolve_commit(top, base_ref, git=git)

    run_dir.mkdir(parents=True, exist_ok=True)
    staging = run_dir / f".workspace-staging-{uuid.uuid4().hex}"
    published = False
    try:
        # `--no-hardlinks`: copy objects so the clone shares NO inodes with the
        # source (see module docstring) — the writable sandbox mount makes
        # hardlinks a source-corruption vector.
        await _git(
            ["clone", "--no-hardlinks", str(top), str(staging)],
            git=git,
            check=True,
        )
        await _git(
            [
                "-C",
                str(staging),
                "checkout",
                "-b",
                run_branch_name(run_id),
                base_commit,
            ],
            git=git,
            check=True,
        )
        os.rename(staging, workspace)  # atomic publish (same filesystem)
        published = True
        _write_provenance(
            run_dir,
            WorkspaceProvenance(
                version=_PROVENANCE_VERSION,
                isolation_mode=_GIT_CLONE_MODE,
                source_toplevel=str(top),
                branch=run_branch_name(run_id),
                base_commit=base_commit,
            ),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(workspace, ignore_errors=True)
            with contextlib.suppress(OSError):
                _provenance_path(run_dir).unlink()
        raise
    return workspace


async def provision_shadow_workspace(
    *,
    project_dir: Path,
    runs_dir: Path,
    run_id: str,
    git: str = "git",
) -> Path:
    """Snapshot a non-git project into an isolated shadow repository.

    Same atomic-publish and idempotency contract as `provision_run_workspace`;
    the base is a copy of the project rather than a clone, committed onto the
    run branch so the agent still gets git semantics.
    """
    _, run_dir, workspace = _workspace_paths(runs_dir, run_id)
    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise GitError(f"shadow workspace source is not a directory: {project_dir}")

    if workspace.exists():
        await _validate_existing_workspace(
            workspace,
            run_dir=run_dir,
            source_toplevel=project_dir,
            expected_branch=run_branch_name(run_id),
            expected_mode=_SHADOW_COPY_MODE,
            requested_base_commit=None,
            git=git,
        )
        return workspace
    if _provenance_path(run_dir).exists():
        raise GitError(
            f"workspace provenance exists without a workspace (partial/corrupt): {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    staging = run_dir / f".workspace-staging-{uuid.uuid4().hex}"
    published = False
    try:
        # A killable child process: a large `copytree` in-process would block the
        # event loop and ignore cancellation.
        await _run_subprocess(
            [
                sys.executable,
                "-m",
                "ravana.runtime.shadow_copy",
                str(project_dir),
                str(staging),
            ],
            operation="non-git project snapshot",
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
        await _initialize_shadow_repo(staging, run_id=run_id, git=git)
        base_commit = (
            await _git(
                ["-C", str(staging), "rev-parse", "--verify", "HEAD^{commit}"],
                git=git,
                check=True,
            )
        ).stdout.strip()
        os.rename(staging, workspace)  # atomic publish (same filesystem)
        published = True
        _write_provenance(
            run_dir,
            WorkspaceProvenance(
                version=_PROVENANCE_VERSION,
                isolation_mode=_SHADOW_COPY_MODE,
                source_toplevel=str(project_dir),
                branch=run_branch_name(run_id),
                base_commit=base_commit,
            ),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(workspace, ignore_errors=True)
            with contextlib.suppress(OSError):
                _provenance_path(run_dir).unlink()
        raise
    return workspace


async def _resolve_commit(repo: Path, base_ref: str, *, git: str) -> str:
    if not base_ref:
        raise GitError("base ref must not be empty")
    result = await _git(
        ["-C", str(repo), "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        git=git,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise GitError(f"base ref does not resolve to a commit: {base_ref!r}")
    return commit


def _workspace_paths(runs_dir: Path, run_id: str) -> tuple[Path, Path, Path]:
    runs_dir = runs_dir.resolve()
    if (
        not run_id
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
        or run_id in (".", "..")
    ):
        raise GitError(f"refusing an invalid run id ({run_id!r})")

    run_dir = runs_dir / run_id
    workspace = run_dir / "workspace"
    if (
        not run_dir.resolve().is_relative_to(runs_dir)
        or run_dir.is_symlink()
        or run_dir.resolve() != run_dir
    ):
        raise GitError(f"refusing an aliased run directory ({run_id!r})")
    if workspace.is_symlink() or workspace.resolve() != workspace:
        raise GitError(f"refusing an aliased workspace ({run_id!r})")
    return runs_dir, run_dir, workspace


def _provenance_path(run_dir: Path) -> Path:
    return run_dir / _PROVENANCE_FILENAME


def _write_provenance(run_dir: Path, provenance: WorkspaceProvenance) -> None:
    path = _provenance_path(run_dir)
    temp_path = run_dir / f".{_PROVENANCE_FILENAME}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(
        {
            "version": provenance.version,
            "isolation_mode": provenance.isolation_mode,
            "source_toplevel": provenance.source_toplevel,
            "branch": provenance.branch,
            "base_commit": provenance.base_commit,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_PROVENANCE_BYTES:
        raise GitError("workspace provenance exceeds its size limit")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(temp_path, flags, 0o600)
        with os.fdopen(fd, "wb") as output:
            fd = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        raise GitError(
            f"workspace provenance could not be written ({type(exc).__name__})"
        ) from exc
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            temp_path.unlink()


def _read_provenance(run_dir: Path) -> WorkspaceProvenance:
    path = _provenance_path(run_dir)
    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise GitError(f"workspace has no trusted no-hardlinks provenance: {path}")
        if path_stat.st_size > _MAX_PROVENANCE_BYTES:
            raise GitError(f"workspace provenance exceeds its size limit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except GitError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitError(f"workspace provenance is missing or invalid: {path}") from exc

    if not isinstance(payload, dict):
        raise GitError(f"workspace provenance is invalid: {path}")
    version = payload.get("version")
    isolation_mode = payload.get("isolation_mode")
    source_toplevel = payload.get("source_toplevel")
    branch = payload.get("branch")
    base_commit = payload.get("base_commit")
    fields = (version, isolation_mode, source_toplevel, branch, base_commit)
    if type(fields[0]) is not int or any(
        type(value) is not str or not value for value in fields[1:]
    ):
        raise GitError(f"workspace provenance is invalid: {path}")
    return WorkspaceProvenance(
        version=cast(int, version),
        isolation_mode=cast(str, isolation_mode),
        source_toplevel=cast(str, source_toplevel),
        branch=cast(str, branch),
        base_commit=cast(str, base_commit),
    )


async def _validate_existing_workspace(
    workspace: Path,
    *,
    run_dir: Path,
    source_toplevel: Path,
    expected_branch: str,
    expected_mode: str,
    requested_base_commit: str | None,
    git: str,
) -> None:
    workspace_toplevel = await git_toplevel(workspace, git=git)
    if workspace_toplevel is None or workspace_toplevel.resolve() != workspace:
        raise GitError(
            f"existing run workspace is not an independent git repo (partial/corrupt): {workspace}"
        )

    branch = await _git(
        ["-C", str(workspace), "symbolic-ref", "--quiet", "--short", "HEAD"],
        git=git,
        check=False,
    )
    if branch.returncode != 0 or branch.stdout.strip() != expected_branch:
        raise GitError(
            f"existing run workspace is not on expected branch {expected_branch!r}: {workspace}"
        )

    if expected_mode == _GIT_CLONE_MODE:
        origin = await _git(
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

    _validate_provenance(
        _read_provenance(run_dir),
        source_toplevel=source_toplevel,
        expected_branch=expected_branch,
        expected_mode=expected_mode,
        requested_base_commit=requested_base_commit,
    )


def _validate_provenance(
    provenance: WorkspaceProvenance,
    *,
    source_toplevel: Path,
    expected_branch: str,
    expected_mode: str,
    requested_base_commit: str | None,
) -> None:
    if (
        provenance.version != _PROVENANCE_VERSION
        or provenance.isolation_mode != expected_mode
        or Path(provenance.source_toplevel).expanduser().resolve() != source_toplevel
        or provenance.branch != expected_branch
    ):
        raise GitError(
            "existing run workspace provenance does not match the requested source"
        )
    if (
        requested_base_commit is not None
        and provenance.base_commit != requested_base_commit
    ):
        raise GitError(
            "existing run workspace was provisioned from a different base commit"
        )


async def _initialize_shadow_repo(staging: Path, *, run_id: str, git: str) -> None:
    commands = [
        ["-C", str(staging), "init", "-q"],
        ["-C", str(staging), "config", "user.name", "Ravana"],
        ["-C", str(staging), "config", "user.email", "ravana@local"],
        ["-C", str(staging), "add", "-A"],
        [
            "-C",
            str(staging),
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Ravana shadow base",
        ],
        ["-C", str(staging), "checkout", "-q", "-b", run_branch_name(run_id)],
    ]
    for command in commands:
        await _git(command, git=git, check=True)


async def _git(
    args: list[str], *, git: str, check: bool
) -> subprocess.CompletedProcess[str]:
    try:
        return await _run_subprocess(
            [git, *args],
            operation=f"git {args[0]}",
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
            check=check,
        )
    except FileNotFoundError as exc:
        raise GitError(
            f"'{git}' not found on PATH — the Local tier needs git for workspace isolation (§10.1)"
        ) from exc


async def _run_subprocess(
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
        await _stop_process_group(proc)
        raise GitError(f"{operation} timed out after {timeout_seconds:g}s") from exc
    except BaseException:
        await _stop_process_group(proc)
        raise

    stdout = stdout_bytes.decode("utf-8", "replace")
    stderr = stderr_bytes.decode("utf-8", "replace")
    returncode = proc.returncode or 0
    if check and returncode != 0:
        # git's stderr names paths/refs, not credentials; it's the actionable part.
        raise GitError(f"{operation} failed (exit {returncode}): {stderr.strip()}")
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


async def _stop_process_group(proc: asyncio.subprocess.Process) -> None:
    # Best-effort: killing the group (the child + any helper/hook) must never
    # mask the timeout or cancellation that triggered it.
    with contextlib.suppress(ProcessLookupError):
        # start_new_session=True makes the child's pid its process-group id.
        os.killpg(proc.pid, signal.SIGKILL)

    async def reap() -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.communicate(), timeout=_GIT_CLEANUP_SECONDS)

    # Shielded so cancellation during cleanup can't leave a zombie behind.
    cleanup = asyncio.create_task(reap())
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
