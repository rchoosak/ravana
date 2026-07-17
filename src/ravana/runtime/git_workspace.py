"""Per-run git workspace provisioning (§10.1).

The Local tier installs `.ravana/` onto a project path, and agents need real
filesystem access to that project — but must NEVER touch the developer's actual
checkout. §10.1 resolves this: each run gets an **isolated `git clone --local`**
of the base repo at `runs/<run_id>/workspace`, on its own branch
`ravana/run-<run_id>`. A local clone is a fully independent repository (its own
`.git`, refs, index and HEAD), so there is categorically no way for anything the
agent does inside the sandbox — a bad `git reset --hard`, an `rm -rf`, not just a
bad edit — to reach the developer's checkout. `--local` hardlinks objects on the
same filesystem, so it costs about as much as a `git worktree` without the
shared-object-store caveat.

The `code_interpreter` sandbox then bind-mounts ONLY that workspace, so the
isolation holds at the mount level too (§10.1 point 4). Handing results back to
the real project (a PR or patch from the run branch) is a separate, deliberate,
reviewable step — never an auto-merge — and lands in its own slice.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

RUN_BRANCH_PREFIX = "ravana/run-"
_GIT_TIMEOUT_SECONDS = 120


class GitError(Exception):
    """A git operation needed to provision a run workspace failed."""


def run_branch_name(run_id: str) -> str:
    return f"{RUN_BRANCH_PREFIX}{run_id}"


def is_git_repo(path: Path, *, git: str = "git") -> bool:
    """True if `path` is inside a git working tree — the precondition for
    cloning it as a run's base repo."""
    if not path.is_dir():
        return False
    result = _git(["-C", str(path), "rev-parse", "--is-inside-work-tree"], git=git, check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def provision_run_workspace(*, base_repo: Path, runs_dir: Path, run_id: str, git: str = "git") -> Path:
    """Clone `base_repo` into `runs/<run_id>/workspace` on a fresh
    `ravana/run-<run_id>` branch and return the workspace path.

    Idempotent: if the workspace already exists (a resumed run, a retry), it is
    returned untouched — re-cloning would blow away work the run already did.
    The clone is `--local` (independent repo, hardlinked objects) so nothing the
    run does can reach `base_repo`'s working tree.
    """
    runs_dir = runs_dir.resolve()
    workspace = (runs_dir / run_id / "workspace").resolve()
    # §10.1: the workspace must stay under the runs dir. run_id is engine-minted
    # (a UUID), but verify rather than trust so a bad id can't escape it.
    if not workspace.is_relative_to(runs_dir):
        raise GitError(f"refusing a workspace path outside the runs dir ({run_id!r})")
    if workspace.exists():
        return workspace

    base = base_repo.resolve()
    if not is_git_repo(base, git=git):
        raise GitError(f"base repo is not a git working tree: {base}")

    workspace.parent.mkdir(parents=True, exist_ok=True)
    # `git clone` creates the target dir itself; it refuses a non-empty target,
    # which is the behavior we want (never clobber existing content).
    _git(["clone", "--local", str(base), str(workspace)], git=git, check=True)
    _git(["-C", str(workspace), "checkout", "-b", run_branch_name(run_id)], git=git, check=True)
    return workspace


def _git(args: list[str], *, git: str, check: bool) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [git, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise GitError(f"'{git}' not found on PATH — the Local tier needs git for workspace isolation (§10.1)") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s") from exc
    if check and result.returncode != 0:
        # git's stderr names paths/refs, not credentials; safe to surface, and
        # it's the actionable part of the failure.
        raise GitError(f"git {args[0]} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result
