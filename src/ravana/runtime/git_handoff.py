"""Terminal handoff of a run's workspace back to the developer (§10.1).

§10.1: "Handoff back to the real project is a deliberate, reviewable step, not
an automatic one." When a run reaches `COMPLETED`, the work sitting on the
isolated workspace branch is surfaced as a **patch under
`runs/<run_id>/artifacts/handoff/`** — something the developer reads, applies
if they want it, and discards if they don't.

What this module must never do is the whole point of it: it never merges, never
rebases, never checks anything out in the developer's repo, and never writes
outside `runs/<run_id>/`. Every git invocation here is `-C <workspace>`, scoped
to the run's own clone. The isolation §10.1 buys during the run would be
worthless if delivery quietly spent it.

Two facts make the patch honest rather than approximate:

- The base is the **recorded provenance commit**, not a guess. The workspace's
  own `HEAD~n` or its origin's current tip could both have moved; provenance
  records what the run actually started from.
- A dirty worktree is reported, not swallowed. Agents do not reliably commit,
  so uncommitted work — **including brand-new untracked files, which are the
  most common thing a code-writing agent produces** — is captured as a separate
  `uncommitted.diff`. `git diff HEAD` alone would silently emit nothing for an
  untracked file, so those paths are marked intent-to-add first, and the diff
  is taken with `--binary` so a build artifact or image survives as something
  `git apply` can actually restore (see `_write_patches`). Inventing a commit to
  tidy this up would put words in the agent's mouth; dropping it would lose the
  work outright.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ravana.runtime.git_exec import GitError, remove_tree, run_git
from ravana.runtime.git_workspace import read_provenance, workspace_paths

HANDOFF_DIRNAME = "handoff"
UNCOMMITTED_PATCH_NAME = "uncommitted.diff"

HandoffMode = Literal["patch", "no_changes"]


@dataclass(frozen=True)
class HandoffResult:
    """What the run left behind, and where the developer can find it.

    The counts always describe **the patch that was delivered**, never the
    workspace's current state — on a re-report they are read back off disk. A
    delivery receipt that described something other than what was delivered
    would be worse than no receipt.

    Fields are added here only when something reads them. A `head_commit` was
    drafted and dropped within this slice for that reason: the re-report path
    reads its counts off disk and cannot know the workspace HEAD at the time the
    patch was cut, so the field would have had to be either absent or untrue on
    exactly the path this docstring exists to keep honest. This type is new in
    this slice and has never been importable outside it, so its shape is still
    being settled rather than a compatibility surface.
    """

    mode: HandoffMode
    branch: str
    base_commit: str
    commit_count: int
    has_uncommitted_changes: bool
    # None exactly when mode == "no_changes" — nothing was written.
    patch_dir: Path | None
    # True when this call found an earlier handoff and reported it unchanged.
    previously_handed_off: bool = False

    def summary(self) -> str:
        if self.mode == "no_changes":
            return (
                f"run branch {self.branch} has no commits beyond its base "
                f"{self.base_commit[:12]} and a clean worktree — nothing to hand off"
            )
        parts = [f"{self.commit_count} commit(s)"]
        if self.has_uncommitted_changes:
            parts.append("uncommitted changes")
        verb = "already handed off" if self.previously_handed_off else "handed off"
        return (
            f"{verb} {' + '.join(parts)} from {self.branch} as a patch in "
            f"{self.patch_dir} — apply with `git am` (never auto-merged)"
        )


async def hand_off_run(
    *,
    runs_dir: Path,
    run_id: str,
    git: str = "git",
) -> HandoffResult:
    """Surface the run branch as a patch and return what was produced.

    Idempotent: an existing handoff directory is reported as-is rather than
    rewritten, so a re-entry (a resumed drain, a retried finalization) cannot
    produce a second copy or clobber a patch the developer is already reading.
    """
    _, run_dir, workspace = workspace_paths(runs_dir, run_id)
    # Provenance and the patch set both live in the run dir, NOT in the
    # workspace, so re-reporting an existing handoff is answered entirely from
    # durable state. It deliberately runs before any workspace inspection: once
    # the patch exists it is the delivered artifact, and a workspace that was
    # cleaned up or moved off its branch afterwards cannot invalidate it.
    provenance = read_provenance(run_dir)
    artifacts_dir, patch_dir = _artifacts_paths(run_dir)
    if patch_dir.exists():
        # Counts come off the patch set itself: this result describes what the
        # developer will actually find on disk, not what the workspace holds now.
        return HandoffResult(
            mode="patch",
            branch=provenance.branch,
            base_commit=provenance.base_commit,
            commit_count=len(list(patch_dir.glob("*.patch"))),
            has_uncommitted_changes=(patch_dir / UNCOMMITTED_PATCH_NAME).exists(),
            patch_dir=patch_dir,
            previously_handed_off=True,
        )

    # Producing a NEW patch does need a sound workspace: the diff has to be
    # against the history provenance recorded, or it misrepresents the run.
    if not workspace.is_dir():
        raise GitError(f"run workspace does not exist, nothing to hand off: {workspace}")
    await _assert_on_run_branch(workspace, expected_branch=provenance.branch, git=git)

    commit_count = await _count_commits(
        workspace, base=provenance.base_commit, git=git
    )
    dirty = await _has_uncommitted_changes(workspace, git=git)
    if commit_count == 0 and not dirty:
        return HandoffResult(
            mode="no_changes",
            branch=provenance.branch,
            base_commit=provenance.base_commit,
            commit_count=0,
            has_uncommitted_changes=False,
            patch_dir=None,
        )

    await _write_patches(
        workspace,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        patch_dir=patch_dir,
        base_commit=provenance.base_commit,
        include_uncommitted=dirty,
        git=git,
    )
    return HandoffResult(
        mode="patch",
        branch=provenance.branch,
        base_commit=provenance.base_commit,
        commit_count=commit_count,
        has_uncommitted_changes=dirty,
        patch_dir=patch_dir,
    )


def _artifacts_paths(run_dir: Path) -> tuple[Path, Path]:
    """Resolve `artifacts/` and `artifacts/handoff/`, refusing aliased ones.

    `mkdir(exist_ok=True)` succeeds against a symlink pointing at an existing
    directory, and the later `rename` then resolves through it — so a symlinked
    `artifacts/` writes the patch outside `runs/<run_id>/` entirely while the
    returned path still claims it landed inside (verified: the patch escaped).
    Nothing an agent controls can plant that link today, since the sandbox
    mounts only `workspace/`; this holds the same line `workspace_paths` already
    holds for the run dir, so the guarantee doesn't depend on that mount
    remaining the only writer.
    """
    for candidate in (run_dir / "artifacts", run_dir / "artifacts" / HANDOFF_DIRNAME):
        if candidate.is_symlink() or (
            candidate.exists() and candidate.resolve() != candidate
        ):
            raise GitError(
                f"refusing an aliased handoff path (symlink out of the run dir): {candidate}"
            )
    return run_dir / "artifacts", run_dir / "artifacts" / HANDOFF_DIRNAME


async def _assert_on_run_branch(workspace: Path, *, expected_branch: str, git: str) -> None:
    """Fail closed if the workspace is not on the branch provenance recorded.

    A workspace that drifted off its run branch is not something to guess at:
    the diff would be against the wrong history, so the patch would misrepresent
    what the run did.
    """
    branch = await run_git(
        ["-C", str(workspace), "symbolic-ref", "--quiet", "--short", "HEAD"],
        git=git,
        check=False,
    )
    if branch.returncode != 0 or branch.stdout.strip() != expected_branch:
        raise GitError(
            f"run workspace is not on its recorded branch {expected_branch!r}; refusing to hand off: {workspace}"
        )


async def _count_commits(workspace: Path, *, base: str, git: str) -> int:
    result = await run_git(
        ["-C", str(workspace), "rev-list", "--count", f"{base}..HEAD"],
        git=git,
        check=False,
    )
    if result.returncode != 0:
        raise GitError(
            f"run workspace no longer contains its recorded base commit {base[:12]}: {workspace}"
        )
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise GitError(f"could not count commits on the run branch: {workspace}") from exc


async def _has_uncommitted_changes(workspace: Path, *, git: str) -> bool:
    # --porcelain covers staged, unstaged AND untracked; agents routinely leave
    # new files unstaged, and those are exactly the ones worth not losing.
    result = await run_git(
        ["-C", str(workspace), "status", "--porcelain"], git=git, check=True
    )
    return bool(result.stdout.strip())


async def _write_patches(
    workspace: Path,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    patch_dir: Path,
    base_commit: str,
    include_uncommitted: bool,
    git: str,
) -> None:
    """Build the patch set in a staging dir, then publish it with one rename.

    Same reason provisioning stages its clone: a partially-written handoff that
    a later read mistook for a complete one would understate what the run did.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    staging = run_dir / f".handoff-staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        await run_git(
            [
                "-C",
                str(workspace),
                "format-patch",
                f"{base_commit}..HEAD",
                "-o",
                str(staging),
            ],
            git=git,
            check=True,
        )
        if include_uncommitted:
            # `git diff HEAD` only sees paths git already knows about, so on its
            # own it captures NOTHING of a brand-new file — the single most
            # common thing a code-writing agent produces. `add -N` marks
            # untracked paths intent-to-add so their content lands in the diff.
            #
            # It stages no content, honours .gitignore (no node_modules/ in the
            # patch), and mutates only the index of this run's disposable clone
            # after the run is already terminal. The developer's repo is not
            # touched by it — that guarantee is unaffected.
            await run_git(
                ["-C", str(workspace), "add", "-N", "--", "."], git=git, check=True
            )
            # `--binary`: without it a binary file degrades to the single line
            # "Binary files ... differ" and `git apply` then refuses it outright
            # ("without full index line"), so the file is unrecoverable. The
            # committed path needs no flag — `format-patch` emits a GIT binary
            # patch by default — but plain `diff` does not.
            tracked = await run_git(
                ["-C", str(workspace), "diff", "--binary", "HEAD"], git=git, check=True
            )
            (staging / UNCOMMITTED_PATCH_NAME).write_text(
                tracked.stdout, encoding="utf-8"
            )
        os.rename(staging, patch_dir)  # atomic publish (same filesystem)
    except BaseException:
        await remove_tree(staging)
        raise
