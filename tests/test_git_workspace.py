"""Per-run git workspace provisioning (§10.1). Uses real temp git repos to
prove the `--no-hardlinks` run workspace is independent from the source.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from ravana.runtime.git_workspace import (
    GitError,
    git_toplevel,
    is_git_repo,
    provision_run_workspace,
    run_branch_name,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


def _make_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "README.md").write_text("base\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return path


def test_provisions_isolated_clone_on_run_branch(tmp_path):
    base = _make_repo(tmp_path / "project")
    runs = tmp_path / ".ravana" / "runs"
    ws = provision_run_workspace(base_repo=base, runs_dir=runs, run_id="run-1")

    assert ws == (runs / "run-1" / "workspace").resolve()
    assert (ws / ".git").exists()  # a fully independent repository
    assert (ws / "README.md").read_text() == "base\n"  # base content is present
    branch = _git(ws, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert branch == run_branch_name("run-1") == "ravana/run-run-1"


def test_workspace_changes_cannot_reach_the_base_checkout(tmp_path):
    # The whole point of §10.1: even destructive work in the clone leaves the
    # developer's actual checkout untouched.
    base = _make_repo(tmp_path / "project")
    ws = provision_run_workspace(base_repo=base, runs_dir=tmp_path / "runs", run_id="r")

    (ws / "README.md").write_text("MUTATED BY AGENT")
    (ws / "new_file.txt").write_text("agent output")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "agent work")

    assert (base / "README.md").read_text() == "base\n"  # base working tree untouched
    assert not (base / "new_file.txt").exists()
    assert _git(base, "log", "--oneline").stdout.strip().count("\n") == 0  # base still one commit
    assert _git(base, "branch", "--list", "ravana/*").stdout.strip() == ""  # run branch isn't in base


def test_idempotent_returns_existing_without_reclone(tmp_path):
    base = _make_repo(tmp_path / "project")
    runs = tmp_path / "runs"
    ws = provision_run_workspace(base_repo=base, runs_dir=runs, run_id="r")
    (ws / "in_progress.txt").write_text("do not clobber")

    ws2 = provision_run_workspace(base_repo=base, runs_dir=runs, run_id="r")
    assert ws2 == ws
    assert (ws / "in_progress.txt").read_text() == "do not clobber"  # re-clone would have wiped it


def test_existing_workspace_on_wrong_branch_is_rejected(tmp_path):
    base = _make_repo(tmp_path / "project")
    runs = tmp_path / "runs"
    ws = provision_run_workspace(base_repo=base, runs_dir=runs, run_id="r")
    source_branch = _git(base, "branch", "--show-current").stdout.strip()
    _git(ws, "checkout", "-q", source_branch)

    with pytest.raises(GitError, match="expected branch"):
        provision_run_workspace(base_repo=base, runs_dir=runs, run_id="r")


def test_existing_workspace_from_wrong_source_is_rejected(tmp_path):
    expected_source = _make_repo(tmp_path / "expected")
    other_source = _make_repo(tmp_path / "other")
    runs = tmp_path / "runs"
    provision_run_workspace(
        base_repo=other_source,
        runs_dir=runs,
        run_id="r",
    )

    with pytest.raises(GitError, match="does not belong to source repo"):
        provision_run_workspace(
            base_repo=expected_source,
            runs_dir=runs,
            run_id="r",
        )


def test_requested_base_ref_is_used(tmp_path):
    base = _make_repo(tmp_path / "project")
    first_commit = _git(base, "rev-parse", "HEAD").stdout.strip()
    (base / "README.md").write_text("second\n")
    _git(base, "commit", "-qam", "second")

    ws = provision_run_workspace(
        base_repo=base,
        runs_dir=tmp_path / "runs",
        run_id="r",
        base_ref=first_commit,
    )
    assert _git(ws, "rev-parse", "HEAD").stdout.strip() == first_commit
    assert (ws / "README.md").read_text() == "base\n"


def test_invalid_base_ref_is_rejected_without_publishing_workspace(tmp_path):
    base = _make_repo(tmp_path / "project")
    runs = tmp_path / "runs"
    with pytest.raises(GitError, match="does not resolve"):
        provision_run_workspace(
            base_repo=base,
            runs_dir=runs,
            run_id="r",
            base_ref="refs/heads/does-not-exist",
        )
    assert not (runs / "r" / "workspace").exists()


def test_clone_objects_are_independent_of_the_source(tmp_path):
    # P0: the sandbox mounts the workspace `.git` writable, so a hardlinked
    # clone would let agent code corrupt the SOURCE repo's objects. --no-hardlinks
    # copies them: corrupting a clone object must leave the source's fsck clean.
    base = _make_repo(tmp_path / "project")
    ws = provision_run_workspace(base_repo=base, runs_dir=tmp_path / "runs", run_id="r")
    clone_objects = [p for p in (ws / ".git" / "objects").rglob("*") if p.is_file()]
    assert clone_objects, "expected packed/loose objects in the clone"
    victim = clone_objects[0]
    os.chmod(victim, 0o644)
    victim.write_bytes(b"CORRUPTED BY AGENT")
    fsck = subprocess.run(["git", "-C", str(base), "fsck"], capture_output=True, text=True)
    assert fsck.returncode == 0, f"source repo was corrupted via the clone: {fsck.stderr}"


def test_provisions_from_monorepo_toplevel_when_ravana_in_a_subdir(tmp_path):
    # §10.1 nested-project: given a path INSIDE a work tree (a monorepo subdir),
    # provisioning clones the repo TOPLEVEL, not the subdir (which git can't clone).
    base = _make_repo(tmp_path / "monorepo")
    subdir = base / "packages" / "app"
    subdir.mkdir(parents=True)
    ws = provision_run_workspace(base_repo=subdir, runs_dir=tmp_path / "runs", run_id="r")
    assert (ws / ".git").exists()
    assert (ws / "README.md").read_text() == "base\n"  # toplevel content


def test_partial_workspace_is_rejected_not_silently_reused(tmp_path):
    # A half-written workspace (interrupted provision) must NOT be accepted as a
    # finished one — the run would execute against a broken repo.
    base = _make_repo(tmp_path / "project")
    # Production puts runs below the source worktree. A plain workspace must
    # not inherit the source's ancestor .git and pass validation.
    runs = base / ".ravana" / "runs"
    partial = runs / "r" / "workspace"
    partial.mkdir(parents=True)
    (partial / "junk").write_text("half")  # exists, but not a git repo
    with pytest.raises(GitError, match="not an independent git repo"):
        provision_run_workspace(base_repo=base, runs_dir=runs, run_id="r")


def test_git_toplevel(tmp_path):
    base = _make_repo(tmp_path / "repo")
    sub = base / "a" / "b"
    sub.mkdir(parents=True)
    assert git_toplevel(sub).resolve() == base.resolve()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert git_toplevel(plain) is None


def test_non_git_base_raises(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitError, match="not a git working tree"):
        provision_run_workspace(base_repo=plain, runs_dir=tmp_path / "runs", run_id="r")


def test_workspace_path_escape_is_refused(tmp_path):
    base = _make_repo(tmp_path / "project")
    with pytest.raises(GitError, match="outside the runs dir"):
        provision_run_workspace(base_repo=base, runs_dir=tmp_path / "runs", run_id="../evil")


def test_is_git_repo(tmp_path):
    assert is_git_repo(_make_repo(tmp_path / "repo")) is True
    plain = tmp_path / "plain"
    plain.mkdir()
    assert is_git_repo(plain) is False
    assert is_git_repo(tmp_path / "does-not-exist") is False
