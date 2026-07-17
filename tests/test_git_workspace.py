"""Per-run git workspace provisioning (§10.1). Uses real temp git repos — the
clone is `--local` of a tiny repo, so it's fast — to prove the run workspace is
a fully independent clone that can't reach the base repo's checkout.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from ravana.runtime.git_workspace import GitError, is_git_repo, provision_run_workspace, run_branch_name

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
