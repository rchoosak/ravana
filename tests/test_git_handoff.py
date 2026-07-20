"""Terminal handoff of a run workspace back to the developer (§10.1).

§10.1 makes handoff "a deliberate, reviewable step, not an automatic one", so
the load-bearing assertions here are about what handoff does NOT do: it must
never merge, never move the source's HEAD or branches, and never write outside
the run directory. Real temp git repos throughout — a mocked git would prove
nothing about the isolation these tests exist to protect.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from ravana.runtime.git_exec import GitError
from ravana.runtime.git_handoff import hand_off_run
from ravana.runtime.git_workspace import provision_run_workspace, run_branch_name

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    )


def _make_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "README.md").write_text("base\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return path


async def _workspace(tmp_path, *, runs=None):
    base = _make_repo(tmp_path / "project")
    runs = runs or tmp_path / "runs"
    ws = await provision_run_workspace(base_repo=base, runs_dir=runs, run_id="r")
    _git(ws, "config", "user.email", "agent@ravana.local")
    _git(ws, "config", "user.name", "Agent")
    return base, runs, ws


def _agent_commit(ws, filename, content, message):
    (ws / filename).write_text(content)
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", message)


def _source_fingerprint(base):
    """Everything about the developer's repo handoff must leave alone."""
    return {
        "head": _git(base, "rev-parse", "HEAD").stdout.strip(),
        "branches": _git(base, "branch", "--list", "--all").stdout,
        "status": _git(base, "status", "--porcelain").stdout,
        "log": _git(base, "log", "--oneline", "--all").stdout,
        "readme": (base / "README.md").read_text(),
    }


async def test_commits_on_the_run_branch_become_a_patch(tmp_path):
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")

    result = await hand_off_run(runs_dir=runs, run_id="r")

    assert result.mode == "patch"
    assert result.commit_count == 1
    assert result.branch == run_branch_name("r")
    assert result.has_uncommitted_changes is False
    patches = sorted(p.name for p in result.patch_dir.iterdir())
    assert patches == ["0001-add-feature.patch"]
    assert "feature.py" in (result.patch_dir / patches[0]).read_text()


async def test_handoff_never_touches_the_source_repo(tmp_path):
    # The §10.1 guarantee: delivery must not spend the isolation the run bought.
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")
    (ws / "scratch.txt").write_text("uncommitted")
    before = _source_fingerprint(base)

    await hand_off_run(runs_dir=runs, run_id="r")

    assert _source_fingerprint(base) == before
    assert not (base / "feature.py").exists()
    assert not (base / "scratch.txt").exists()
    # No run branch leaked into the developer's repo.
    assert _git(base, "branch", "--list", "ravana/*").stdout.strip() == ""


async def test_produced_patch_actually_applies_to_the_base(tmp_path):
    # A patch that doesn't apply is not a handoff, it's a receipt.
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")
    result = await hand_off_run(runs_dir=runs, run_id="r")

    applied = _make_repo(tmp_path / "reviewer")
    _git(applied, "am", str(result.patch_dir / "0001-add-feature.patch"))

    assert (applied / "feature.py").read_text() == "print('hi')\n"
    assert "add feature" in _git(applied, "log", "--oneline").stdout


async def test_uncommitted_work_is_captured_not_dropped(tmp_path):
    # Agents do not reliably commit; silently losing their work would be worse
    # than handing back a messy patch. A brand-new uncommitted file is the most
    # common thing a code-writing agent produces, so its CONTENT — not merely
    # its name — has to survive the handoff.
    base, runs, ws = await _workspace(tmp_path)
    (ws / "README.md").write_text("edited by agent\n")  # tracked, uncommitted
    (ws / "brand_new.py").write_text("def important():\n    return 42\n")  # untracked

    result = await hand_off_run(runs_dir=runs, run_id="r")

    assert result.mode == "patch"
    assert result.commit_count == 0
    assert result.has_uncommitted_changes is True
    diff = (result.patch_dir / "uncommitted.diff").read_text()
    assert "edited by agent" in diff
    assert "def important():" in diff and "return 42" in diff


async def test_uncommitted_patch_applies_and_restores_the_agents_work(tmp_path):
    # The real test of "captured": a reviewer can reconstruct the work from the
    # patch alone. Asserting a filename appears somewhere in the file would pass
    # even with every line of code missing.
    base, runs, ws = await _workspace(tmp_path)
    (ws / "README.md").write_text("edited by agent\n")
    (ws / "brand_new.py").write_text("def important():\n    return 42\n")
    result = await hand_off_run(runs_dir=runs, run_id="r")

    reviewer = _make_repo(tmp_path / "reviewer")
    _git(reviewer, "apply", str(result.patch_dir / "uncommitted.diff"))

    assert (reviewer / "brand_new.py").read_text() == "def important():\n    return 42\n"
    assert (reviewer / "README.md").read_text() == "edited by agent\n"


async def test_gitignored_files_stay_out_of_the_patch(tmp_path):
    # `add -N` honours .gitignore, so build output and dependencies don't get
    # swept into a patch a human has to read.
    base, runs, ws = await _workspace(tmp_path)
    (ws / ".gitignore").write_text("junk/\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "ignore junk")
    (ws / "junk").mkdir()
    (ws / "junk" / "huge.bin").write_text("NOISE THAT MUST NOT SHIP\n")
    (ws / "real.py").write_text("kept = True\n")

    result = await hand_off_run(runs_dir=runs, run_id="r")

    diff = (result.patch_dir / "uncommitted.diff").read_text()
    assert "kept = True" in diff
    assert "NOISE THAT MUST NOT SHIP" not in diff


async def test_clean_run_with_no_commits_writes_nothing(tmp_path):
    base, runs, ws = await _workspace(tmp_path)

    result = await hand_off_run(runs_dir=runs, run_id="r")

    assert result.mode == "no_changes"
    assert result.patch_dir is None
    assert result.commit_count == 0
    assert not (runs / "r" / "artifacts" / "handoff").exists()


async def test_handoff_is_idempotent_and_does_not_rewrite(tmp_path):
    # A re-entry (resumed drain, retried finalization) must not produce a second
    # copy or clobber a patch the developer may already be reading.
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")
    first = await hand_off_run(runs_dir=runs, run_id="r")
    (first.patch_dir / "0001-add-feature.patch").write_text("REVIEWER ANNOTATED\n")

    second = await hand_off_run(runs_dir=runs, run_id="r")

    assert second.patch_dir == first.patch_dir
    assert second.previously_handed_off is True
    assert (first.patch_dir / "0001-add-feature.patch").read_text() == "REVIEWER ANNOTATED\n"
    assert sorted(p.name for p in first.patch_dir.iterdir()) == ["0001-add-feature.patch"]


async def test_re_report_describes_the_patch_not_the_moved_on_workspace(tmp_path):
    # A delivery receipt has to describe what was delivered. Recomputing the
    # counts from the live workspace would claim commits the patch on disk does
    # not contain.
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")
    first = await hand_off_run(runs_dir=runs, run_id="r")
    assert first.commit_count == 1

    _agent_commit(ws, "later.py", "print('later')\n", "work after handoff")
    (ws / "scratch.txt").write_text("dirty after handoff")

    second = await hand_off_run(runs_dir=runs, run_id="r")

    assert second.commit_count == 1  # the patch has one commit, not two
    assert second.has_uncommitted_changes is False  # and no uncommitted.diff in it
    assert "already handed off 1 commit(s)" in second.summary()


async def test_workspace_off_its_run_branch_is_refused(tmp_path):
    # Diffing against the wrong history would misrepresent what the run did.
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")
    _git(ws, "checkout", "-q", "-b", "some-other-branch")

    with pytest.raises(GitError, match="not on its recorded branch"):
        await hand_off_run(runs_dir=runs, run_id="r")


async def test_missing_workspace_is_refused(tmp_path):
    base, runs, ws = await _workspace(tmp_path)
    shutil.rmtree(ws)

    with pytest.raises(GitError, match="does not exist"):
        await hand_off_run(runs_dir=runs, run_id="r")


async def test_run_id_path_escape_is_refused(tmp_path):
    _make_repo(tmp_path / "project")
    with pytest.raises(GitError, match="invalid run id"):
        await hand_off_run(runs_dir=tmp_path / "runs", run_id="../evil")


async def test_patch_lands_under_the_run_artifacts_dir(tmp_path):
    # §10.1's layout: non-code artifacts live in runs/<run_id>/artifacts/.
    base, runs, ws = await _workspace(tmp_path)
    _agent_commit(ws, "feature.py", "print('hi')\n", "add feature")

    result = await hand_off_run(runs_dir=runs, run_id="r")

    assert result.patch_dir == runs / "r" / "artifacts" / "handoff"
    # Nothing was written into the workspace the sandbox mounts.
    assert not (ws / "artifacts").exists()
