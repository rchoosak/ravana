"""code_interpreter toolkit + sandbox (§8, §10.1). The Docker isolation flags
are asserted on the pure argv builder; the handler is exercised through a fake
SandboxRunner, so nothing here needs a Docker daemon.
"""

from __future__ import annotations

import asyncio
import fcntl
import multiprocessing
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import ravana.runtime.toolkits.code_interpreter as code_interpreter_module
import ravana.runtime.toolkits.sandbox as sandbox_module
from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolRetrySafeCancellation,
    ToolkitError,
    ToolOutcomeUnknown,
)
from ravana.runtime.toolkits.code_interpreter import CodeInterpreterHandler
from ravana.runtime.toolkits.sandbox import (
    DockerSandboxRunner,
    SandboxCancelledBeforeStart,
    SandboxError,
    SandboxLimits,
    SandboxResult,
    SandboxSpec,
    SandboxOutcomeUnknown,
    build_docker_argv,
    build_docker_start_argv,
)


class FakeRunner:
    """Captures the SandboxSpec it was handed and returns a scripted result (or
    raises), so the handler's request shaping + security scoping is testable
    without Docker."""

    def __init__(self, result: SandboxResult | None = None, raise_error: Exception | None = None):
        self.spec: SandboxSpec | None = None
        self._result = result or SandboxResult(exit_code=0, stdout="ok", stderr="")
        self._raise = raise_error

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        self.spec = spec
        if self._raise is not None:
            raise self._raise
        return self._result


def _spec(tmp_path, *, limits=None) -> SandboxSpec:
    return SandboxSpec(
        image="python:3.11-slim", argv=["python", "main.py"], workspace=tmp_path,
        limits=limits or SandboxLimits(),
    )


# --- docker argv (§8 isolation), pure + no daemon ---------------------------
def test_docker_argv_has_isolation_flags(tmp_path):
    startup_marker = "ravana-started-test"
    argv = build_docker_argv(
        _spec(tmp_path), name="ravana-ci-abc", startup_marker=startup_marker
    )
    assert argv[:3] == ["docker", "create", "--rm"]
    assert argv[argv.index("--network") + 1] == "none"  # §8: no default egress
    assert "--read-only" in argv  # root fs read-only
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    # memory == memory-swap: a hard RAM cap with no swap escape hatch
    assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1] == "2048m"
    assert argv[argv.index("--cpus") + 1] == "2.0"
    assert "--pids-limit" in argv
    assert argv[argv.index("--name") + 1] == "ravana-ci-abc"
    assert argv[argv.index("--user") + 1] == (
        f"{tmp_path.stat().st_uid}:{tmp_path.stat().st_gid}"
    )
    assert argv[argv.index("--log-driver") + 1] == "none"
    assert "--log-opt" not in argv
    ulimits = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--ulimit"]
    assert f"fsize={SandboxLimits().workspace_bytes}:{SandboxLimits().workspace_bytes}" in ulimits
    # exactly ONE bind mount, scoped strictly to the run's workspace (§10.1)
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts == [f"{tmp_path.resolve()}:/workspace:rw"]
    assert argv[argv.index("-w") + 1] == "/workspace"
    # The launcher emits an unguessable start marker before exec'ing the
    # in-container command, so Docker failures can be distinguished from an
    # agent process that itself exits with Docker's reserved code 125.
    image_index = argv.index("python:3.11-slim")
    assert argv[image_index + 1 : image_index + 5] == [
        "/bin/sh",
        "-c",
        'printf "%s\\n" "$1" >&2; shift; exec "$@"',
        "ravana-launcher",
    ]
    assert argv[-3:] == [startup_marker, "python", "main.py"]
    assert build_docker_start_argv("a" * 64) == [
        "docker",
        "start",
        "--attach",
        "a" * 64,
    ]


def test_docker_argv_rejects_working_directory_outside_workspace(tmp_path):
    spec = SandboxSpec(
        image="python:3.11-slim",
        argv=["python", "main.py"],
        workspace=tmp_path,
        working_directory="/etc",
    )
    with pytest.raises(ValueError, match="must stay under /workspace"):
        build_docker_argv(spec, name="ravana-ci-abc", startup_marker="started")


# --- handler ----------------------------------------------------------------
def _handler(tmp_path, *, runtime="python3.11", runner=None, **config):
    return CodeInterpreterHandler({"runtime": runtime, **config}, runs_dir=tmp_path, runner=runner or FakeRunner())


def test_hand_off_run_on_a_closed_handler_is_refused(tmp_path):
    # §10.1 handoff shells out to git under the run dir, so it goes through the
    # same lifecycle gate as prepare_run/call rather than running against a
    # handler whose resources are already torn down.
    handler = _handler(tmp_path)
    (tmp_path / "r" / "workspace").mkdir(parents=True)

    async def close_then_hand_off():
        await handler.aclose()
        return await handler.hand_off_run("r")

    with pytest.raises(ToolkitError, match="closed"):
        asyncio.run(close_then_hand_off())


def _capture_toolkit_outcome(call, *args):
    try:
        return call(*args)
    except ToolkitError as exc:
        return exc


async def _wait_for_pid_file(
    pid_file,
    task: asyncio.Task,
    *,
    timeout_seconds: float = 0.5,
) -> int:
    async def wait_until_ready() -> int:
        while True:
            try:
                worker_pid = int(pid_file.read_text())
            except (FileNotFoundError, OSError, ValueError):
                worker_pid = 0
            if worker_pid > 0:
                return worker_pid
            if task.done():
                await task
                raise AssertionError("task completed before its worker became ready")
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(
        wait_until_ready(), timeout=timeout_seconds
    )


async def _await_task_with_hard_timeout(
    task: asyncio.Task,
    *,
    timeout_seconds: float,
    emergency_pid: int | None = None,
):
    done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    if task not in done:
        if emergency_pid is not None:
            try:
                os.kill(emergency_pid, 9)
            except ProcessLookupError:
                pass
        task.cancel()
        await asyncio.wait({task}, timeout=0.2)
        raise AssertionError(
            f"task did not finish within {timeout_seconds} seconds"
        )
    return await task


def _run_async_scenario_in_killable_process(
    scenario,
    *,
    timeout_seconds: float = 3.0,
) -> None:
    def force_stop() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join(1)

    def run() -> None:
        os.setsid()
        asyncio.run(scenario())

    process = multiprocessing.get_context("fork").Process(target=run)
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        force_stop()
        assert not process.is_alive()
        raise AssertionError(
            f"async scenario exceeded {timeout_seconds} seconds"
        )
    if process.exitcode != 0:
        force_stop()
        raise AssertionError(
            f"async scenario process exited {process.exitcode}"
        )


def test_writes_code_into_the_run_workspace_and_runs_it(tmp_path):
    runner = FakeRunner(SandboxResult(exit_code=0, stdout="42\n", stderr=""))
    handler = _handler(tmp_path, runner=runner)
    out = asyncio.run(handler.call(arguments={"code": "print(42)"}, idempotency_key="k", run_id="run-1"))
    workspace = (tmp_path / "run-1" / "workspace").resolve()
    assert (workspace / "main.py").read_text() == "print(42)"
    assert runner.spec.workspace == workspace  # scoped to THIS run
    assert runner.spec.argv == ["python", "main.py"]
    assert runner.spec.image == "python:3.11-slim"
    assert "exit_code: 0" in out and "42" in out


def test_monorepo_project_runs_from_its_subdirectory(tmp_path):
    repo = tmp_path / "repo"
    project = repo / "packages" / "app"
    project.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (project / "app.txt").write_text("app")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"],
        check=True,
    )
    runs = project / ".ravana" / "runs"
    runner = FakeRunner()
    handler = CodeInterpreterHandler(
        {"runtime": "python3.11"},
        runs_dir=runs,
        runner=runner,
    )

    async def prepare_call_and_close() -> None:
        try:
            await handler.prepare_run("r")
            await handler.call(
                arguments={"code": "print('app')"},
                idempotency_key="k",
                run_id="r",
            )
        finally:
            await handler.aclose()

    asyncio.run(prepare_call_and_close())
    workspace = runs / "r" / "workspace"
    assert runner.spec is not None
    assert runner.spec.workspace == workspace
    assert runner.spec.working_directory == "/workspace/packages/app"
    assert (workspace / "packages" / "app" / "main.py").exists()


def test_monorepo_project_symlink_created_by_agent_is_rejected_on_next_call(tmp_path):
    repo = tmp_path / "repo"
    project = repo / "packages" / "app"
    project.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (project / "app.txt").write_text("app")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"],
        check=True,
    )
    runs = project / ".ravana" / "runs"
    handler = CodeInterpreterHandler(
        {"runtime": "python3.11"},
        runs_dir=runs,
        runner=FakeRunner(),
    )

    async def scenario() -> None:
        try:
            await handler.prepare_run("r")
            cloned_project = runs / "r" / "workspace" / "packages" / "app"
            for child in cloned_project.iterdir():
                child.unlink()
            cloned_project.rmdir()
            cloned_project.symlink_to(project, target_is_directory=True)

            with pytest.raises(ToolkitError, match="project path"):
                await handler.call(
                    arguments={"code": "print('must not publish')"},
                    idempotency_key="symlink-after-prepare",
                    run_id="r",
                )
        finally:
            await handler.aclose()

    asyncio.run(scenario())
    assert not (project / "main.py").exists()


def test_node_runtime_uses_node_image_and_default_file(tmp_path):
    runner = FakeRunner()
    handler = _handler(tmp_path, runtime="node20", runner=runner)
    asyncio.run(handler.call(arguments={"code": "console.log(1)", "args": ["x"]}, idempotency_key="k", run_id="r"))
    assert runner.spec.image == "node:20-slim"
    assert runner.spec.argv == ["node", "main.js", "x"]


def test_unsupported_runtime_rejected_at_construction(tmp_path):
    with pytest.raises(ToolkitError, match="unsupported runtime"):
        CodeInterpreterHandler({"runtime": "ruby"}, runs_dir=tmp_path)


@pytest.mark.parametrize("sandbox", ["e2b", "none", "dockre", None, True, []])
def test_unsupported_sandbox_backend_is_rejected(tmp_path, sandbox):
    with pytest.raises(ToolkitError, match="sandbox.*docker"):
        CodeInterpreterHandler(
            {"runtime": "python3.11", "sandbox": sandbox}, runs_dir=tmp_path
        )


def test_podman_backend_selects_podman_executable(tmp_path, monkeypatch):
    selected = {}

    def runner_factory(*, docker):
        selected["executable"] = docker
        return FakeRunner()

    monkeypatch.setattr(
        code_interpreter_module, "DockerSandboxRunner", runner_factory
    )

    CodeInterpreterHandler(
        {"runtime": "python3.11", "sandbox": "podman"}, runs_dir=tmp_path
    )

    assert selected == {"executable": "podman"}


@pytest.mark.parametrize("bad", ["../escape.py", "/etc/passwd", "sub/dir.py", "..", "a\\b.py"])
def test_filename_must_be_a_bare_name(tmp_path, bad):
    handler = _handler(tmp_path)
    with pytest.raises(ToolkitError, match="bare name"):
        asyncio.run(handler.call(arguments={"code": "x", "filename": bad}, idempotency_key="k", run_id="r"))


def test_custom_bare_filename_is_allowed(tmp_path):
    runner = FakeRunner()
    handler = _handler(tmp_path, runner=runner)
    asyncio.run(handler.call(arguments={"code": "x", "filename": "solve.py"}, idempotency_key="k", run_id="r"))
    assert (tmp_path / "r" / "workspace" / "solve.py").exists()
    assert runner.spec.argv == ["python", "solve.py"]


def test_script_write_replaces_symlink_without_touching_its_host_target(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("HOST DATA")
    workspace = tmp_path / "r" / "workspace"
    workspace.mkdir(parents=True)
    script = workspace / "main.py"
    script.symlink_to(outside)

    asyncio.run(
        _handler(tmp_path).call(
            arguments={"code": "print('sandbox')"}, idempotency_key="k", run_id="r"
        )
    )

    assert outside.read_text() == "HOST DATA"
    assert not script.is_symlink()
    assert script.read_text() == "print('sandbox')"


def test_workspace_symlink_cannot_alias_another_run(tmp_path):
    other_workspace = tmp_path / "other" / "workspace"
    other_workspace.mkdir(parents=True)
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "workspace").symlink_to(other_workspace, target_is_directory=True)

    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(
            _handler(tmp_path).call(
                arguments={"code": "print(1)"}, idempotency_key="k", run_id="r"
            )
        )

    assert exc_info.value.kind is ToolFailureKind.FATAL
    assert not (other_workspace / "main.py").exists()


@pytest.mark.parametrize("network", [True, "false", 1, ["example.com"]])
def test_network_egress_config_is_rejected_until_a_host_allowlist_exists(tmp_path, network):
    with pytest.raises(ToolkitError, match="network|allow-list"):
        _handler(tmp_path, network=network)


def test_limits_are_clamped_to_the_ceiling(tmp_path):
    runner = FakeRunner()
    handler = _handler(
        tmp_path,
        runner=runner,
        memory_mb=99999,
        cpus=64,
        timeout_seconds=99999,
        workspace_mb=99999,
        workspace_files=999999,
    )
    asyncio.run(handler.call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert runner.spec.limits.memory_mb == 8192  # ceiling
    assert runner.spec.limits.cpus == 8.0
    assert runner.spec.limits.timeout_seconds == 300
    assert runner.spec.limits.workspace_bytes == 8192 * 1024 * 1024
    assert runner.spec.limits.workspace_files == 100_000


def test_script_over_workspace_quota_is_rejected_before_host_write(tmp_path):
    runner = FakeRunner()
    handler = _handler(tmp_path, runner=runner, workspace_mb=16)

    with pytest.raises(ToolkitError, match="workspace byte limit"):
        asyncio.run(
            handler.call(
                arguments={"code": "x" * (16 * 1024 * 1024 + 1)},
                idempotency_key="k",
                run_id="r",
            )
        )

    workspace = tmp_path / "r" / "workspace"
    assert not (workspace / "main.py").exists()
    assert list(workspace.iterdir()) == []
    assert runner.spec is None


def test_cancellation_during_script_staging_is_bounded_and_never_publishes(
    tmp_path, monkeypatch
):
    pid_file = tmp_path / "script-stage.pid"

    def stalled_stage_worker_argv(temp_path):
        return [
            sys.executable,
            "-c",
            (
                "import os, pathlib, time, sys; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(60)"
            ),
            str(pid_file),
        ]

    monkeypatch.setattr(
        sandbox_module, "_script_stage_worker_argv", stalled_stage_worker_argv
    )

    async def scenario() -> None:
        workspace = tmp_path / "r" / "workspace"
        workspace.mkdir(parents=True)
        destination = workspace / "main.py"
        destination.write_text("old")
        runner = FakeRunner()
        task = asyncio.create_task(
            _handler(tmp_path, runner=runner).call(
                arguments={"code": "x" * (4 * 1024 * 1024)},
                idempotency_key="cancelled-stage",
                run_id="r",
            )
        )
        worker_pid = await _wait_for_pid_file(pid_file, task)

        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.1)

        task.cancel()
        with pytest.raises(ToolRetrySafeCancellation):
            await _await_task_with_hard_timeout(
                task,
                timeout_seconds=0.5,
                emergency_pid=worker_pid,
            )

        assert runner.spec is None
        assert destination.read_text() == "old"
        assert list(workspace.glob(".ravana-script-*")) == []
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)

    _run_async_scenario_in_killable_process(scenario)


def test_cancelled_late_stage_spawn_cannot_mutate_workspace_after_unlock(
    tmp_path, monkeypatch
):
    async def capacity_available(*args, **kwargs):
        return None

    real_spawn = sandbox_module._create_workspace_subprocess_exec

    async def scenario() -> None:
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()

        async def delayed_spawn(*args, **kwargs):
            spawn_started.set()
            await release_spawn.wait()
            return await real_spawn(*args, **kwargs)

        monkeypatch.setattr(
            code_interpreter_module,
            "workspace_capacity_violation_async",
            capacity_available,
        )
        monkeypatch.setattr(
            sandbox_module,
            "_create_workspace_subprocess_exec",
            delayed_spawn,
        )
        workspace = tmp_path / "r" / "workspace"
        workspace.mkdir(parents=True)
        destination = workspace / "main.py"
        destination.write_text("old")
        handler = _handler(tmp_path, runner=FakeRunner())
        task = asyncio.create_task(
            handler.call(
                arguments={"code": "print('new')"},
                idempotency_key="late-stage",
                run_id="r",
            )
        )
        await asyncio.wait_for(spawn_started.wait(), timeout=0.2)

        try:
            task.cancel()
            with pytest.raises(ToolRetrySafeCancellation):
                await _await_task_with_hard_timeout(
                    task, timeout_seconds=0.5
                )

            assert destination.read_text() == "old"
            assert list(workspace.glob(".ravana-script-*")) == []
        finally:
            release_spawn.set()
            await handler.aclose()

        assert destination.read_text() == "old"
        assert list(workspace.glob(".ravana-script-*")) == []

    _run_async_scenario_in_killable_process(scenario)


def test_script_cleanup_failure_is_outcome_unknown(tmp_path, monkeypatch):
    runner = FakeRunner()
    handler = _handler(tmp_path, runner=runner)

    def fail_replace(*args, **kwargs):
        raise OSError("forced replace failure")

    def fail_cleanup(temp_path):
        raise sandbox_module.WorkspaceStagingCleanupError(
            "forced cleanup failure"
        )

    monkeypatch.setattr(code_interpreter_module.os, "replace", fail_replace)
    monkeypatch.setattr(
        code_interpreter_module,
        "cleanup_workspace_staged_file",
        fail_cleanup,
    )

    with pytest.raises(ToolOutcomeUnknown, match="cleanup failed"):
        asyncio.run(
            handler.call(
                arguments={"code": "print(1)"},
                idempotency_key="cleanup-failure",
                run_id="r",
            )
        )

    workspace = tmp_path / "r" / "workspace"
    assert runner.spec is None
    staged_files = list(workspace.glob(".ravana-script-*"))
    assert len(staged_files) == 1
    staged_files[0].unlink()


def test_unexpected_stage_spawn_failure_closes_fd_and_removes_temp(
    tmp_path, monkeypatch
):
    async def capacity_available(*args, **kwargs):
        return None

    async def fail_spawn(*args, **kwargs):
        raise RuntimeError("forced spawn failure")

    monkeypatch.setattr(
        code_interpreter_module,
        "workspace_capacity_violation_async",
        capacity_available,
    )
    monkeypatch.setattr(
        sandbox_module,
        "_create_workspace_subprocess_exec",
        fail_spawn,
    )
    runner = FakeRunner()

    with pytest.raises(ToolkitError, match="failed to start"):
        asyncio.run(
            _handler(tmp_path, runner=runner).call(
                arguments={"code": "print(1)"},
                idempotency_key="unexpected-spawn",
                run_id="r",
            )
        )

    workspace = tmp_path / "r" / "workspace"
    assert runner.spec is None
    assert list(workspace.glob(".ravana-script-*")) == []


def test_stage_fd_cleanup_failure_is_outcome_unknown(
    tmp_path, monkeypatch
):
    captured_fds: list[int] = []

    def fail_close(stage_fd):
        captured_fds.append(stage_fd)
        raise sandbox_module.WorkspaceStagingCleanupError(
            "forced descriptor cleanup failure"
        )

    monkeypatch.setattr(
        sandbox_module, "_close_workspace_stage_fd", fail_close
    )
    runner = FakeRunner()

    try:
        with pytest.raises(ToolOutcomeUnknown, match="cleanup failed"):
            asyncio.run(
                _handler(tmp_path, runner=runner).call(
                    arguments={"code": "print(1)"},
                    idempotency_key="fd-cleanup-failure",
                    run_id="r",
                )
            )
    finally:
        for stage_fd in captured_fds:
            try:
                os.close(stage_fd)
            except OSError:
                pass

    workspace = tmp_path / "r" / "workspace"
    assert runner.spec is None
    assert list(workspace.glob(".ravana-script-*")) == []


def test_parallel_script_publication_cannot_overcommit_workspace(
    tmp_path, monkeypatch
):
    real_capacity_check = sandbox_module.workspace_capacity_violation
    before_scan = threading.Barrier(2)
    after_scan = threading.Barrier(2)

    def concurrent_capacity_check_sync(*args, **kwargs):
        kwargs.pop("timeout_seconds")
        kwargs.pop("supervisor")
        for barrier in (before_scan,):
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
        result = real_capacity_check(*args, **kwargs)
        for barrier in (after_scan,):
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
        return result

    async def concurrent_capacity_check(*args, **kwargs):
        return await asyncio.to_thread(
            concurrent_capacity_check_sync, *args, **kwargs
        )

    monkeypatch.setattr(
        code_interpreter_module,
        "workspace_capacity_violation_async",
        concurrent_capacity_check,
    )

    def publish(filename: str) -> str:
        return asyncio.run(
            _handler(tmp_path, workspace_mb=16).call(
                arguments={"code": "x" * (10 * 1024 * 1024), "filename": filename},
                idempotency_key=filename,
                run_id="r",
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda filename: _capture_toolkit_outcome(publish, filename),
                ("first.py", "second.py"),
            )
        )

    workspace = tmp_path / "r" / "workspace"
    published_bytes = sum(path.stat().st_size for path in workspace.iterdir())
    assert sum(isinstance(outcome, ToolkitError) for outcome in outcomes) == 1
    assert published_bytes <= 16 * 1024 * 1024


def test_parallel_calls_cannot_overwrite_the_active_default_script(tmp_path):
    class SequencedRunner:
        def __init__(self) -> None:
            self.calls = 0
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def run(self, spec: SandboxSpec) -> SandboxResult:
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                await self.release_first.wait()
            code = (spec.workspace / "main.py").read_text()
            return SandboxResult(exit_code=0, stdout=code, stderr="")

    async def scenario() -> None:
        runner = SequencedRunner()
        handler = _handler(tmp_path, runner=runner)
        first = asyncio.create_task(
            handler.call(
                arguments={"code": "print('first')"},
                idempotency_key="first",
                run_id="r",
            )
        )
        await runner.first_started.wait()
        second = asyncio.create_task(
            handler.call(
                arguments={"code": "print('second')"},
                idempotency_key="second",
                run_id="r",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        workspace = tmp_path / "r" / "workspace"
        assert (workspace / "main.py").read_text() == "print('first')"

        runner.release_first.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert "print('first')" in first_result
        assert "print('second')" in second_result

    asyncio.run(scenario())


def test_workspace_lock_wait_is_cancellable_without_blocking_event_loop(
    tmp_path
):
    workspace = tmp_path / "r" / "workspace"
    workspace.mkdir(parents=True)
    lock_path = workspace.parent / ".ravana-workspace.lock"
    lock_held = threading.Event()

    def hold_workspace_lock() -> None:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            lock_held.set()
            time.sleep(0.2)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    holder = threading.Thread(target=hold_workspace_lock)
    holder.start()
    assert lock_held.wait(timeout=0.2)

    async def scenario() -> None:
        runner = FakeRunner()
        task = asyncio.create_task(
            _handler(tmp_path, runner=runner).call(
                arguments={"code": "print(1)"},
                idempotency_key="waiting-lock",
                run_id="r",
            )
        )
        started = time.monotonic()
        await asyncio.sleep(0.02)
        assert time.monotonic() - started < 0.1

        task.cancel()
        with pytest.raises(ToolRetrySafeCancellation):
            await task
        assert runner.spec is None
        assert not (workspace / "main.py").exists()

    try:
        asyncio.run(scenario())
    finally:
        holder.join(timeout=0.5)
    assert not holder.is_alive()


def test_cancellation_during_workspace_scan_is_bounded_and_does_not_publish(
    tmp_path, monkeypatch
):
    pid_file = tmp_path / "workspace-scan.pid"

    def stalled_worker_argv(*args, **kwargs):
        return [
            sys.executable,
            "-c",
            (
                "import os, pathlib, time, sys; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(60)"
            ),
            str(pid_file),
        ]

    monkeypatch.setattr(
        sandbox_module, "_workspace_worker_argv", stalled_worker_argv
    )

    async def scenario() -> None:
        runner = FakeRunner()
        task = asyncio.create_task(
            _handler(tmp_path, runner=runner).call(
                arguments={"code": "print(1)"},
                idempotency_key="cancelled-scan",
                run_id="r",
            )
        )
        worker_pid = await _wait_for_pid_file(pid_file, task)

        task.cancel()
        with pytest.raises(ToolRetrySafeCancellation):
            await _await_task_with_hard_timeout(
                task,
                timeout_seconds=0.5,
                emergency_pid=worker_pid,
            )

        assert runner.spec is None
        assert not (tmp_path / "r" / "workspace" / "main.py").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)

    _run_async_scenario_in_killable_process(scenario)


def test_default_limits(tmp_path):
    runner = FakeRunner()
    asyncio.run(_handler(tmp_path, runner=runner).call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert runner.spec.limits == SandboxLimits(memory_mb=2048, cpus=2.0, timeout_seconds=60)


def test_is_side_effecting_true(tmp_path):
    assert _handler(tmp_path).is_side_effecting({"code": "x"}) is True


def test_sandbox_infrastructure_error_is_transient(tmp_path):
    handler = _handler(tmp_path, runner=FakeRunner(raise_error=SandboxError("docker not found")))
    with pytest.raises(ToolkitError) as ei:
        asyncio.run(handler.call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert ei.value.kind is ToolFailureKind.TRANSIENT  # §3.6 sandbox cold-start → engine retries


def test_indeterminate_sandbox_outcome_is_fatal_and_not_retryable(tmp_path):
    handler = _handler(
        tmp_path,
        runner=FakeRunner(raise_error=SandboxOutcomeUnknown("cleanup failed")),
    )
    with pytest.raises(ToolOutcomeUnknown) as exc_info:
        asyncio.run(
            handler.call(
                arguments={"code": "x"}, idempotency_key="k", run_id="r"
            )
        )
    assert exc_info.value.kind is ToolFailureKind.FATAL


def test_no_runs_dir_configured_fails_fatally_only_when_called(tmp_path):
    # Building the handler without a runs dir is fine (the gateway is built
    # before a run exists); calling it without one is a fatal misconfig.
    handler = CodeInterpreterHandler({"runtime": "python3.11"}, runs_dir=None, runner=FakeRunner())
    with pytest.raises(ToolkitError) as ei:
        asyncio.run(handler.call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert ei.value.kind is ToolFailureKind.FATAL


def test_executor_threads_run_id_and_dedups_side_effect(con, tmp_path):
    # End-to-end through RavanaToolExecutor: run_id reaches the handler (workspace
    # scoped to it), and because code execution is side-effecting, a retried
    # logical invocation (same idempotency key) is deduped, not re-run.
    from ravana.runtime.tool_executor import RavanaToolExecutor
    from tests.test_tool_execution import _seed_run

    _seed_run(con, run_id="run-9")  # tool_invocation.run_id FKs to run(id)
    runner = FakeRunner(SandboxResult(exit_code=0, stdout="once", stderr=""))
    executor = RavanaToolExecutor(con, {"ci": _handler(tmp_path, runner=runner)})
    out1 = asyncio.run(
        executor.execute(run_id="run-9", node_id="n", tool="ci", arguments={"code": "print(1)"}, idempotency_key="key-1")
    )
    assert (tmp_path / "run-9" / "workspace" / "main.py").exists()  # scoped to run-9
    assert "once" in out1

    runner.spec = None  # a re-run would set this again
    out2 = asyncio.run(
        executor.execute(run_id="run-9", node_id="n", tool="ci", arguments={"code": "print(1)"}, idempotency_key="key-1")
    )
    assert out2 == out1
    assert runner.spec is None  # deduped — the sandbox was NOT invoked a second time


def test_prestart_cancellation_marks_invocation_retryable(con, tmp_path, monkeypatch):
    from ravana.runtime.tool_executor import RavanaToolExecutor
    from tests.test_tool_execution import _seed_run

    async def scenario() -> None:
        _seed_run(con, run_id="run-cancel")
        pid_file = tmp_path / "prestart-scan.pid"

        async def capacity_available(*args, **kwargs):
            return None

        def stalled_worker_argv(*args, **kwargs):
            return [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, time, sys; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(60)"
                ),
                str(pid_file),
            ]

        monkeypatch.setattr(
            code_interpreter_module,
            "workspace_capacity_violation_async",
            capacity_available,
        )
        monkeypatch.setattr(
            sandbox_module, "_workspace_worker_argv", stalled_worker_argv
        )
        executor = RavanaToolExecutor(
            con,
            {
                "ci": _handler(
                    tmp_path,
                    runner=DockerSandboxRunner(docker=sys.executable),
                )
            },
        )
        task = asyncio.create_task(
            executor.execute(
                run_id="run-cancel",
                node_id="n",
                tool="ci",
                arguments={"code": "print(1)"},
                idempotency_key="cancel-key",
            )
        )
        worker_pid = await _wait_for_pid_file(pid_file, task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _await_task_with_hard_timeout(
                task,
                timeout_seconds=0.5,
                emergency_pid=worker_pid,
            )
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)

        row = con.execute(
            "SELECT status FROM tool_invocation WHERE idempotency_key = ?",
            ("cancel-key",),
        ).fetchone()
        assert row["status"] == "FAILED"

        retry_runner = FakeRunner(
            SandboxResult(exit_code=0, stdout="retried", stderr="")
        )
        retry_executor = RavanaToolExecutor(
            con, {"ci": _handler(tmp_path, runner=retry_runner)}
        )
        result = await retry_executor.execute(
            run_id="run-cancel",
            node_id="n",
            tool="ci",
            arguments={"code": "print(1)"},
            idempotency_key="cancel-key",
        )

        assert "retried" in result
        assert retry_runner.spec is not None

    _run_async_scenario_in_killable_process(scenario)


def test_nonzero_exit_and_timeout_are_results_not_exceptions(tmp_path):
    nonzero = FakeRunner(SandboxResult(exit_code=1, stdout="", stderr="boom"))
    out = asyncio.run(_handler(tmp_path, runner=nonzero).call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert "exit_code: 1" in out and "boom" in out

    timed = FakeRunner(SandboxResult(exit_code=124, stdout="", stderr="", timed_out=True))
    out = asyncio.run(_handler(tmp_path, runner=timed).call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert "timed out" in out


def test_runner_owns_output_truncation_metadata(tmp_path):
    retained = "x" * 10_000
    runner = FakeRunner(
        SandboxResult(
            exit_code=0,
            stdout=retained + "\n... [truncated, 190000 more bytes]",
            stderr="",
        )
    )

    out = asyncio.run(
        _handler(tmp_path, runner=runner).call(
            arguments={"code": "x"}, idempotency_key="k", run_id="r"
        )
    )

    assert "190000 more bytes" in out


# --- Docker runner process boundary -----------------------------------------
def _fake_docker(tmp_path, body: str):
    executable = tmp_path / "fake-docker"
    create_argv = tmp_path / "fake-docker-create-argv.json"
    container_id = "a" * 64
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"state = pathlib.Path({str(create_argv)!r})\n"
        "if sys.argv[1] == 'create':\n"
        "    state.write_text(json.dumps(sys.argv[1:]))\n"
        f"    sys.stdout.write({container_id!r} + '\\n')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1] == 'rm': raise SystemExit(0)\n"
        "if sys.argv[1] == 'container':\n"
        "    sys.stderr.write('Error: No such object\\n')\n"
        "    raise SystemExit(1)\n"
        "if sys.argv[1] == 'start':\n"
        "    create_args = json.loads(state.read_text())\n"
        "    sys.argv = [sys.argv[0], 'run', *create_args[1:]]\n"
        f"{body}"
    )
    executable.chmod(0o700)
    return executable


def _fake_shell_docker(tmp_path, body: str):
    executable = tmp_path / "fake-docker"
    executable.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"create\" ]; then\n"
        f"  echo {'a' * 64}\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"start\" ]; then set -- run; fi\n"
        f"{body}"
    )
    executable.chmod(0o700)
    return executable


class _ControlledProcess:
    def __init__(
        self,
        *,
        exit_code=None,
        stdout="",
        stderr="",
        terminate_completes=True,
        kill_completes=True,
        on_terminate=None,
        on_kill=None,
    ):
        self.stdout = asyncio.StreamReader()
        if stdout:
            self.stdout.feed_data(stdout.encode())
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        if stderr:
            self.stderr.feed_data(stderr.encode())
        self.stderr.feed_eof()
        self.returncode = exit_code
        self.terminate_completes = terminate_completes
        self.kill_completes = kill_completes
        self.on_terminate = on_terminate
        self.on_kill = on_kill
        self.terminate_called = False
        self.kill_called = False
        self._done = asyncio.Event()
        if exit_code is not None:
            self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        if self.on_terminate is not None:
            self.on_terminate()
        if self.terminate_completes:
            self.returncode = -15
            self._done.set()

    def kill(self):
        self.kill_called = True
        if self.on_kill is not None:
            self.on_kill()
        if self.kill_completes:
            self.returncode = -9
            self._done.set()


_FAKE_CONTAINER_ID = "a" * 64


def _created_container_process() -> _ControlledProcess:
    return _ControlledProcess(
        exit_code=0, stdout=f"{_FAKE_CONTAINER_ID}\n"
    )


def _feed_startup_marker(argv, process) -> None:
    marker = argv[argv.index("ravana-launcher") + 1]
    process.stderr = asyncio.StreamReader()
    process.stderr.feed_data(f"{marker}\n".encode())
    process.stderr.feed_eof()


def test_runner_creates_container_before_starting_agent_code(tmp_path, monkeypatch):
    async def scenario() -> None:
        commands: list[str] = []
        container_id = "a" * 64
        run_process = _ControlledProcess(exit_code=0)

        async def fake_create_subprocess_exec(*argv, **kwargs):
            command = argv[1]
            commands.append(command)
            if command in {"create", "run"}:
                marker = argv[argv.index("ravana-launcher") + 1]
                run_process.stderr = asyncio.StreamReader()
                run_process.stderr.feed_data(f"{marker}\n".encode())
                run_process.stderr.feed_eof()
            if command == "create":
                return _ControlledProcess(
                    exit_code=0, stdout=f"{container_id}\n"
                )
            if command in {"start", "run"}:
                return run_process
            if command == "container":
                return _ControlledProcess(
                    exit_code=1, stderr="Error: No such object"
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        result = await DockerSandboxRunner(docker=sys.executable).run(
            _spec(tmp_path)
        )

        assert result.exit_code == 0
        assert commands[:2] == ["create", "start"]

    asyncio.run(scenario())


def test_runner_close_cannot_finish_while_an_admitted_run_continues(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        runner = DockerSandboxRunner(
            docker=sys.executable, cleanup_timeout_seconds=0.01
        )
        run_started = asyncio.Event()
        release_run = asyncio.Event()

        async def delayed_run(*args, **kwargs):
            run_started.set()
            await release_run.wait()
            return SandboxResult(exit_code=0, stdout="", stderr="")

        monkeypatch.setattr(runner, "_run_container", delayed_run)
        run_task = asyncio.create_task(runner.run(_spec(tmp_path)))
        await asyncio.wait_for(run_started.wait(), timeout=0.2)

        with pytest.raises(TimeoutError, match="did not finish"):
            await runner.aclose()
        assert not run_task.done()

        release_run.set()
        result = await asyncio.wait_for(run_task, timeout=0.2)
        assert result.exit_code == 0
        await runner.aclose()

        with pytest.raises(SandboxError, match="closed"):
            await runner.run(_spec(tmp_path))

    asyncio.run(scenario())


def test_cancelling_container_creation_never_starts_agent_code(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        create_started = asyncio.Event()
        create_stopped = asyncio.Event()
        start_called = asyncio.Event()
        create_process = _ControlledProcess(
            stdout=f"{_FAKE_CONTAINER_ID}\n",
            on_terminate=create_stopped.set,
            on_kill=create_stopped.set,
        )

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                create_started.set()
                return create_process
            if argv[1] == "start":
                start_called.set()
                raise AssertionError("agent code must not start after cancellation")
            if argv[1] == "rm":
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        task = asyncio.create_task(
            DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.2
            ).run(_spec(tmp_path))
        )
        await create_started.wait()
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(SandboxCancelledBeforeStart):
            await task
        assert create_stopped.is_set()
        assert not start_called.is_set()

    asyncio.run(scenario())


def test_docker_exit_125_is_sandbox_infrastructure_failure(tmp_path):
    docker = _fake_docker(
        tmp_path,
        "import sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'rm': raise SystemExit(0)\n"
        "if args[0] == 'container':\n"
        "    sys.stderr.write('Error: No such object\\n')\n"
        "    raise SystemExit(1)\n"
        "sys.stderr.write('daemon unavailable\\n')\n"
        "raise SystemExit(125)\n",
    )

    with pytest.raises(SandboxError, match="exit 125|daemon unavailable"):
        asyncio.run(DockerSandboxRunner(docker=str(docker)).run(_spec(tmp_path)))


def test_any_terminal_exit_without_startup_marker_is_not_an_agent_result(tmp_path):
    docker = _fake_docker(
        tmp_path,
        "import sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'rm': raise SystemExit(0)\n"
        "if args[0] == 'container':\n"
        "    sys.stderr.write('Error: No such object\\n')\n"
        "    raise SystemExit(1)\n"
        "sys.stderr.write('command could not start\\n')\n"
        "raise SystemExit(127)\n",
    )

    with pytest.raises(SandboxError, match="before.*start|exit 127"):
        asyncio.run(
            DockerSandboxRunner(
                docker=str(docker), cleanup_timeout_seconds=0.2
            ).run(_spec(tmp_path))
        )


def test_agent_exit_125_is_a_result_and_dedupes_after_workspace_mutation(
    con, tmp_path
):
    from ravana.runtime.tool_executor import RavanaToolExecutor
    from tests.test_tool_execution import _seed_run

    marker = tmp_path / "effects.txt"
    docker = _fake_docker(
        tmp_path,
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "marker = args[args.index('ravana-launcher') + 1]\n"
        "sys.stderr.write('pull-output-' * 2000)\n"
        "sys.stderr.write(marker + '\\n')\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "marker.write_text(marker.read_text() + 'x' if marker.exists() else 'x')\n"
        "raise SystemExit(125)\n",
    )
    _seed_run(con, run_id="run-125")
    handler = _handler(
        tmp_path,
        runner=DockerSandboxRunner(docker=str(docker)),
    )
    executor = RavanaToolExecutor(con, {"ci": handler})

    first = asyncio.run(
        executor.execute(
            run_id="run-125",
            node_id="n",
            tool="ci",
            arguments={"code": "raise SystemExit(125)"},
            idempotency_key="key-125",
        )
    )
    second = asyncio.run(
        executor.execute(
            run_id="run-125",
            node_id="n",
            tool="ci",
            arguments={"code": "raise SystemExit(125)"},
            idempotency_key="key-125",
        )
    )

    assert first == second
    assert "exit_code: 125" in first
    assert "ravana-started-" not in first
    assert marker.read_text() == "x"
    row = con.execute(
        "SELECT status FROM tool_invocation WHERE idempotency_key = 'key-125'"
    ).fetchone()
    assert row["status"] == "SUCCEEDED"


def test_attached_client_exit_with_live_container_is_outcome_unknown(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        run_process = _ControlledProcess(exit_code=0)
        container_exists = True
        cleanup_targets: list[str] = []

        async def fake_create_subprocess_exec(*argv, **kwargs):
            nonlocal container_exists
            if argv[1] == "create":
                _feed_startup_marker(argv, run_process)
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            if argv[1] == "rm":
                cleanup_targets.append(argv[-1])
                container_exists = False
                return _ControlledProcess(exit_code=0)
            if container_exists:
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        with pytest.raises(SandboxOutcomeUnknown, match="container|attached"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.2
            ).run(_spec(tmp_path))

        assert cleanup_targets == [_FAKE_CONTAINER_ID]

    asyncio.run(scenario())


def test_cancellation_during_terminal_output_drain_removes_container(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        run_process = _ControlledProcess(exit_code=0)
        run_process.stderr = asyncio.StreamReader()
        cleanup_targets: list[str] = []
        start_returned = asyncio.Event()

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                marker = argv[argv.index("ravana-launcher") + 1]
                run_process.stderr.feed_data(f"{marker}\n".encode())
                return _created_container_process()
            if argv[1] == "start":
                start_returned.set()
                return run_process
            if argv[1] == "rm":
                cleanup_targets.append(argv[-1])
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        task = asyncio.create_task(
            DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.2
            ).run(_spec(tmp_path))
        )
        await start_returned.wait()
        await asyncio.sleep(0.02)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup_targets == [_FAKE_CONTAINER_ID]

    asyncio.run(scenario())


def test_cancellation_during_workspace_monitor_shutdown_is_not_swallowed(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        run_process = _ControlledProcess(exit_code=0)
        runner = DockerSandboxRunner(
            docker=sys.executable, cleanup_timeout_seconds=0.2
        )
        monitor_stopping = asyncio.Event()
        allow_monitor_stop = asyncio.Event()

        async def stalled_monitor(spec):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                monitor_stopping.set()
                await allow_monitor_stop.wait()
                raise

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                marker = argv[argv.index("ravana-launcher") + 1]
                run_process.stderr = asyncio.StreamReader()
                run_process.stderr.feed_data(f"{marker}\n".encode())
                run_process.stderr.feed_eof()
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(runner, "_watch_workspace", stalled_monitor)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        task = asyncio.create_task(runner.run(_spec(tmp_path)))
        await monitor_stopping.wait()
        task.cancel()
        allow_monitor_stop.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "process_exited",
    [True, False],
    ids=["process-exit", "run-timeout"],
)
def test_workspace_violation_winning_during_monitor_stop_is_not_discarded(
    tmp_path, monkeypatch, process_exited
):
    async def scenario() -> None:
        startup_marker = "ravana-started-monitor-race"
        run_process = _ControlledProcess(
            exit_code=0 if process_exited else None,
            stderr=f"{startup_marker}\n",
        )
        runner = DockerSandboxRunner(
            docker=sys.executable, cleanup_timeout_seconds=0.2
        )
        release_monitor = asyncio.Event()
        real_stop_task = runner._stop_task

        async def monitor(spec):
            await release_monitor.wait()
            return sandbox_module.WorkspaceViolation(
                "workspace byte limit exceeded (2048 > 1024)"
            )

        async def stop_after_monitor_completes(task):
            release_monitor.set()
            await asyncio.sleep(0)
            return await real_stop_task(task)

        async def finish_without_external_cleanup(*args, **kwargs):
            return None, None, False

        async def verify_absent(*args, **kwargs):
            return None, False

        async def clean_final_scan(*args, **kwargs):
            return None

        monkeypatch.setattr(runner, "_watch_workspace", monitor)
        monkeypatch.setattr(runner, "_stop_task", stop_after_monitor_completes)
        monkeypatch.setattr(
            runner, "_finish_uninterruptibly", finish_without_external_cleanup
        )
        monkeypatch.setattr(
            runner, "_verify_attached_exit_uninterruptibly", verify_absent
        )
        monkeypatch.setattr(
            sandbox_module, "_workspace_violation_in_worker", clean_final_scan
        )

        result = await runner._run_spawned_process(
            _spec(
                tmp_path,
                limits=SandboxLimits(
                    workspace_bytes=1024, workspace_files=100
                ),
            ),
            run_process,
            _FAKE_CONTAINER_ID,
            startup_marker,
            asyncio.get_running_loop().time() + (
                1 if process_exited else 0
            ),
        )

        assert result.exit_code == 122
        assert "workspace byte limit exceeded" in result.stderr

    asyncio.run(scenario())


def test_initial_workspace_scan_is_deadline_bounded_before_container_creation(
    tmp_path, monkeypatch
):
    pid_file = tmp_path / "deadline-scan.pid"

    def stalled_worker_argv(*args, **kwargs):
        return [
            sys.executable,
            "-c",
            (
                "import os, pathlib, time, sys; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(60)"
            ),
            str(pid_file),
        ]

    monkeypatch.setattr(
        sandbox_module, "_workspace_worker_argv", stalled_worker_argv
    )
    monkeypatch.setattr(
        sandbox_module, "_MIN_WORKSPACE_SCAN_TIMEOUT_SECONDS", 0.01
    )

    async def scenario() -> None:
        runner = DockerSandboxRunner(docker=sys.executable)
        started = time.monotonic()
        with pytest.raises(SandboxError, match="measurement.*deadline"):
            await runner.run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.1))
            )

        assert pid_file.exists()
        worker_pid = int(pid_file.read_text())
        assert time.monotonic() - started < 0.5
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)

    asyncio.run(scenario())


def test_workspace_scan_communication_failure_kills_and_reaps_worker(
    tmp_path, monkeypatch
):
    class BrokenWorkerProcess:
        def __init__(self) -> None:
            self.stdout = object()
            self.returncode = None
            self.kill_called = False
            self.wait_called = False
            self._killed = asyncio.Event()

        async def communicate(self):
            raise RuntimeError("worker pipe failed")

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9
            self._killed.set()

        async def wait(self):
            self.wait_called = True
            await self._killed.wait()
            return self.returncode

    async def scenario() -> None:
        worker = BrokenWorkerProcess()
        supervisor = sandbox_module.WorkspaceWorkerSupervisor()

        async def spawn_worker(*args, **kwargs):
            return worker

        monkeypatch.setattr(
            sandbox_module,
            "_create_workspace_subprocess_exec",
            spawn_worker,
        )
        violation = await sandbox_module._workspace_violation_in_worker(
            tmp_path,
            max_bytes=1024,
            max_files=10,
            timeout_seconds=0.2,
            supervisor=supervisor,
        )

        assert violation is not None
        assert violation.measurement_failed is True
        assert "RuntimeError" in violation.message
        assert worker.kill_called is True
        assert worker.wait_called is True
        await supervisor.aclose(timeout_seconds=0.2)

    asyncio.run(scenario())


def test_delayed_workspace_worker_spawn_cleanup_is_tracked_and_reaped(
    monkeypatch,
):
    class LateWorker:
        def __init__(self) -> None:
            self.returncode = None
            self.kill_called = False
            self.reaped = asyncio.Event()

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9

        async def wait(self):
            self.reaped.set()
            return self.returncode

    async def scenario() -> None:
        release_spawn = asyncio.Event()
        worker = LateWorker()
        supervisor = sandbox_module.WorkspaceWorkerSupervisor()

        async def delayed_spawn():
            await release_spawn.wait()
            return worker

        spawn_task = asyncio.create_task(delayed_spawn())
        sandbox_module._schedule_workspace_worker_spawn_cleanup(
            spawn_task, supervisor=supervisor
        )

        assert supervisor.pending_count == 1
        with pytest.raises(
            sandbox_module.WorkspaceStagingCleanupError,
            match="did not finish",
        ):
            await supervisor.aclose(timeout_seconds=0.01)
        release_spawn.set()
        await asyncio.wait_for(worker.reaped.wait(), timeout=0.2)
        await supervisor.aclose(timeout_seconds=0.2)

        assert worker.kill_called is True
        assert supervisor.pending_count == 0

    asyncio.run(scenario())


def test_workspace_worker_wait_continues_after_foreground_cleanup_deadline(
    monkeypatch,
):
    class SlowReapWorker:
        def __init__(self) -> None:
            self.returncode = None
            self.kill_called = False
            self.wait_started = asyncio.Event()
            self.release_wait = asyncio.Event()
            self.reaped = asyncio.Event()

        def kill(self) -> None:
            self.kill_called = True

        async def wait(self):
            self.wait_started.set()
            await self.release_wait.wait()
            self.returncode = -9
            self.reaped.set()
            return self.returncode

    async def scenario() -> None:
        worker = SlowReapWorker()
        supervisor = sandbox_module.WorkspaceWorkerSupervisor()
        monkeypatch.setattr(
            sandbox_module, "_WORKSPACE_WORKER_CLEANUP_SECONDS", 0.01
        )

        await sandbox_module._terminate_workspace_worker_uninterruptibly(
            worker, supervisor=supervisor
        )

        assert worker.kill_called is True
        assert worker.wait_started.is_set()
        assert supervisor.pending_count == 1

        worker.release_wait.set()
        await asyncio.wait_for(worker.reaped.wait(), timeout=0.2)
        await supervisor.aclose(timeout_seconds=0.2)

        assert supervisor.pending_count == 0

    asyncio.run(scenario())


def test_workspace_worker_background_cleanup_error_surfaces_on_close(
    tmp_path, monkeypatch
):
    class FinishedWorker:
        def __init__(self) -> None:
            self.returncode = None
            self.reaped = asyncio.Event()

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self):
            self.reaped.set()
            return self.returncode

    async def scenario() -> None:
        worker = FinishedWorker()
        supervisor = sandbox_module.WorkspaceWorkerSupervisor()
        cleanup_path = tmp_path / ".ravana-script-test"
        cleanup_path.write_text("stale")

        async def spawn_worker():
            return worker

        def fail_cleanup(temp_path):
            raise sandbox_module.WorkspaceStagingCleanupError(
                "forced cleanup failure"
            )

        monkeypatch.setattr(
            sandbox_module,
            "cleanup_workspace_staged_file",
            fail_cleanup,
        )
        sandbox_module._schedule_workspace_worker_spawn_cleanup(
            asyncio.create_task(spawn_worker()),
            supervisor=supervisor,
            cleanup_path=cleanup_path,
        )
        await asyncio.wait_for(worker.reaped.wait(), timeout=0.2)

        with pytest.raises(
            sandbox_module.WorkspaceStagingCleanupError,
            match="cleanup failed",
        ):
            await supervisor.aclose(timeout_seconds=0.2)

    asyncio.run(scenario())


def test_workspace_worker_supervisor_rejects_work_after_close():
    async def scenario() -> None:
        supervisor = sandbox_module.WorkspaceWorkerSupervisor()
        await supervisor.aclose(timeout_seconds=0.1)

        with pytest.raises(
            sandbox_module.WorkspaceStagingCleanupError,
            match="closed",
        ):
            supervisor.begin_operation()

        late_task = asyncio.create_task(asyncio.sleep(0))
        with pytest.raises(
            sandbox_module.WorkspaceStagingCleanupError,
            match="closed",
        ):
            supervisor.track(late_task)
        await asyncio.sleep(0)
        assert late_task.cancelled()

    asyncio.run(scenario())


def test_terminal_output_failure_reserves_part_of_the_cleanup_deadline(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        run_process = _ControlledProcess(exit_code=0)
        runner = DockerSandboxRunner(
            docker=sys.executable, cleanup_timeout_seconds=0.2
        )
        real_settle_readers = runner._settle_readers
        settle_calls = 0
        terminal_deadlines: list[float | None] = []

        async def staged_settle_readers(readers, deadline):
            nonlocal settle_calls
            settle_calls += 1
            if settle_calls == 1:
                return await real_settle_readers(readers, deadline)
            terminal_deadlines.append(deadline)
            return "forced terminal drain failure"

        async def recorded_finish(
            proc,
            process_waiter,
            name,
            readers,
            *,
            deadline=None,
            identity_known=True,
        ):
            terminal_deadlines.append(deadline)
            return None, None, False

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                marker = argv[argv.index("ravana-launcher") + 1]
                run_process.stderr = asyncio.StreamReader()
                run_process.stderr.feed_data(f"{marker}\n".encode())
                run_process.stderr.feed_eof()
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(runner, "_settle_readers", staged_settle_readers)
        monkeypatch.setattr(runner, "_finish_uninterruptibly", recorded_finish)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )

        with pytest.raises(SandboxOutcomeUnknown, match="output"):
            await runner.run(_spec(tmp_path))

        assert len(terminal_deadlines) == 2
        assert terminal_deadlines[0] is not None
        assert terminal_deadlines[1] is not None
        assert terminal_deadlines[1] > terminal_deadlines[0]

    asyncio.run(scenario())


def test_terminal_output_deadline_reserves_time_to_remove_container(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        run_process = _ControlledProcess(exit_code=0)
        run_process.stderr = asyncio.StreamReader()
        cleanup_targets: list[str] = []
        container_exists = True

        async def fake_create_subprocess_exec(*argv, **kwargs):
            nonlocal container_exists
            if argv[1] == "create":
                marker = argv[argv.index("ravana-launcher") + 1]
                run_process.stderr.feed_data(f"{marker}\n".encode())
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            if argv[1] == "rm":
                cleanup_targets.append(argv[-1])
                container_exists = False
                return _ControlledProcess(exit_code=0)
            if container_exists:
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        with pytest.raises(SandboxOutcomeUnknown, match="output"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.1
            ).run(_spec(tmp_path))

        assert cleanup_targets == [_FAKE_CONTAINER_ID]

    asyncio.run(scenario())


def test_final_workspace_scan_exception_is_outcome_unknown_after_start(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        run_process = _ControlledProcess(exit_code=0)
        runner = DockerSandboxRunner(
            docker=sys.executable, cleanup_timeout_seconds=0.2
        )

        async def dormant_monitor(spec):
            await asyncio.Future()

        scan_calls = 0

        async def broken_final_scan(*args, **kwargs):
            nonlocal scan_calls
            scan_calls += 1
            if scan_calls == 1:
                return None
            raise RuntimeError("quota scanner crashed")

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                marker = argv[argv.index("ravana-launcher") + 1]
                run_process.stderr = asyncio.StreamReader()
                run_process.stderr.feed_data(f"{marker}\n".encode())
                run_process.stderr.feed_eof()
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(runner, "_watch_workspace", dormant_monitor)
        monkeypatch.setattr(
            sandbox_module, "_workspace_violation_in_worker", broken_final_scan
        )
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )

        with pytest.raises(SandboxOutcomeUnknown, match="workspace enforcement"):
            await runner.run(_spec(tmp_path))

    asyncio.run(scenario())


def test_docker_runner_bounds_captured_output_while_draining_the_process(tmp_path):
    docker = _fake_docker(
        tmp_path,
        "import sys\n"
        "args = sys.argv[1:]\n"
        "marker = args[args.index('ravana-launcher') + 1]\n"
        "sys.stderr.write(marker + '\\n')\n"
        "sys.stdout.write('x' * 200_000)\n"
        "sys.stderr.write('y' * 200_000)\n",
    )
    limits = SandboxLimits(output_bytes=4096)

    result = asyncio.run(
        DockerSandboxRunner(docker=str(docker)).run(_spec(tmp_path, limits=limits))
    )

    assert result.exit_code == 0
    assert len(result.stdout) < 5000
    assert len(result.stderr) < 5000
    assert "truncated" in result.stdout
    assert "truncated" in result.stderr


def test_workspace_quota_stops_the_container(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docker = _fake_docker(
        tmp_path,
        "import pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'rm': raise SystemExit(0)\n"
        "if args[0] == 'container':\n"
        "    sys.stderr.write('Error: No such object\\n')\n"
        "    raise SystemExit(1)\n"
        "marker = args[args.index('ravana-launcher') + 1]\n"
        "mount = args[args.index('-v') + 1].split(':/workspace:', 1)[0]\n"
        "sys.stderr.write(marker + '\\n')\n"
        "sys.stderr.flush()\n"
        "(pathlib.Path(mount) / 'too-large.bin').write_bytes(b'x' * 4096)\n"
        "time.sleep(60)\n",
    )
    limits = SandboxLimits(
        timeout_seconds=2, workspace_bytes=1024, workspace_files=100
    )

    result = asyncio.run(
        DockerSandboxRunner(
            docker=str(docker), cleanup_timeout_seconds=0.3
        ).run(
            SandboxSpec(
                image="python:3.11-slim",
                argv=["python", "main.py"],
                workspace=workspace,
                limits=limits,
            )
        )
    )

    assert result.exit_code == 122
    assert "workspace" in result.stderr and "limit" in result.stderr


def test_workspace_measurement_failure_before_spawn_is_infrastructure_error(
    tmp_path,
):
    workspace = tmp_path / "not-a-directory"
    workspace.write_text("x")
    docker = _fake_docker(tmp_path, "raise SystemExit(0)\n")

    with pytest.raises(SandboxError, match="measured before start"):
        asyncio.run(
            DockerSandboxRunner(docker=str(docker)).run(
                SandboxSpec(
                    image="python:3.11-slim",
                    argv=["python", "main.py"],
                    workspace=workspace,
                )
            )
        )


def test_timeout_cleanup_failure_returns_within_a_hard_bound(tmp_path):
    pid_file = tmp_path / "run.pid"
    docker = _fake_shell_docker(
        tmp_path,
        "if [ \"$1\" = \"rm\" ]; then exit 2; fi\n"
        "if [ \"$1\" = \"container\" ]; then echo 'daemon unavailable' >&2; exit 2; fi\n"
        f"echo $$ > {shlex.quote(str(pid_file))}\n"
        "exec sleep 60\n",
    )
    limits = SandboxLimits(timeout_seconds=1)
    started = time.monotonic()

    with pytest.raises(SandboxOutcomeUnknown, match="cleanup") as exc_info:
        asyncio.run(
            DockerSandboxRunner(docker=str(docker), cleanup_timeout_seconds=0.5).run(
                _spec(tmp_path, limits=limits)
            )
        )

    assert time.monotonic() - started < 3.0
    assert pid_file.exists(), str(exc_info.value)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_cancelling_runner_cleans_up_the_started_process(tmp_path):
    pid_file = tmp_path / "run.pid"
    docker = _fake_shell_docker(
        tmp_path,
        "if [ \"$1\" = \"rm\" ]; then exit 0; fi\n"
        "if [ \"$1\" = \"container\" ]; then echo 'Error: No such object' >&2; exit 1; fi\n"
        f"echo $$ > {shlex.quote(str(pid_file))}\n"
        "exec sleep 60\n",
    )

    async def cancel_started_runner() -> None:
        task = asyncio.create_task(
            DockerSandboxRunner(docker=str(docker), cleanup_timeout_seconds=0.1).run(
                _spec(tmp_path)
            )
        )
        worker_pid = await _wait_for_pid_file(
            pid_file, task, timeout_seconds=1.0
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _await_task_with_hard_timeout(
                task,
                timeout_seconds=0.5,
                emergency_pid=worker_pid,
            )

    _run_async_scenario_in_killable_process(cancel_started_runner)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_repeated_cancellation_during_spawn_still_cleans_up_the_created_process(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()

        class FakeProcess:
            def __init__(self, *, exit_code=None, stderr=""):
                self.stdout = asyncio.StreamReader()
                self.stdout.feed_eof()
                self.stderr = asyncio.StreamReader()
                if stderr:
                    self.stderr.feed_data(stderr.encode())
                self.stderr.feed_eof()
                self.returncode = exit_code
                self.terminated = False
                self._done = asyncio.Event()
                if exit_code is not None:
                    self._done.set()

            async def wait(self):
                await self._done.wait()
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15
                self._done.set()

            def kill(self):
                self.terminate()

        run_process = FakeProcess()

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                _feed_startup_marker(argv, run_process)
                return _created_container_process()
            if argv[1] == "start":
                spawn_started.set()
                await release_spawn.wait()
                return run_process
            if argv[1] == "container":
                return FakeProcess(exit_code=1, stderr="Error: No such object")
            return FakeProcess(exit_code=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        task = asyncio.create_task(
            DockerSandboxRunner(docker=sys.executable, cleanup_timeout_seconds=0.1).run(
                _spec(tmp_path)
            )
        )
        await spawn_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_spawn.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run_process.terminated

    _run_async_scenario_in_killable_process(scenario)


def test_timeout_before_launcher_start_is_transient(tmp_path, monkeypatch):
    async def scenario() -> None:
        run_process = _ControlledProcess()

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            if argv[1] == "rm":
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        with pytest.raises(SandboxError, match="before.*start|never started"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.2
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )

    asyncio.run(scenario())


def test_late_create_client_is_reaped_without_starting_agent_code(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        release_create = asyncio.Event()
        late_process_stopped = asyncio.Event()
        start_called = asyncio.Event()
        late_create_process = _ControlledProcess(
            stdout=f"{_FAKE_CONTAINER_ID}\n",
            on_terminate=late_process_stopped.set,
            on_kill=late_process_stopped.set,
        )

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                await release_create.wait()
                return late_create_process
            if argv[1] == "start":
                start_called.set()
                raise AssertionError("agent code must not start after create timeout")
            if argv[1] == "rm":
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        with pytest.raises(SandboxError, match="container creation"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.1
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )
        assert not start_called.is_set()

        release_create.set()
        await asyncio.wait_for(late_process_stopped.wait(), timeout=0.3)
        assert not start_called.is_set()

    asyncio.run(scenario())


def test_main_process_creation_is_bounded_and_late_process_is_reaped(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        release_spawn = asyncio.Event()
        late_process_stopped = asyncio.Event()
        late_process = _ControlledProcess(
            on_terminate=late_process_stopped.set,
            on_kill=late_process_stopped.set,
        )

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                return _created_container_process()
            if argv[1] == "start":
                await release_spawn.wait()
                return late_process
            if argv[1] == "rm":
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        started = time.monotonic()
        with pytest.raises(SandboxOutcomeUnknown, match="process creation|spawn"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.1
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )
        assert time.monotonic() - started < 0.3

        release_spawn.set()
        await asyncio.wait_for(late_process_stopped.wait(), timeout=0.3)

    asyncio.run(scenario())


def test_delayed_spawn_oserror_is_classified_as_sandbox_failure(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        cleanup_targets: list[str] = []

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                return _created_container_process()
            if argv[1] == "start":
                await asyncio.sleep(0.02)
                raise OSError("delayed spawn failure")
            if argv[1] == "rm":
                cleanup_targets.append(argv[-1])
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        with pytest.raises(SandboxError, match="failed to spawn"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.2
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )
        assert cleanup_targets == [_FAKE_CONTAINER_ID]

    asyncio.run(scenario())


def test_cancellation_wins_over_delayed_spawn_oserror(tmp_path, monkeypatch):
    async def scenario() -> None:
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()
        cleanup_targets: list[str] = []

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                return _created_container_process()
            if argv[1] == "start":
                spawn_started.set()
                await release_spawn.wait()
                raise OSError("delayed spawn failure")
            if argv[1] == "rm":
                cleanup_targets.append(argv[-1])
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        task = asyncio.create_task(
            DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.2
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )
        )
        await spawn_started.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0)
        release_spawn.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup_targets == [_FAKE_CONTAINER_ID]

    asyncio.run(scenario())


def test_cancelled_start_spawn_failure_cleans_container_and_reopens_ledger(
    con, tmp_path, monkeypatch
):
    from ravana.runtime.tool_executor import RavanaToolExecutor
    from tests.test_tool_execution import _seed_run

    async def scenario() -> None:
        _seed_run(con, run_id="run-start-cancel")
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()
        cleanup_targets: list[str] = []

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                return _created_container_process()
            if argv[1] == "start":
                spawn_started.set()
                await release_spawn.wait()
                raise OSError("delayed start spawn failure")
            if argv[1] == "rm":
                cleanup_targets.append(argv[-1])
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        executor = RavanaToolExecutor(
            con,
            {
                "ci": _handler(
                    tmp_path,
                    runner=DockerSandboxRunner(docker=sys.executable),
                )
            },
        )
        task = asyncio.create_task(
            executor.execute(
                run_id="run-start-cancel",
                node_id="n",
                tool="ci",
                arguments={"code": "print(1)"},
                idempotency_key="start-cancel-key",
            )
        )
        await spawn_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        release_spawn.set()

        with pytest.raises(ToolRetrySafeCancellation):
            await task

        assert cleanup_targets == [_FAKE_CONTAINER_ID]
        row = con.execute(
            "SELECT status FROM tool_invocation WHERE idempotency_key = ?",
            ("start-cancel-key",),
        ).fetchone()
        assert row["status"] == "FAILED"

    asyncio.run(scenario())


def test_second_cancellation_cannot_interrupt_cleanup(tmp_path, monkeypatch):
    async def scenario() -> None:
        run_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        run_process = _ControlledProcess()

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                _feed_startup_marker(argv, run_process)
                return _created_container_process()
            if argv[1] == "start":
                run_started.set()
                return run_process
            if argv[1] == "rm":
                cleanup_started.set()
                await release_cleanup.wait()
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        task = asyncio.create_task(
            DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.5
            ).run(_spec(tmp_path))
        )
        await asyncio.wait_for(run_started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await _await_task_with_hard_timeout(
                task, timeout_seconds=1
            )
        assert run_process.terminate_called

    _run_async_scenario_in_killable_process(scenario)


def test_cleanup_uses_one_shared_deadline(tmp_path, monkeypatch):
    async def scenario() -> None:
        run_process = _ControlledProcess(
            terminate_completes=False,
            kill_completes=False,
        )

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                _feed_startup_marker(argv, run_process)
                return _created_container_process()
            assert argv[1] == "start"
            return run_process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        started = time.monotonic()
        with pytest.raises(SandboxOutcomeUnknown, match="cleanup"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.1
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )

        assert time.monotonic() - started < 0.3
        assert run_process.terminate_called
        assert run_process.kill_called

    asyncio.run(scenario())


def test_cleanup_deadline_bounds_control_process_creation(tmp_path, monkeypatch):
    async def scenario() -> None:
        control_spawn_started = asyncio.Event()
        release_control_spawn = asyncio.Event()
        late_process_killed = asyncio.Event()
        run_process = _ControlledProcess()
        late_process = _ControlledProcess(on_kill=late_process_killed.set)

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "create":
                _feed_startup_marker(argv, run_process)
                return _created_container_process()
            if argv[1] == "start":
                return run_process
            control_spawn_started.set()
            await release_control_spawn.wait()
            return late_process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        started = time.monotonic()
        with pytest.raises(SandboxOutcomeUnknown, match="cleanup"):
            await DockerSandboxRunner(
                docker=sys.executable, cleanup_timeout_seconds=0.1
            ).run(
                _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
            )

        assert control_spawn_started.is_set()
        assert time.monotonic() - started < 0.3
        release_control_spawn.set()
        await asyncio.wait_for(late_process_killed.wait(), timeout=0.2)

    asyncio.run(scenario())


def test_tool_outcome_unknown_is_always_fatal():
    assert ToolOutcomeUnknown("unknown").kind is ToolFailureKind.FATAL
