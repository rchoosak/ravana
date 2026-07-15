"""code_interpreter toolkit + sandbox (§8, §10.1). The Docker isolation flags
are asserted on the pure argv builder; the handler is exercised through a fake
SandboxRunner, so nothing here needs a Docker daemon.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time

import pytest

from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolkitError,
    ToolOutcomeUnknown,
)
from ravana.runtime.toolkits.code_interpreter import CodeInterpreterHandler
from ravana.runtime.toolkits.sandbox import (
    DockerSandboxRunner,
    SandboxError,
    SandboxLimits,
    SandboxResult,
    SandboxSpec,
    SandboxOutcomeUnknown,
    build_docker_argv,
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
    assert argv[:3] == ["docker", "run", "--rm"]  # ephemeral: filesystem never durable
    assert argv[argv.index("--network") + 1] == "none"  # §8: no default egress
    assert "--read-only" in argv  # root fs read-only
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    # memory == memory-swap: a hard RAM cap with no swap escape hatch
    assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1] == "2048m"
    assert argv[argv.index("--cpus") + 1] == "2.0"
    assert "--pids-limit" in argv
    assert argv[argv.index("--name") + 1] == "ravana-ci-abc"
    # exactly ONE bind mount, scoped strictly to the run's workspace (§10.1)
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts == [f"{tmp_path.resolve()}:/workspace:rw"]
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


# --- handler ----------------------------------------------------------------
def _handler(tmp_path, *, runtime="python3.11", runner=None, **config):
    return CodeInterpreterHandler({"runtime": runtime, **config}, runs_dir=tmp_path, runner=runner or FakeRunner())


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


def test_node_runtime_uses_node_image_and_default_file(tmp_path):
    runner = FakeRunner()
    handler = _handler(tmp_path, runtime="node20", runner=runner)
    asyncio.run(handler.call(arguments={"code": "console.log(1)", "args": ["x"]}, idempotency_key="k", run_id="r"))
    assert runner.spec.image == "node:20-slim"
    assert runner.spec.argv == ["node", "main.js", "x"]


def test_unsupported_runtime_rejected_at_construction(tmp_path):
    with pytest.raises(ToolkitError, match="unsupported runtime"):
        CodeInterpreterHandler({"runtime": "ruby"}, runs_dir=tmp_path)


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
    handler = _handler(tmp_path, runner=runner, memory_mb=99999, cpus=64, timeout_seconds=99999)
    asyncio.run(handler.call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert runner.spec.limits.memory_mb == 8192  # ceiling
    assert runner.spec.limits.cpus == 8.0
    assert runner.spec.limits.timeout_seconds == 300


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


def test_nonzero_exit_and_timeout_are_results_not_exceptions(tmp_path):
    nonzero = FakeRunner(SandboxResult(exit_code=1, stdout="", stderr="boom"))
    out = asyncio.run(_handler(tmp_path, runner=nonzero).call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert "exit_code: 1" in out and "boom" in out

    timed = FakeRunner(SandboxResult(exit_code=124, stdout="", stderr="", timed_out=True))
    out = asyncio.run(_handler(tmp_path, runner=timed).call(arguments={"code": "x"}, idempotency_key="k", run_id="r"))
    assert "timed out" in out


# --- Docker runner process boundary -----------------------------------------
def _fake_docker(tmp_path, body: str):
    executable = tmp_path / "fake-docker"
    executable.write_text(f"#!{sys.executable}\n{body}")
    executable.chmod(0o700)
    return executable


def _fake_shell_docker(tmp_path, body: str):
    executable = tmp_path / "fake-docker"
    executable.write_text(f"#!/bin/sh\n{body}")
    executable.chmod(0o700)
    return executable


class _ControlledProcess:
    def __init__(
        self,
        *,
        exit_code=None,
        stderr="",
        terminate_completes=True,
        kill_completes=True,
        on_terminate=None,
        on_kill=None,
    ):
        self.stdout = asyncio.StreamReader()
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


def test_docker_exit_125_is_sandbox_infrastructure_failure(tmp_path):
    docker = _fake_docker(
        tmp_path,
        "import sys\nsys.stderr.write('daemon unavailable\\n')\nraise SystemExit(125)\n",
    )

    with pytest.raises(SandboxError, match="exit 125|daemon unavailable"):
        asyncio.run(DockerSandboxRunner(docker=str(docker)).run(_spec(tmp_path)))


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


def test_docker_runner_bounds_captured_output_while_draining_the_process(tmp_path):
    docker = _fake_docker(
        tmp_path,
        "import sys\nsys.stdout.write('x' * 200_000)\nsys.stderr.write('y' * 200_000)\n",
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
        while not pid_file.exists():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_started_runner())
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
            if argv[1] == "run":
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

    asyncio.run(scenario())


def test_timeout_stops_client_then_removes_a_late_created_container(
    tmp_path, monkeypatch
):
    async def scenario() -> None:
        client_stopped = asyncio.Event()
        remove_calls = 0
        inspect_calls = 0
        run_process = _ControlledProcess(on_terminate=client_stopped.set)

        async def fake_create_subprocess_exec(*argv, **kwargs):
            nonlocal remove_calls, inspect_calls
            if argv[1] == "run":
                return run_process
            assert client_stopped.is_set()
            if argv[1] == "rm":
                remove_calls += 1
                if remove_calls == 1:
                    return _ControlledProcess(
                        exit_code=1, stderr="Error: No such container"
                    )
                return _ControlledProcess(exit_code=0)
            inspect_calls += 1
            if inspect_calls == 1:
                return _ControlledProcess(exit_code=0)
            return _ControlledProcess(
                exit_code=1, stderr="Error: No such object"
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        result = await DockerSandboxRunner(
            docker=sys.executable, cleanup_timeout_seconds=0.2
        ).run(
            _spec(tmp_path, limits=SandboxLimits(timeout_seconds=0.01))
        )

        assert result.timed_out
        assert run_process.terminate_called
        assert remove_calls == 2
        assert inspect_calls == 2

    asyncio.run(scenario())


def test_second_cancellation_cannot_interrupt_cleanup(tmp_path, monkeypatch):
    async def scenario() -> None:
        run_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        run_process = _ControlledProcess()

        async def fake_create_subprocess_exec(*argv, **kwargs):
            if argv[1] == "run":
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
        await run_started.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run_process.terminate_called

    asyncio.run(scenario())


def test_cleanup_uses_one_shared_deadline(tmp_path, monkeypatch):
    async def scenario() -> None:
        run_process = _ControlledProcess(
            terminate_completes=False,
            kill_completes=False,
        )

        async def fake_create_subprocess_exec(*argv, **kwargs):
            assert argv[1] == "run"
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
            if argv[1] == "run":
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
