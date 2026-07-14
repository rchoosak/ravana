"""code_interpreter toolkit + sandbox (§8, §10.1). The Docker isolation flags
are asserted on the pure argv builder; the handler is exercised through a fake
SandboxRunner, so nothing here needs a Docker daemon.
"""

from __future__ import annotations

import asyncio

import pytest

from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.toolkits.code_interpreter import CodeInterpreterHandler
from ravana.runtime.toolkits.sandbox import (
    SandboxError,
    SandboxLimits,
    SandboxResult,
    SandboxSpec,
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


def _spec(tmp_path, *, network=False, limits=None) -> SandboxSpec:
    return SandboxSpec(
        image="python:3.11-slim", argv=["python", "main.py"], workspace=tmp_path,
        limits=limits or SandboxLimits(), network=network,
    )


# --- docker argv (§8 isolation), pure + no daemon ---------------------------
def test_docker_argv_has_isolation_flags(tmp_path):
    argv = build_docker_argv(_spec(tmp_path), name="ravana-ci-abc")
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
    # image then the in-container command last
    assert argv[-3:] == ["python:3.11-slim", "python", "main.py"]


def test_docker_argv_network_is_opt_in(tmp_path):
    argv = build_docker_argv(_spec(tmp_path, network=True), name="n")
    assert argv[argv.index("--network") + 1] == "bridge"


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
    assert runner.spec.network is False  # §8 default
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
