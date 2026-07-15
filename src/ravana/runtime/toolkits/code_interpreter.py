"""`code_interpreter` toolkit (§1.7, §8, §10.1). The agent submits code; the
handler writes it into THIS run's isolated workspace and executes it in a
sandbox with §8's isolation (no network, scoped mount, hard quotas).

Security is layered:
  - The workspace is `runs/<run_id>/workspace` — per run, never the parent
    project (§10.1). The handler locates it from `run_id` at call time (the
    registry/gateway is built before the run exists) and refuses a path that
    escapes the runs dir.
  - The agent's `filename` is forced to a bare name (no `/`, no `..`), and the
    script is atomically published so a prior sandbox-created symlink cannot
    redirect the host write outside the workspace.
  - Actual isolation (network/mount/quotas) is the SandboxRunner's job
    (sandbox.py) — injectable, so this handler is testable without a container
    runtime.

The code *running* is a side effect on the workspace filesystem, so the call is
side-effecting: RavanaToolExecutor dedupes a retried invocation on its logical
key rather than re-running it. A non-zero exit (or a cleanly terminated timeout)
is a normal RESULT fed back to the model. A sandbox that never started is a
TRANSIENT failure; an indeterminate outcome after failed cleanup is FATAL and
keeps the invocation STARTED so it cannot be executed twice.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolRetrySafeCancellation,
    ToolkitError,
    ToolOutcomeUnknown,
)
from ravana.runtime.toolkits.sandbox import (
    DockerSandboxRunner,
    SandboxCancelledBeforeStart,
    SandboxError,
    SandboxLimits,
    SandboxResult,
    SandboxRunner,
    SandboxSpec,
    SandboxOutcomeUnknown,
    workspace_capacity_violation,
)

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "filename": {"type": "string"},
        "args": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["code"],
    "additionalProperties": False,
}

# runtime → (image, interpreter argv prefix, default script filename).
_RUNTIMES: dict[str, tuple[str, list[str], str]] = {
    "python3.11": ("python:3.11-slim", ["python"], "main.py"),
    "node20": ("node:20-slim", ["node"], "main.js"),
}

# Hard ceilings a per-toolkit `config` override can't exceed (§8: "up to an
# org-wide ceiling"). A workflow can ask for less, never more.
_MAX_MEMORY_MB = 8192
_MAX_CPUS = 8.0
_MAX_TIMEOUT_S = 300
_MAX_WORKSPACE_MB = 8192
_MAX_WORKSPACE_FILES = 100_000


class CodeInterpreterHandler:
    input_schema = INPUT_SCHEMA
    executable = True

    def __init__(
        self,
        config: dict[str, Any],
        *,
        runs_dir: Path | None,
        runner: SandboxRunner | None = None,
    ):
        runtime = config.get("runtime")
        if runtime not in _RUNTIMES:
            raise ToolkitError(
                f"code_interpreter: unsupported runtime {runtime!r} (one of {sorted(_RUNTIMES)})"
            )
        self._image, self._interp, self._default_filename = _RUNTIMES[runtime]
        sandbox_executable = _sandbox_executable(config)
        _validate_network_config(config)
        self._limits = SandboxLimits(
            memory_mb=_clamp_int(config.get("memory_mb", 2048), 64, _MAX_MEMORY_MB),
            cpus=_clamp_float(config.get("cpus", 2.0), 0.1, _MAX_CPUS),
            timeout_seconds=_clamp_int(config.get("timeout_seconds", 60), 1, _MAX_TIMEOUT_S),
            workspace_bytes=(
                _clamp_int(
                    config.get("workspace_mb", 512), 16, _MAX_WORKSPACE_MB
                )
                * 1024
                * 1024
            ),
            workspace_files=_clamp_int(
                config.get("workspace_files", 10_000),
                100,
                _MAX_WORKSPACE_FILES,
            ),
        )
        self._runs_dir = runs_dir
        self._runner = runner or DockerSandboxRunner(docker=sandbox_executable)
        self.description = (
            f"Execute code in an isolated {runtime} sandbox (no network, workspace-only filesystem). "
            "Provide 'code'; optional 'filename' (a bare name) and 'args'. Returns exit code, stdout and stderr."
        )

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        # Running code writes to the run's workspace filesystem — a side effect,
        # so a retried logical invocation dedupes rather than re-executing.
        return True

    async def call(self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str) -> str:
        workspace = self._workspace_for(run_id)
        filename = _safe_filename(arguments.get("filename"), self._default_filename)
        _write_script(
            workspace,
            filename,
            str(arguments["code"]),
            limits=self._limits,
        )

        args = [str(a) for a in arguments.get("args", [])]
        spec = SandboxSpec(
            image=self._image,
            argv=[*self._interp, filename, *args],
            workspace=workspace,
            limits=self._limits,
        )
        try:
            result = await self._runner.run(spec)
        except SandboxCancelledBeforeStart as exc:
            raise ToolRetrySafeCancellation(
                "code_interpreter cancelled before sandbox execution"
            ) from exc
        except SandboxOutcomeUnknown as exc:
            raise ToolOutcomeUnknown(
                f"code_interpreter sandbox outcome is unknown: {exc}"
            ) from exc
        except SandboxError as exc:
            # Infrastructure (docker absent/unreachable, image pull) — §3.6
            # "sandbox cold-start" is TRANSIENT: the engine retries the attempt.
            raise ToolkitError(f"code_interpreter sandbox unavailable: {exc}", kind=ToolFailureKind.TRANSIENT) from exc
        return _format_result(result)

    async def aclose(self) -> None:
        return None

    def _workspace_for(self, run_id: str) -> Path:
        if self._runs_dir is None:
            raise ToolkitError(
                "code_interpreter has no runs directory configured — the Local tier must wire runs_dir",
                kind=ToolFailureKind.FATAL,
            )
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise ToolkitError(
                f"code_interpreter: refusing an invalid run id ({run_id!r})",
                kind=ToolFailureKind.FATAL,
            )

        runs_dir = self._runs_dir.resolve()
        run_dir = runs_dir / run_id
        workspace = run_dir / "workspace"
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            if run_dir.is_symlink():
                raise ToolkitError(
                    "code_interpreter: refusing a run directory symlink",
                    kind=ToolFailureKind.FATAL,
                )
            run_dir.mkdir(exist_ok=True)
            if workspace.is_symlink():
                raise ToolkitError(
                    "code_interpreter: refusing a workspace symlink",
                    kind=ToolFailureKind.FATAL,
                )
            workspace.mkdir(exist_ok=True)
        except ToolkitError:
            raise
        except OSError as exc:
            raise ToolkitError(
                f"code_interpreter: unable to prepare the run workspace ({type(exc).__name__})",
                kind=ToolFailureKind.FATAL,
            ) from exc

        # Re-check after creation so an existing alias within runs_dir cannot
        # point this run at another run's workspace.
        if workspace.resolve() != workspace:
            raise ToolkitError(
                "code_interpreter: refusing an aliased workspace",
                kind=ToolFailureKind.FATAL,
            )
        return workspace


def _write_script(
    workspace: Path,
    filename: str,
    code: str,
    *,
    limits: SandboxLimits,
) -> None:
    """Publish the script atomically so an existing symlink or hard link is
    replaced as a directory entry instead of being followed by the host write."""
    encoded = code.encode("utf-8")
    with _workspace_publication_lock(workspace):
        violation = workspace_capacity_violation(
            workspace,
            limits=limits,
            additional_bytes=len(encoded),
            additional_files=1,
        )
        if violation is not None:
            kind = (
                ToolFailureKind.TRANSIENT
                if violation.measurement_failed
                else ToolFailureKind.MODEL_ADDRESSABLE
            )
            raise ToolkitError(
                f"code_interpreter: {violation.message}", kind=kind
            )

        _publish_script(workspace, filename, encoded)


@contextlib.contextmanager
def _workspace_publication_lock(workspace: Path) -> Iterator[None]:
    """Serialize quota measurement and publication across local processes."""
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    locked = False
    try:
        fd = os.open(workspace.parent / ".ravana-workspace.lock", flags, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("workspace publication lock is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    except ToolkitError:
        raise
    except OSError as exc:
        raise ToolkitError(
            "code_interpreter: unable to lock the run workspace "
            f"({type(exc).__name__})",
            kind=ToolFailureKind.TRANSIENT,
        ) from exc
    finally:
        if fd is not None:
            if locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _publish_script(workspace: Path, filename: str, encoded: bytes) -> None:
    """Atomically replace the destination while the quota lock is held."""

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=workspace,
            prefix=".ravana-script-",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(encoded)
        os.replace(temp_path, workspace / filename)
        temp_path = None
    except OSError as exc:
        raise ToolkitError(
            f"code_interpreter: unable to prepare the script ({type(exc).__name__})"
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _safe_filename(raw: Any, default: str) -> str:
    """Force the script name to a bare filename in the workspace root — no
    directory parts, no `..`, no absolute path — so agent-supplied code can't be
    written outside the sandboxed workspace."""
    if raw is None or raw == "":
        return default
    name = str(raw)
    if "/" in name or "\\" in name or name in (".", "..") or ".." in Path(name).parts or Path(name).is_absolute():
        raise ToolkitError(f"code_interpreter: 'filename' must be a bare name, not a path ({name!r})")
    return name


def _validate_network_config(config: dict[str, Any]) -> None:
    """Keep container egress disabled until Ravana can enforce the architecture's
    required per-toolkit host allow-list. In particular, reject stringly values
    such as ``"false"`` instead of treating them as truthy opt-ins."""
    if "network" not in config or config["network"] is False:
        return
    raise ToolkitError(
        "code_interpreter: network egress is disabled; host allow-list networking is not implemented"
    )


def _sandbox_executable(config: dict[str, Any]) -> str:
    """Resolve the supported local OCI runtime without ignoring the manifest.

    Silently ignoring a managed/disabled/typoed backend would execute code in a
    different trust boundary than the workflow author selected.
    """
    sandbox = config.get("sandbox", "docker")
    if isinstance(sandbox, str) and sandbox in {"docker", "podman"}:
        return sandbox
    raise ToolkitError(
        "code_interpreter: config.sandbox must be 'docker' or 'podman' in Phase 0b"
    )


def _format_result(result: SandboxResult) -> str:
    parts = [f"exit_code: {result.exit_code}"]
    if result.timed_out:
        parts.append("status: timed out")
    if result.stdout:
        parts.append("stdout:\n" + result.stdout)
    if result.stderr:
        parts.append("stderr:\n" + result.stderr)
    return "\n".join(parts)


def _clamp_int(value: Any, low: int, high: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ToolkitError(f"code_interpreter: expected an int, got {value!r}") from None
    return max(low, min(high, n))


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ToolkitError(f"code_interpreter: expected a number, got {value!r}") from None
    return max(low, min(high, n))
