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

import asyncio
import contextlib
import fcntl
import os
import stat
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ravana.runtime import git_handoff
from ravana.runtime.git_workspace import (
    DEFAULT_BASE_REF,
    GitError,
    git_toplevel,
    provision_run_workspace,
    provision_shadow_workspace,
)
from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolRetrySafeCancellation,
    ToolkitError,
    ToolOutcomeUnknown,
)
from ravana.runtime.toolkits.sandbox import (
    AsyncResourceLifecycle,
    DockerSandboxRunner,
    SandboxCancelledBeforeStart,
    SandboxError,
    SandboxLimits,
    SandboxResult,
    SandboxRunner,
    SandboxSpec,
    SandboxOutcomeUnknown,
    WorkspaceStagingCleanupError,
    WorkspaceStagingError,
    WorkspaceWorkerSupervisor,
    cleanup_workspace_staged_file,
    stage_workspace_script_async,
    workspace_capacity_violation_async,
)

_SCRIPT_SIZE_CHUNK_CHARACTERS = 64 * 1024

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
_WORKSPACE_LOCK_POLL_SECONDS = 0.02
_WORKSPACE_WORKER_CLOSE_SECONDS = 1.0


@dataclass(frozen=True)
class _WorkspaceContext:
    workspace: Path
    project_subpath: Path


class CodeInterpreterHandler:
    input_schema = INPUT_SCHEMA
    executable = True

    def __init__(
        self,
        config: dict[str, Any],
        *,
        runs_dir: Path | None,
        project_dir: Path | None = None,
        base_ref: str = DEFAULT_BASE_REF,
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
        self._project_dir = project_dir or _project_from_runs_dir(runs_dir)
        self._base_ref = base_ref
        self._prepared_workspaces: dict[str, _WorkspaceContext] = {}
        self._runner = runner or DockerSandboxRunner(docker=sandbox_executable)
        self._workspace_workers = WorkspaceWorkerSupervisor()
        self._lifecycle = AsyncResourceLifecycle(
            "code_interpreter handler"
        )
        self._resources_closed = False
        self.description = (
            f"Execute code in an isolated {runtime} sandbox (no network, workspace-only filesystem). "
            "Provide 'code'; optional 'filename' (a bare name) and 'args'. Returns exit code, stdout and stderr."
        )

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        # Running code writes to the run's workspace filesystem — a side effect,
        # so a retried logical invocation dedupes rather than re-executing.
        return True

    async def prepare_run(self, run_id: str) -> None:
        """Provision the Local-tier workspace before the run is persisted."""
        try:
            self._lifecycle.enter()
        except RuntimeError as exc:
            raise ToolkitError(
                "code_interpreter handler is closed",
                kind=ToolFailureKind.FATAL,
            ) from exc
        try:
            context = await self._workspace_context(
                run_id, validate_requested_base=True
            )
            self._prepared_workspaces[run_id] = context
        finally:
            self._lifecycle.exit()

    async def hand_off_run(self, run_id: str) -> str | None:
        """Surface this run's workspace branch as a patch (§10.1).

        The bookend to `prepare_run`: this handler provisioned the workspace, so
        it owns handing it back. A run that never provisioned one (no runs dir,
        or a workspace that was never created) has nothing to surface and
        reports None rather than treating it as a failure.
        """
        if self._runs_dir is None:
            return None
        try:
            self._lifecycle.enter()
        except RuntimeError as exc:
            raise ToolkitError(
                "code_interpreter handler is closed",
                kind=ToolFailureKind.FATAL,
            ) from exc
        try:
            runs_dir = self._runs_dir.resolve()
            if not (runs_dir / run_id / "workspace").is_dir():
                return None
            result = await git_handoff.hand_off_run(runs_dir=runs_dir, run_id=run_id)
            return result.summary()
        finally:
            self._lifecycle.exit()

    async def call(self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str) -> str:
        try:
            self._lifecycle.enter()
        except RuntimeError as exc:
            raise ToolkitError(
                "code_interpreter handler is closed",
                kind=ToolFailureKind.FATAL,
            ) from exc
        try:
            return await self._call(
                arguments=arguments,
                idempotency_key=idempotency_key,
                run_id=run_id,
            )
        finally:
            self._lifecycle.exit()

    async def _call(
        self,
        *,
        arguments: dict[str, Any],
        idempotency_key: str,
        run_id: str,
    ) -> str:
        context = self._prepared_workspaces.get(run_id)
        if context is None:
            context = await self._workspace_context(
                run_id, validate_requested_base=False
            )
        workspace = context.workspace
        filename = _safe_filename(arguments.get("filename"), self._default_filename)
        args = [str(a) for a in arguments.get("args", [])]
        async with _workspace_execution_lock(
            workspace, timeout_seconds=self._limits.timeout_seconds
        ):
            execution_dir = _validated_execution_dir(
                workspace, context.project_subpath
            )
            container_cwd = (
                "/workspace"
                if context.project_subpath == Path(".")
                else f"/workspace/{context.project_subpath.as_posix()}"
            )
            spec = SandboxSpec(
                image=self._image,
                argv=[*self._interp, filename, *args],
                workspace=workspace,
                limits=self._limits,
                working_directory=container_cwd,
            )
            await _write_script(
                workspace,
                execution_dir,
                filename,
                str(arguments["code"]),
                limits=self._limits,
                supervisor=self._workspace_workers,
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
        await self._lifecycle.aclose(
            timeout_seconds=_WORKSPACE_WORKER_CLOSE_SECONDS
        )
        if self._resources_closed:
            return
        first_error: Exception | None = None
        try:
            await self._workspace_workers.aclose(
                timeout_seconds=_WORKSPACE_WORKER_CLOSE_SECONDS
            )
        except Exception as exc:
            first_error = exc

        close_runner = getattr(self._runner, "aclose", None)
        if close_runner is not None:
            try:
                await close_runner()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
        self._resources_closed = True

    async def _workspace_context(
        self, run_id: str, *, validate_requested_base: bool
    ) -> _WorkspaceContext:
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
            if workspace.is_symlink():
                raise ToolkitError(
                    "code_interpreter: refusing a workspace symlink",
                    kind=ToolFailureKind.FATAL,
                )

            project_subpath = Path()
            project_toplevel = (
                await git_toplevel(self._project_dir)
                if self._project_dir is not None
                else None
            )
            if project_toplevel is not None and self._project_dir is not None:
                project_subpath = self._project_dir.resolve().relative_to(
                    project_toplevel.resolve()
                )
                workspace = await provision_run_workspace(
                    base_repo=project_toplevel,
                    runs_dir=runs_dir,
                    run_id=run_id,
                    base_ref=(
                        self._base_ref
                        if validate_requested_base or not workspace.exists()
                        else None
                    ),
                )
            elif self._project_dir is not None:
                if self._base_ref != DEFAULT_BASE_REF:
                    raise ToolkitError(
                        "code_interpreter: --base-ref requires a git project",
                        kind=ToolFailureKind.FATAL,
                    )
                workspace = await provision_shadow_workspace(
                    project_dir=self._project_dir,
                    runs_dir=runs_dir,
                    run_id=run_id,
                )
            else:
                run_dir.mkdir(exist_ok=True)
                workspace.mkdir(exist_ok=True)
        except ToolkitError:
            raise
        except GitError as exc:
            raise ToolkitError(
                f"code_interpreter: unable to provision git workspace ({exc})",
                kind=ToolFailureKind.FATAL,
            ) from exc
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

        _validated_execution_dir(workspace, project_subpath)
        return _WorkspaceContext(
            workspace=workspace,
            project_subpath=project_subpath,
        )


def _project_from_runs_dir(runs_dir: Path | None) -> Path | None:
    if (
        runs_dir is not None
        and runs_dir.name == "runs"
        and runs_dir.parent.name == ".ravana"
    ):
        return runs_dir.parent.parent
    return None


def _validated_execution_dir(
    workspace: Path, project_subpath: Path
) -> Path:
    """Resolve the mutable project path afresh for every locked invocation."""
    if (
        workspace.is_symlink()
        or workspace.resolve() != workspace
        or not workspace.is_dir()
    ):
        raise ToolkitError(
            "code_interpreter: refusing an aliased workspace",
            kind=ToolFailureKind.FATAL,
        )
    execution_dir = workspace / project_subpath
    if (
        not execution_dir.is_dir()
        or execution_dir.is_symlink()
        or execution_dir.resolve() != execution_dir
        or not execution_dir.is_relative_to(workspace)
    ):
        raise ToolkitError(
            "code_interpreter: project path is missing or aliased inside the run workspace",
            kind=ToolFailureKind.FATAL,
        )
    return execution_dir


async def _write_script(
    workspace: Path,
    script_dir: Path,
    filename: str,
    code: str,
    *,
    limits: SandboxLimits,
    supervisor: WorkspaceWorkerSupervisor,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limits.timeout_seconds
    try:
        encoded_size = await _utf8_size(
            code,
            stop_after_bytes=limits.workspace_bytes,
            deadline=deadline,
        )
        violation = await workspace_capacity_violation_async(
            workspace,
            limits=limits,
            additional_bytes=encoded_size,
            additional_files=1,
            timeout_seconds=max(0.0, deadline - loop.time()),
            supervisor=supervisor,
        )
    except asyncio.CancelledError as exc:
        raise ToolRetrySafeCancellation(
            "code_interpreter cancelled during workspace measurement"
        ) from exc
    except asyncio.TimeoutError as exc:
        raise ToolkitError(
            "code_interpreter: script preparation exceeded its deadline",
            kind=ToolFailureKind.TRANSIENT,
        ) from exc
    except WorkspaceStagingCleanupError as exc:
        raise ToolOutcomeUnknown(
            f"code_interpreter workspace worker cleanup failed: {exc}"
        ) from exc

    if violation is not None:
        kind = (
            ToolFailureKind.TRANSIENT
            if violation.measurement_failed
            else ToolFailureKind.MODEL_ADDRESSABLE
        )
        raise ToolkitError(
            f"code_interpreter: {violation.message}", kind=kind
        )

    temp_path: Path | None = None
    workspace_fd: int | None = None
    script_dir_fd: int | None = None
    try:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise WorkspaceStagingError(
                "script staging worker exceeded its deadline"
            )
        # Stage in the workspace root, which is the stable bind mount, then
        # publish through directory descriptors opened with O_NOFOLLOW. The
        # nested project directory is mutable container state and must never be
        # trusted as a path again after validation.
        workspace_fd = _open_directory_fd(workspace)
        script_dir_fd = _open_directory_fd(script_dir)
        temp_path = await stage_workspace_script_async(
            workspace,
            code,
            timeout_seconds=remaining,
            supervisor=supervisor,
        )
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError

        staged_stat = temp_path.lstat()
        if (
            not stat.S_ISREG(staged_stat.st_mode)
            or staged_stat.st_size != encoded_size
        ):
            raise WorkspaceStagingError(
                "script staging worker returned an invalid file"
            )
        os.replace(
            temp_path.name,
            filename,
            src_dir_fd=workspace_fd,
            dst_dir_fd=script_dir_fd,
        )
        temp_path = None
    except asyncio.CancelledError as exc:
        raise ToolRetrySafeCancellation(
            "code_interpreter cancelled before script publication"
        ) from exc
    except WorkspaceStagingCleanupError as exc:
        raise ToolOutcomeUnknown(
            f"code_interpreter script cleanup failed: {exc}"
        ) from exc
    except WorkspaceStagingError as exc:
        raise ToolkitError(
            f"code_interpreter: unable to prepare the script ({exc})",
            kind=ToolFailureKind.TRANSIENT,
        ) from exc
    except OSError as exc:
        raise ToolkitError(
            "code_interpreter: unable to prepare the script "
            f"({type(exc).__name__})",
            kind=ToolFailureKind.TRANSIENT,
        ) from exc
    finally:
        if temp_path is not None:
            try:
                cleanup_workspace_staged_file(temp_path)
            except WorkspaceStagingCleanupError as exc:
                raise ToolOutcomeUnknown(
                    f"code_interpreter script cleanup failed: {exc}"
                ) from exc
        for directory_fd in (script_dir_fd, workspace_fd):
            if directory_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(directory_fd)


def _open_directory_fd(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(f"not a directory: {path}")
        return fd
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


async def _utf8_size(
    value: str,
    *,
    stop_after_bytes: int,
    deadline: float,
) -> int:
    """Count UTF-8 bytes in bounded chunks without monopolizing the loop."""
    total = 0
    loop = asyncio.get_running_loop()
    for offset in range(0, len(value), _SCRIPT_SIZE_CHUNK_CHARACTERS):
        if loop.time() >= deadline:
            raise asyncio.TimeoutError
        chunk = value[offset : offset + _SCRIPT_SIZE_CHUNK_CHARACTERS]
        total += len(chunk.encode("utf-8"))
        if total > stop_after_bytes:
            return total
        await asyncio.sleep(0)
    return total


@contextlib.asynccontextmanager
async def _workspace_execution_lock(
    workspace: Path, *, timeout_seconds: float
) -> AsyncIterator[None]:
    """Own one mutable run workspace from publication through execution."""
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    locked = False
    try:
        try:
            fd = os.open(workspace.parent / ".ravana-workspace.lock", flags, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("workspace publication lock is not a regular file")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_seconds
            while not locked:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise ToolkitError(
                            "code_interpreter: timed out waiting for the run workspace",
                            kind=ToolFailureKind.TRANSIENT,
                        )
                    try:
                        await asyncio.sleep(
                            min(_WORKSPACE_LOCK_POLL_SECONDS, remaining)
                        )
                    except asyncio.CancelledError as exc:
                        raise ToolRetrySafeCancellation(
                            "code_interpreter cancelled before workspace ownership"
                        ) from exc
        except (ToolkitError, ToolRetrySafeCancellation):
            raise
        except OSError as exc:
            raise ToolkitError(
                "code_interpreter: unable to lock the run workspace "
                f"({type(exc).__name__})",
                kind=ToolFailureKind.TRANSIENT,
            ) from exc
        yield
    finally:
        if fd is not None:
            if locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)


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
