"""Sandbox execution boundary for the `code_interpreter` toolkit (§8, §10.1).

`code_interpreter` runs agent-authored code — the highest-blast-radius thing in
the system — so execution is isolated behind a `SandboxRunner` the handler is
handed. The Local/Embedded tier (§10.1) backs it with a local OCI container;
hosted tiers swap in a managed provider (E2B/Modal) behind this same interface
(§8 "hidden behind the manifest, a reversible implementation detail").

The security posture is enforced in `build_docker_argv` (a pure function, so it
is exhaustively unit-testable without a container runtime): §8's mandate — no
default network egress, a bind mount scoped strictly to that run's workspace and
nothing else, hard per-invocation resource quotas, and a filesystem never
treated as durable (`--rm`). `DockerSandboxRunner` is the thin process wrapper
that actually spawns it; tests inject a fake `SandboxRunner`, so nothing here
requires Docker or Podman to be present.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SandboxLimits:
    """Hard per-invocation quotas (§8 default: 2 vCPU / 2GB / 60s)."""

    memory_mb: int = 2048
    cpus: float = 2.0
    timeout_seconds: int = 60
    pids: int = 256
    output_bytes: int = 10_000
    workspace_bytes: int = 512 * 1024 * 1024
    workspace_files: int = 10_000


@dataclass(frozen=True)
class SandboxSpec:
    image: str
    argv: list[str]  # the command run INSIDE the container (e.g. ["python", "main.py"])
    workspace: Path  # host dir bind-mounted read-write at /workspace — the ONLY mount
    limits: SandboxLimits = SandboxLimits()


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxError(Exception):
    """The sandbox itself could not run the code — a runtime that's absent
    or unreachable, an image that won't pull, a spawn failure. This is
    infrastructure (§3.6 "sandbox cold-start"), NOT the agent's code failing;
    the handler maps it to a TRANSIENT ToolkitError so the engine retries."""


class SandboxOutcomeUnknown(SandboxError):
    """The sandbox started, but cleanup/output failure hid its final outcome.

    Retrying this invocation is unsafe because agent code may already have
    mutated the workspace.
    """


class SandboxCancelledBeforeStart(asyncio.CancelledError):
    """Cancellation occurred before any request could start agent code."""


class SandboxRunner(Protocol):
    async def run(self, spec: SandboxSpec) -> SandboxResult: ...


# Root filesystem is read-only; only the workspace mount and a small tmpfs are
# writable, so nothing the code does persists beyond the workspace or the run.
_TMPFS_SIZE = "64m"
_LOG_DRIVER = "none"


def build_docker_argv(
    spec: SandboxSpec, *, name: str, startup_marker: str
) -> list[str]:
    """The `docker create` argv enforcing §8's isolation. Pure and deterministic so
    every security flag is asserted in tests without invoking Docker:

    `DockerSandboxRunner` completes this non-executing create first, captures
    the immutable container ID, then uses `docker start --attach <id>`. That
    lifecycle prevents a daemon-side create delayed past client cancellation
    from ever starting agent code.

    - `--rm`: the container filesystem is removed after the started process exits.
    - `--network none`: no egress until a per-host allow-list can be enforced (§8).
    - `--memory`/`--memory-swap` equal: a hard RAM cap with no swap escape hatch.
    - `--cpus`, `--pids-limit`: CPU quota and fork-bomb ceiling.
    - `--read-only` root + a size-capped `--tmpfs /tmp`: the ONLY writable places
      are the workspace mount and scratch tmp.
    - `-v <workspace>:/workspace`: the bind mount is scoped to THIS run's
      workspace and nothing else — the isolation §10.1 requires is enforced at
      the mount, not by convention.
    - `--cap-drop ALL` + `--security-opt no-new-privileges`: drop Linux
      capabilities and block privilege escalation.
    """
    workspace = spec.workspace.resolve()
    workspace_stat = workspace.stat()
    limits = spec.limits
    argv = [
        "docker",
        "create",
        "--rm",
        "--name",
        name,
        # Match the workspace owner rather than Docker's default UID 0. This
        # keeps native-Linux bind mounts readable/writable after dropping every
        # capability and prevents root-owned artifacts on the host.
        "--user",
        f"{workspace_stat.st_uid}:{workspace_stat.st_gid}",
        "--network",
        "none",
        "--memory",
        f"{limits.memory_mb}m",
        "--memory-swap",
        f"{limits.memory_mb}m",
        "--cpus",
        str(limits.cpus),
        "--pids-limit",
        str(limits.pids),
        # Disable daemon-side persistence. Ravana still receives attached
        # stdout/stderr and drains them through its own bounded stream readers.
        # `none` is supported by both Docker and Podman.
        "--log-driver",
        _LOG_DRIVER,
        # RLIMIT_FSIZE is a hard per-file backstop. Aggregate workspace usage is
        # enforced concurrently by the host-side monitor below.
        "--ulimit",
        f"fsize={limits.workspace_bytes}:{limits.workspace_bytes}",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,exec,size={_TMPFS_SIZE}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{workspace}:/workspace:rw",
        "-w",
        "/workspace",
        spec.image,
        "/bin/sh",
        "-c",
        'printf "%s\\n" "$1" >&2; shift; exec "$@"',
        "ravana-launcher",
        startup_marker,
        *spec.argv,
    ]
    return argv


def build_docker_start_argv(container_id: str) -> list[str]:
    """Start and attach only after create returned a concrete identity."""
    return ["docker", "start", "--attach", container_id]


_OUTPUT_READ_CHUNK = 64 * 1024
_DEFAULT_CLEANUP_TIMEOUT_SECONDS = 10.0
_PODMAN_CLEANUP_TIMEOUT_SECONDS = 30.0
_WORKSPACE_POLL_SECONDS = 0.05
_CLEANUP_POLL_SECONDS = 0.02
_WORKSPACE_LIMIT_EXIT_CODE = 122
_CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{12,64}")


@dataclass(frozen=True)
class WorkspaceViolation:
    message: str
    measurement_failed: bool = False


def _workspace_violation(
    workspace: Path, *, max_bytes: int, max_files: int
) -> WorkspaceViolation | None:
    """Measure a mutable workspace without following sandbox-created symlinks."""
    total_bytes = 0
    total_files = 0
    pending = [workspace]
    try:
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            stat_result = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        total_files += 1
                        if total_files > max_files:
                            return WorkspaceViolation(
                                "workspace file limit exceeded "
                                f"({total_files} > {max_files})"
                            )
                        if stat.S_ISDIR(stat_result.st_mode):
                            pending.append(Path(entry.path))
                            continue
                        total_bytes += stat_result.st_size
                        if total_bytes > max_bytes:
                            return WorkspaceViolation(
                                "workspace byte limit exceeded "
                                f"({total_bytes} > {max_bytes})"
                            )
            except FileNotFoundError:
                # The sandbox may remove a directory after its parent was read.
                # The next polling pass will measure the current tree again.
                continue
    except OSError as exc:
        return WorkspaceViolation(
            f"workspace usage could not be measured ({type(exc).__name__})",
            measurement_failed=True,
        )
    return None


def workspace_capacity_violation(
    workspace: Path,
    *,
    limits: SandboxLimits,
    additional_bytes: int,
    additional_files: int,
) -> WorkspaceViolation | None:
    """Check capacity before a host-side atomic publish enters the workspace.

    The additional file must coexist briefly with the current destination, so
    reserve both its bytes and directory entry before creating the temporary
    file. This keeps host preparation under the same aggregate limits enforced
    while the sandbox is running.
    """
    if additional_bytes < 0 or additional_files < 0:
        raise ValueError("additional workspace usage must not be negative")
    if additional_bytes > limits.workspace_bytes:
        return WorkspaceViolation(
            "workspace byte limit exceeded "
            f"({additional_bytes} > {limits.workspace_bytes})"
        )
    if additional_files > limits.workspace_files:
        return WorkspaceViolation(
            "workspace file limit exceeded "
            f"({additional_files} > {limits.workspace_files})"
        )
    return _workspace_violation(
        workspace,
        max_bytes=limits.workspace_bytes - additional_bytes,
        max_files=limits.workspace_files - additional_files,
    )


class _BoundedOutput:
    """Drain a subprocess stream without retaining more than its configured cap."""

    def __init__(self, limit: int, *, sentinel: bytes | None = None) -> None:
        self._limit = max(0, limit)
        self._data = bytearray()
        self._omitted = 0
        self._sentinel = sentinel
        self._sentinel_tail = b""
        self.sentinel_seen = False

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(_OUTPUT_READ_CHUNK):
            if self._sentinel is not None and not self.sentinel_seen:
                combined = self._sentinel_tail + chunk
                self.sentinel_seen = self._sentinel in combined
                tail_length = max(0, len(self._sentinel) - 1)
                self._sentinel_tail = combined[-tail_length:] if tail_length else b""
            remaining = max(0, self._limit - len(self._data))
            self._data.extend(chunk[:remaining])
            self._omitted += max(0, len(chunk) - remaining)

    def text(self) -> str:
        value = bytes(self._data).decode("utf-8", "replace")
        if self._omitted:
            value += f"\n... [truncated, {self._omitted} more bytes]"
        return value


class DockerSandboxRunner:
    """Runs a SandboxSpec through a local Docker-compatible OCI CLI. The
    container is created first and started only after its immutable ID is
    known. The wall-clock timeout is enforced from OUTSIDE the container; on
    expiry that ID is force-removed because killing the attached runtime client
    alone could leave agent code running."""

    def __init__(
        self,
        *,
        docker: str = "docker",
        cleanup_timeout_seconds: float | None = None,
    ) -> None:
        if cleanup_timeout_seconds is None:
            executable = Path(docker).name
            cleanup_timeout_seconds = (
                _PODMAN_CLEANUP_TIMEOUT_SECONDS
                if executable.startswith("podman")
                else _DEFAULT_CLEANUP_TIMEOUT_SECONDS
            )
        if cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be positive")
        self._docker = docker
        self._cleanup_timeout_seconds = cleanup_timeout_seconds

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        if shutil.which(self._docker) is None:
            raise SandboxError(
                f"'{self._docker}' not found on PATH — the Local tier needs "
                "Docker or Podman for code_interpreter"
            )
        name = f"ravana-ci-{_short_id()}"
        startup_marker = f"ravana-started-{uuid.uuid4().hex}"
        return await self._run_container(spec, name, startup_marker)

    async def _run_container(
        self, spec: SandboxSpec, name: str, startup_marker: str
    ) -> SandboxResult:
        try:
            initial_violation = await asyncio.to_thread(
                _workspace_violation,
                spec.workspace,
                max_bytes=spec.limits.workspace_bytes,
                max_files=spec.limits.workspace_files,
            )
        except asyncio.CancelledError as exc:
            # A caller can safely retry because container provisioning has not
            # begun and no agent code could have run.
            raise SandboxCancelledBeforeStart(
                "sandbox cancelled before process creation"
            ) from exc
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise SandboxCancelledBeforeStart(
                "sandbox cancelled before process creation"
            )
        if initial_violation is not None:
            if initial_violation.measurement_failed:
                raise SandboxError(
                    "sandbox workspace could not be measured before start: "
                    f"{initial_violation.message}"
                )
            return SandboxResult(
                exit_code=_WORKSPACE_LIMIT_EXIT_CODE,
                stdout="",
                stderr=initial_violation.message,
            )

        try:
            create_argv = build_docker_argv(
                spec, name=name, startup_marker=startup_marker
            )
        except OSError as exc:
            raise SandboxError(
                f"sandbox workspace could not be prepared ({type(exc).__name__})"
            ) from exc
        run_deadline = (
            asyncio.get_running_loop().time() + spec.limits.timeout_seconds
        )
        create_argv[0] = self._docker
        container_id = await self._create_container(
            create_argv, name, run_deadline
        )
        return await self._start_container(
            spec, container_id, startup_marker, run_deadline
        )

    async def _create_container(
        self, argv: list[str], name: str, run_deadline: float
    ) -> str:
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        )
        try:
            # Shield process creation so cancellation cannot land after the OS
            # spawned the runtime client but before Ravana receives the handle
            # needed to clean it up.
            proc = await asyncio.wait_for(
                asyncio.shield(spawn_task), timeout=self._remaining(run_deadline)
            )
        except asyncio.TimeoutError:
            return await self._finish_expired_create(spawn_task, name)
        except asyncio.CancelledError as exc:
            await self._cancel_pending_spawn(
                spawn_task, name, exc, identity_known=False
            )
            raise SandboxCancelledBeforeStart(
                "sandbox cancelled during container creation"
            ) from exc
        except OSError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                ) from exc
            raise SandboxError(
                f"failed to create the sandbox container ({type(exc).__name__})"
            ) from exc

        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            cancellation = asyncio.CancelledError()
            await self._cancel_pending_spawn(
                spawn_task, name, cancellation, identity_known=False
            )
            raise SandboxCancelledBeforeStart(
                "sandbox cancelled during container creation"
            ) from cancellation

        return await self._wait_for_created_container(
            proc, name, run_deadline
        )

    async def _wait_for_created_container(
        self,
        proc: asyncio.subprocess.Process,
        name: str,
        run_deadline: float,
    ) -> str:
        process_waiter = asyncio.create_task(proc.wait())
        if proc.stdout is None or proc.stderr is None:
            cleanup_error, _, was_cancelled = await self._finish_uninterruptibly(
                proc,
                process_waiter,
                name,
                (),
                identity_known=False,
            )
            if was_cancelled:
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                )
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                f"container creation exposed no output streams{detail}"
            )

        stdout = _BoundedOutput(4096)
        stderr = _BoundedOutput(4096)
        readers = (
            asyncio.create_task(stdout.drain(proc.stdout)),
            asyncio.create_task(stderr.drain(proc.stderr)),
        )
        try:
            if not await self._wait_bounded(process_waiter, run_deadline):
                cleanup_error, output_error, was_cancelled = (
                    await self._finish_uninterruptibly(
                        proc,
                        process_waiter,
                        name,
                        readers,
                        identity_known=False,
                    )
                )
                if was_cancelled:
                    raise SandboxCancelledBeforeStart(
                        "sandbox cancelled during container creation"
                    )
                errors = [
                    error for error in (cleanup_error, output_error) if error
                ]
                detail = f"; {'; '.join(errors)}" if errors else ""
                raise SandboxError(
                    f"sandbox container creation exceeded its deadline{detail}"
                )
            output_error = await self._settle_readers(
                readers, self._new_cleanup_deadline()
            )
        except SandboxCancelledBeforeStart:
            raise
        except asyncio.CancelledError as exc:
            cleanup_error, output_error, _ = await self._finish_uninterruptibly(
                proc,
                process_waiter,
                name,
                readers,
                identity_known=proc.returncode is not None,
            )
            errors = [error for error in (cleanup_error, output_error) if error]
            cancelled = SandboxCancelledBeforeStart(
                "sandbox cancelled during container creation"
            )
            if errors:
                cancelled.add_note(
                    f"container creation cleanup failed: {'; '.join(errors)}"
                )
            raise cancelled from exc

        if output_error is not None:
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                name,
                self._new_cleanup_deadline(),
                identity_known=True,
            )
            if was_cancelled:
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                )
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                f"container creation output failed ({output_error}){detail}"
            )

        return_code = proc.returncode if proc.returncode is not None else -1
        captured_stderr = stderr.text().strip()
        if return_code != 0:
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                name,
                self._new_cleanup_deadline(),
                identity_known=True,
            )
            if was_cancelled:
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                )
            details = [detail for detail in (captured_stderr, cleanup_error) if detail]
            suffix = f": {'; '.join(details)}" if details else ""
            raise SandboxError(
                f"container runtime failed before sandbox start "
                f"(create exit {return_code}){suffix}"
            )

        container_id = stdout.text().strip()
        if not _CONTAINER_ID_PATTERN.fullmatch(container_id):
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                name,
                self._new_cleanup_deadline(),
                identity_known=True,
            )
            if was_cancelled:
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                )
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                f"container creation returned an invalid identity{detail}"
            )
        return container_id

    async def _finish_expired_create(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        name: str,
    ) -> str:
        cleanup_deadline = self._new_cleanup_deadline()
        recovery_deadline = self._spawn_recovery_deadline(cleanup_deadline)
        try:
            proc, spawn_was_cancelled = await self._await_spawn_uninterruptibly(
                spawn_task, recovery_deadline
            )
        except asyncio.TimeoutError:
            self._schedule_late_create_cleanup(spawn_task, name)
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                name, cleanup_deadline, identity_known=False
            )
            current_task = asyncio.current_task()
            if was_cancelled or (
                current_task is not None and current_task.cancelling()
            ):
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                )
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                f"sandbox container creation exceeded its hard deadline{detail}"
            )
        except OSError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled during container creation"
                ) from exc
            raise SandboxError(
                f"failed to create the sandbox container ({type(exc).__name__})"
            ) from exc

        process_waiter = asyncio.create_task(proc.wait())
        readers: tuple[asyncio.Task[None], ...] = ()
        if proc.stdout is not None and proc.stderr is not None:
            discard_stdout = _BoundedOutput(0)
            discard_stderr = _BoundedOutput(0)
            readers = (
                asyncio.create_task(discard_stdout.drain(proc.stdout)),
                asyncio.create_task(discard_stderr.drain(proc.stderr)),
            )
        cleanup_error, output_error, cleanup_was_cancelled = (
            await self._finish_uninterruptibly(
                proc,
                process_waiter,
                name,
                readers,
                deadline=cleanup_deadline,
                identity_known=False,
            )
        )
        if spawn_was_cancelled or cleanup_was_cancelled:
            raise SandboxCancelledBeforeStart(
                "sandbox cancelled during container creation"
            )
        errors = [error for error in (cleanup_error, output_error) if error]
        detail = f"; {'; '.join(errors)}" if errors else ""
        raise SandboxError(
            f"sandbox container creation exceeded its hard deadline{detail}"
        )

    async def _start_container(
        self,
        spec: SandboxSpec,
        container_id: str,
        startup_marker: str,
        run_deadline: float,
    ) -> SandboxResult:
        if self._remaining(run_deadline) <= 0:
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                container_id,
                self._new_cleanup_deadline(),
                identity_known=True,
            )
            current_task = asyncio.current_task()
            if was_cancelled or (
                current_task is not None and current_task.cancelling()
            ):
                raise SandboxCancelledBeforeStart(
                    "sandbox cancelled before command start"
                )
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                f"sandbox deadline expired before command start{detail}"
            )

        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            cleanup_error, _ = await self._cleanup_container_uninterruptibly(
                container_id,
                self._new_cleanup_deadline(),
                identity_known=True,
            )
            cancellation = SandboxCancelledBeforeStart(
                "sandbox cancelled before command start"
            )
            if cleanup_error:
                cancellation.add_note(
                    f"created container cleanup failed: {cleanup_error}"
                )
            raise cancellation

        argv = build_docker_start_argv(container_id)
        argv[0] = self._docker
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        try:
            proc = await asyncio.wait_for(
                asyncio.shield(spawn_task), timeout=self._remaining(run_deadline)
            )
        except asyncio.TimeoutError:
            return await self._finish_expired_spawn(
                spec, spawn_task, container_id, startup_marker
            )
        except asyncio.CancelledError as exc:
            await self._cancel_pending_spawn(spawn_task, container_id, exc)
            raise
        except OSError as exc:
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                container_id,
                self._new_cleanup_deadline(),
                identity_known=True,
            )
            current_task = asyncio.current_task()
            if was_cancelled or (
                current_task is not None and current_task.cancelling()
            ):
                raise asyncio.CancelledError from exc
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                "failed to start the sandbox container "
                f"({type(exc).__name__}){detail}"
            ) from exc

        # A cancellation can race the spawn task's completion: wait_for may
        # return the process handle while the surrounding Task already carries
        # a pending cancellation request. Honor it while the known container ID
        # is still available for bounded cleanup.
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            post_start_cancellation = asyncio.CancelledError()
            await self._cancel_pending_spawn(
                spawn_task, container_id, post_start_cancellation
            )
            raise post_start_cancellation

        return await self._run_spawned_process(
            spec, proc, container_id, startup_marker, run_deadline
        )

    async def _run_spawned_process(
        self,
        spec: SandboxSpec,
        proc: asyncio.subprocess.Process,
        name: str,
        startup_marker: str,
        run_deadline: float,
    ) -> SandboxResult:
        process_waiter = asyncio.create_task(proc.wait())
        if proc.stdout is None or proc.stderr is None:
            cleanup_error, _, was_cancelled = await self._finish_uninterruptibly(
                proc, process_waiter, name, ()
            )
            if was_cancelled:
                raise asyncio.CancelledError
            if cleanup_error is not None:
                raise SandboxOutcomeUnknown(
                    f"sandbox process exposed no output streams and cleanup failed: {cleanup_error}"
                )
            raise SandboxError("sandbox process did not expose output streams")

        stdout = _BoundedOutput(spec.limits.output_bytes)
        stderr = _BoundedOutput(
            spec.limits.output_bytes,
            sentinel=f"{startup_marker}\n".encode("ascii"),
        )
        readers = (
            asyncio.create_task(stdout.drain(proc.stdout)),
            asyncio.create_task(stderr.drain(proc.stderr)),
        )
        workspace_monitor = asyncio.create_task(self._watch_workspace(spec))
        cleanup_started = False
        try:
            done, _ = await asyncio.wait(
                (process_waiter, workspace_monitor),
                timeout=self._remaining(run_deadline),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await self._stop_task(workspace_monitor)
                cleanup_started = True
                return await self._finish_timeout(
                    spec,
                    proc,
                    process_waiter,
                    name,
                    readers,
                    stdout,
                    stderr,
                    startup_marker,
                )

            if workspace_monitor in done:
                violation = workspace_monitor.result()
                cleanup_started = True
                cleanup_error, output_error, was_cancelled = (
                    await self._finish_uninterruptibly(
                        proc, process_waiter, name, readers
                    )
                )
                errors = [
                    error for error in (cleanup_error, output_error) if error
                ]
                if was_cancelled:
                    cancel_error = asyncio.CancelledError()
                    if errors:
                        cancel_error.add_note(
                            f"sandbox quota cleanup failed: {'; '.join(errors)}"
                        )
                    raise cancel_error
                if errors or violation.measurement_failed:
                    detail = "; ".join([violation.message, *errors])
                    raise SandboxOutcomeUnknown(
                        f"sandbox workspace enforcement failed: {detail}"
                    )
                return self._workspace_limit_result(
                    stdout, stderr, startup_marker, violation.message
                )

            await self._stop_task(workspace_monitor)
        except asyncio.CancelledError as exc:
            await self._stop_task(workspace_monitor)
            if not cleanup_started:
                cleanup_error, output_error, _ = await self._finish_uninterruptibly(
                    proc, process_waiter, name, readers
                )
                errors = [error for error in (cleanup_error, output_error) if error]
                if errors:
                    exc.add_note(
                        f"sandbox cancellation cleanup failed: {'; '.join(errors)}"
                    )
            raise
        except BaseException:
            await self._stop_task(workspace_monitor)
            if not cleanup_started:
                await self._finish_uninterruptibly(
                    proc, process_waiter, name, readers
                )
            raise

        output_error = await self._settle_readers(
            readers, self._new_cleanup_deadline()
        )
        if output_error is not None:
            raise SandboxOutcomeUnknown(
                f"sandbox finished but its output could not be captured: {output_error}"
            )

        final_violation = await asyncio.to_thread(
            _workspace_violation,
            spec.workspace,
            max_bytes=spec.limits.workspace_bytes,
            max_files=spec.limits.workspace_files,
        )
        if final_violation is not None:
            if final_violation.measurement_failed:
                raise SandboxOutcomeUnknown(
                    f"sandbox workspace enforcement failed: {final_violation.message}"
                )
            return self._workspace_limit_result(
                stdout, stderr, startup_marker, final_violation.message
            )

        return_code = proc.returncode if proc.returncode is not None else -1
        captured_stdout = stdout.text()
        captured_stderr = _strip_startup_marker(stderr.text(), startup_marker)
        if not stderr.sentinel_seen:
            detail = f": {captured_stderr.strip()}" if captured_stderr.strip() else ""
            cleanup_error, _, was_cancelled = await self._finish_uninterruptibly(
                proc, process_waiter, name, ()
            )
            if was_cancelled:
                raise asyncio.CancelledError
            if cleanup_error is not None:
                raise SandboxOutcomeUnknown(
                    "sandbox exited without startup proof and cleanup failed: "
                    f"{cleanup_error}"
                )
            if return_code == 125:
                raise SandboxError(
                    f"container runtime failed before sandbox start (exit 125){detail}"
                )
            raise SandboxError(
                "container runtime exited before sandbox start "
                f"(exit {return_code}){detail}"
            )
        return SandboxResult(
            exit_code=return_code,
            stdout=captured_stdout,
            stderr=captured_stderr,
        )

    async def _finish_timeout(
        self,
        spec: SandboxSpec,
        proc: asyncio.subprocess.Process,
        process_waiter: asyncio.Task[int],
        name: str,
        readers: tuple[asyncio.Task[None], ...],
        stdout: _BoundedOutput,
        stderr: _BoundedOutput,
        startup_marker: str,
        *,
        deadline: float | None = None,
    ) -> SandboxResult:
        cleanup_error, output_error, was_cancelled = await self._finish_uninterruptibly(
            proc, process_waiter, name, readers, deadline=deadline
        )
        errors = [error for error in (cleanup_error, output_error) if error]
        if was_cancelled:
            cancel_error = asyncio.CancelledError()
            if errors:
                cancel_error.add_note(
                    f"sandbox timeout cleanup failed: {'; '.join(errors)}"
                )
            raise cancel_error
        if errors:
            raise SandboxOutcomeUnknown(
                f"sandbox timed out and cleanup failed: {'; '.join(errors)}"
            )
        if not stderr.sentinel_seen:
            captured = _strip_startup_marker(stderr.text(), startup_marker).strip()
            detail = f": {captured}" if captured else ""
            raise SandboxError(
                f"sandbox timed out before command start{detail}"
            )
        timeout_message = f"timed out after {spec.limits.timeout_seconds}s"
        captured_stderr = _strip_startup_marker(stderr.text(), startup_marker)
        return SandboxResult(
            exit_code=124,
            stdout=stdout.text(),
            stderr=(
                f"{captured_stderr}\n{timeout_message}"
                if captured_stderr
                else timeout_message
            ),
            timed_out=True,
        )

    async def _finish_expired_spawn(
        self,
        spec: SandboxSpec,
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        name: str,
        startup_marker: str,
    ) -> SandboxResult:
        cleanup_deadline = self._new_cleanup_deadline()
        recovery_deadline = self._spawn_recovery_deadline(cleanup_deadline)
        try:
            proc, spawn_was_cancelled = await self._await_spawn_uninterruptibly(
                spawn_task, recovery_deadline
            )
        except asyncio.TimeoutError:
            self._schedule_late_run_cleanup(spawn_task, name)
            cleanup_error, was_cancelled = await self._cleanup_container_uninterruptibly(
                name, cleanup_deadline
            )
            if was_cancelled:
                raise asyncio.CancelledError
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxOutcomeUnknown(
                f"sandbox process creation exceeded its hard deadline{detail}"
            )
        except OSError as exc:
            # The spawn may finish after the run deadline while recovery is
            # waiting for its handle. It is still an infrastructure failure,
            # just like an immediate create_subprocess_exec error, unless the
            # recovery wait consumed cancellation first. Post-spawn
            # cancellation remains fail-closed and takes precedence.
            cleanup_error, cleanup_was_cancelled = (
                await self._cleanup_container_uninterruptibly(
                    name, cleanup_deadline
                )
            )
            current_task = asyncio.current_task()
            if cleanup_was_cancelled or (
                current_task is not None and current_task.cancelling()
            ):
                cancellation = asyncio.CancelledError()
                if cleanup_error:
                    cancellation.add_note(
                        f"created container cleanup failed: {cleanup_error}"
                    )
                raise cancellation from exc
            detail = f"; {cleanup_error}" if cleanup_error else ""
            raise SandboxError(
                f"failed to spawn the sandbox ({type(exc).__name__}){detail}"
            ) from exc

        process_waiter = asyncio.create_task(proc.wait())
        if proc.stdout is None or proc.stderr is None:
            cleanup_error, _, cleanup_was_cancelled = (
                await self._finish_uninterruptibly(
                    proc, process_waiter, name, (), deadline=cleanup_deadline
                )
            )
            if spawn_was_cancelled or cleanup_was_cancelled:
                raise asyncio.CancelledError
            if cleanup_error is not None:
                raise SandboxOutcomeUnknown(
                    f"expired sandbox spawn cleanup failed: {cleanup_error}"
                )
            raise SandboxError("sandbox process did not expose output streams")

        stdout = _BoundedOutput(spec.limits.output_bytes)
        stderr = _BoundedOutput(
            spec.limits.output_bytes,
            sentinel=f"{startup_marker}\n".encode("ascii"),
        )
        readers = (
            asyncio.create_task(stdout.drain(proc.stdout)),
            asyncio.create_task(stderr.drain(proc.stderr)),
        )
        if spawn_was_cancelled:
            await self._finish_uninterruptibly(
                proc,
                process_waiter,
                name,
                readers,
                deadline=cleanup_deadline,
            )
            raise asyncio.CancelledError
        return await self._finish_timeout(
            spec,
            proc,
            process_waiter,
            name,
            readers,
            stdout,
            stderr,
            startup_marker,
            deadline=cleanup_deadline,
        )

    async def _cancel_pending_spawn(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        container: str,
        cancellation: asyncio.CancelledError,
        *,
        identity_known: bool = True,
    ) -> None:
        cleanup_deadline = self._new_cleanup_deadline()
        recovery_deadline = self._spawn_recovery_deadline(cleanup_deadline)
        try:
            proc, _ = await self._await_spawn_uninterruptibly(
                spawn_task, recovery_deadline
            )
        except asyncio.TimeoutError:
            if identity_known:
                self._schedule_late_run_cleanup(spawn_task, container)
            else:
                self._schedule_late_create_cleanup(spawn_task, container)
            cleanup_error, _ = await self._cleanup_container_uninterruptibly(
                container,
                cleanup_deadline,
                identity_known=identity_known,
            )
            detail = cleanup_error or "process creation did not finish"
            cancellation.add_note(
                f"sandbox spawn cancellation cleanup is indeterminate: {detail}"
            )
            return
        except BaseException as spawn_error:
            cancellation.add_note(
                "sandbox spawn did not return a process handle "
                f"({type(spawn_error).__name__})"
            )
            return

        process_waiter = asyncio.create_task(proc.wait())
        readers: tuple[asyncio.Task[None], ...] = ()
        if proc.stdout is not None and proc.stderr is not None:
            discard_stdout = _BoundedOutput(0)
            discard_stderr = _BoundedOutput(0)
            readers = (
                asyncio.create_task(discard_stdout.drain(proc.stdout)),
                asyncio.create_task(discard_stderr.drain(proc.stderr)),
            )
        cleanup_error, output_error, _ = await self._finish_uninterruptibly(
            proc,
            process_waiter,
            container,
            readers,
            deadline=cleanup_deadline,
            identity_known=identity_known,
        )
        errors = [error for error in (cleanup_error, output_error) if error]
        if errors:
            cancellation.add_note(
                f"sandbox spawn cancellation cleanup failed: {'; '.join(errors)}"
            )

    @staticmethod
    def _workspace_limit_result(
        stdout: _BoundedOutput,
        stderr: _BoundedOutput,
        startup_marker: str,
        message: str,
    ) -> SandboxResult:
        captured_stderr = _strip_startup_marker(stderr.text(), startup_marker)
        return SandboxResult(
            exit_code=_WORKSPACE_LIMIT_EXIT_CODE,
            stdout=stdout.text(),
            stderr=(
                f"{captured_stderr}\n{message}" if captured_stderr else message
            ),
        )

    @staticmethod
    async def _watch_workspace(spec: SandboxSpec) -> WorkspaceViolation:
        while True:
            try:
                violation = await asyncio.to_thread(
                    _workspace_violation,
                    spec.workspace,
                    max_bytes=spec.limits.workspace_bytes,
                    max_files=spec.limits.workspace_files,
                )
            except Exception as exc:  # pragma: no cover - defensive executor boundary
                return WorkspaceViolation(
                    f"workspace monitor failed ({type(exc).__name__})",
                    measurement_failed=True,
                )
            if violation is not None:
                return violation
            await asyncio.sleep(_WORKSPACE_POLL_SECONDS)

    @staticmethod
    async def _stop_task(task: asyncio.Task[object]) -> None:
        if not task.done():
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    break
        if not task.cancelled():
            with contextlib.suppress(BaseException):
                task.exception()

    async def _cleanup_process(
        self,
        proc: asyncio.subprocess.Process,
        process_waiter: asyncio.Task[int],
        container: str,
        deadline: float,
        *,
        identity_known: bool,
    ) -> str | None:
        errors: list[str] = []
        client_error = await self._stop_client(proc, process_waiter, deadline)
        if client_error is not None:
            errors.append(client_error)

        # A completed create gives us an immutable ID (or a terminally-resolved
        # name), so one verified absence is final. During an interrupted create,
        # absence alone is not proof: only observing and removing that pending
        # create closes the race.
        remove = (
            self._remove_known_container
            if identity_known
            else self._remove_pending_create
        )
        absence_error = await remove(container, deadline)
        if absence_error is not None:
            errors.append(absence_error)
        return "; ".join(errors) or None

    async def _remove_known_container(
        self, container_id: str, deadline: float
    ) -> str | None:
        """Remove a previously-created immutable identity until it is absent."""
        last_error: str | None = None
        while self._remaining(deadline) > 0:
            _, remove_error = await self._force_remove_container(
                container_id, deadline
            )
            absent, verify_error = await self._container_is_absent(
                container_id, deadline
            )
            if absent:
                return None
            last_error = verify_error or remove_error or (
                "container still exists after forced cleanup"
            )

            delay = min(_CLEANUP_POLL_SECONDS, self._remaining(deadline))
            if delay > 0:
                await asyncio.sleep(delay)
        return last_error or "container cleanup deadline expired"

    async def _remove_pending_create(
        self, name: str, deadline: float
    ) -> str | None:
        """Wait for an interrupted create to become removable by its name."""
        removed = False
        last_error: str | None = None
        while self._remaining(deadline) > 0:
            removed_now, remove_error = await self._force_remove_container(
                name, deadline
            )
            removed = removed or removed_now
            absent, verify_error = await self._container_is_absent(name, deadline)
            if removed and absent:
                return None
            last_error = verify_error or remove_error or (
                "pending container create was not observed before cleanup deadline"
            )
            delay = min(_CLEANUP_POLL_SECONDS, self._remaining(deadline))
            if delay > 0:
                await asyncio.sleep(delay)
        return last_error or "pending container cleanup deadline expired"

    async def _force_remove_container(
        self, container: str, deadline: float
    ) -> tuple[bool, str | None]:
        return_code, detail, command_error = await self._run_docker_control(
            "rm", "--force", container, deadline=deadline
        )
        if command_error is not None:
            return False, f"container cleanup {command_error}"
        if return_code == 0:
            return True, None
        if return_code != 0 and not _is_missing_container_error(detail):
            suffix = f": {detail}" if detail else ""
            return False, f"container cleanup exited {return_code}{suffix}"
        return False, None

    async def _container_is_absent(
        self, name: str, deadline: float
    ) -> tuple[bool, str | None]:
        return_code, detail, command_error = await self._run_docker_control(
            "container", "inspect", name, deadline=deadline
        )
        if command_error is not None:
            return False, f"container cleanup verification {command_error}"
        if return_code == 0:
            return False, None
        if _is_missing_container_error(detail):
            return True, None
        suffix = f": {detail}" if detail else ""
        return False, f"container cleanup verification exited {return_code}{suffix}"

    async def _run_docker_control(
        self, *args: str, deadline: float
    ) -> tuple[int | None, str, str | None]:
        if self._remaining(deadline) <= 0:
            return None, "", "deadline expired"
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                self._docker,
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        try:
            proc = await asyncio.wait_for(
                asyncio.shield(spawn_task), timeout=self._remaining(deadline)
            )
        except asyncio.TimeoutError:
            self._schedule_late_control_cleanup(spawn_task)
            return None, "", "process creation timed out"
        except OSError as exc:
            return None, "", f"failed to start ({type(exc).__name__})"

        if proc.stderr is None:
            return None, "", "exposed no stderr stream"
        captured_stderr = _BoundedOutput(4096)
        stderr_reader = asyncio.create_task(captured_stderr.drain(proc.stderr))
        waiter = asyncio.create_task(proc.wait())
        if not await self._wait_bounded(waiter, deadline):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            if not await self._wait_bounded(waiter, deadline):
                waiter.cancel()
            await self._settle_readers((stderr_reader,), deadline)
            return None, captured_stderr.text().strip(), "command timed out"

        output_error = await self._settle_readers((stderr_reader,), deadline)
        if output_error is not None:
            return None, captured_stderr.text().strip(), f"output failed ({output_error})"
        return proc.returncode, captured_stderr.text().strip(), None

    async def _stop_client(
        self,
        proc: asyncio.subprocess.Process,
        process_waiter: asyncio.Task[int],
        deadline: float,
    ) -> str | None:
        if proc.returncode is not None:
            return None
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        loop = asyncio.get_running_loop()
        grace_seconds = min(1.0, self._cleanup_timeout_seconds / 4)
        graceful_deadline = min(deadline, loop.time() + grace_seconds)
        if await self._wait_bounded(process_waiter, graceful_deadline):
            return None
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        if await self._wait_bounded(process_waiter, deadline):
            return None
        process_waiter.cancel()
        return "container runtime client did not exit after kill"

    async def _wait_bounded(
        self, waiter: asyncio.Task[int], deadline: float
    ) -> bool:
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter), timeout=self._remaining(deadline)
            )
        except asyncio.TimeoutError:
            return False
        return True

    @staticmethod
    def _schedule_late_control_cleanup(
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
    ) -> None:
        async def reap() -> None:
            try:
                proc = await asyncio.shield(spawn_task)
            except BaseException:
                return
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=1)

        asyncio.create_task(reap())

    def _schedule_late_run_cleanup(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        container_id: str,
    ) -> None:
        async def reap() -> None:
            try:
                proc = await asyncio.shield(spawn_task)
            except BaseException:
                return
            process_waiter = asyncio.create_task(proc.wait())
            readers: tuple[asyncio.Task[None], ...] = ()
            if proc.stdout is not None and proc.stderr is not None:
                discard_stdout = _BoundedOutput(0)
                discard_stderr = _BoundedOutput(0)
                readers = (
                    asyncio.create_task(discard_stdout.drain(proc.stdout)),
                    asyncio.create_task(discard_stderr.drain(proc.stderr)),
                )
            with contextlib.suppress(BaseException):
                await self._finish_uninterruptibly(
                    proc, process_waiter, container_id, readers
                )

        asyncio.create_task(reap())

    def _schedule_late_create_cleanup(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        name: str,
    ) -> None:
        async def reap() -> None:
            try:
                proc = await asyncio.shield(spawn_task)
            except BaseException:
                return
            process_waiter = asyncio.create_task(proc.wait())
            readers: tuple[asyncio.Task[None], ...] = ()
            if proc.stdout is not None and proc.stderr is not None:
                discard_stdout = _BoundedOutput(0)
                discard_stderr = _BoundedOutput(0)
                readers = (
                    asyncio.create_task(discard_stdout.drain(proc.stdout)),
                    asyncio.create_task(discard_stderr.drain(proc.stderr)),
                )
            with contextlib.suppress(BaseException):
                await self._finish_uninterruptibly(
                    proc,
                    process_waiter,
                    name,
                    readers,
                    identity_known=False,
                )

        asyncio.create_task(reap())

    @staticmethod
    async def _await_spawn_uninterruptibly(
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        deadline: float,
    ) -> tuple[asyncio.subprocess.Process, bool]:
        was_cancelled = False
        while True:
            try:
                proc = await asyncio.wait_for(
                    asyncio.shield(spawn_task),
                    timeout=DockerSandboxRunner._remaining(deadline),
                )
                return proc, was_cancelled
            except asyncio.CancelledError:
                if spawn_task.cancelled():
                    raise
                was_cancelled = True

    async def _cleanup_container_uninterruptibly(
        self,
        container: str,
        deadline: float,
        *,
        identity_known: bool = True,
    ) -> tuple[str | None, bool]:
        remove = (
            self._remove_known_container
            if identity_known
            else self._remove_pending_create
        )
        cleanup_task = asyncio.create_task(
            remove(container, deadline)
        )
        was_cancelled = False
        while True:
            try:
                return await asyncio.shield(cleanup_task), was_cancelled
            except asyncio.CancelledError:
                if cleanup_task.cancelled():
                    raise
                was_cancelled = True

    async def _finish_uninterruptibly(
        self,
        proc: asyncio.subprocess.Process,
        process_waiter: asyncio.Task[int],
        name: str,
        readers: tuple[asyncio.Task[None], ...],
        *,
        deadline: float | None = None,
        identity_known: bool = True,
    ) -> tuple[str | None, str | None, bool]:
        deadline = deadline or self._new_cleanup_deadline()

        async def finish() -> tuple[str | None, str | None]:
            cleanup_error = await self._cleanup_process(
                proc,
                process_waiter,
                name,
                deadline,
                identity_known=identity_known,
            )
            output_error = (
                await self._settle_readers(readers, deadline) if readers else None
            )
            return cleanup_error, output_error

        finish_task = asyncio.create_task(finish())
        was_cancelled = False
        while True:
            try:
                cleanup_error, output_error = await asyncio.shield(finish_task)
                return cleanup_error, output_error, was_cancelled
            except asyncio.CancelledError:
                if finish_task.cancelled():
                    raise
                was_cancelled = True

    def _spawn_recovery_deadline(self, cleanup_deadline: float) -> float:
        loop = asyncio.get_running_loop()
        return loop.time() + self._remaining(cleanup_deadline) / 2

    async def _settle_readers(
        self, readers: tuple[asyncio.Task[None], ...], deadline: float
    ) -> str | None:
        done, pending = await asyncio.wait(
            readers, timeout=self._remaining(deadline)
        )
        if pending:
            for reader in pending:
                reader.cancel()
            await asyncio.wait(pending, timeout=self._remaining(deadline))
            return "output streams did not close within the cleanup deadline"
        for reader in done:
            if reader.cancelled():
                return "output stream reader was cancelled"
            if error := reader.exception():
                return type(error).__name__
        return None

    def _new_cleanup_deadline(self) -> float:
        return asyncio.get_running_loop().time() + self._cleanup_timeout_seconds

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - asyncio.get_running_loop().time())


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


def _is_missing_container_error(detail: str) -> bool:
    lowered = detail.lower()
    return "no such container" in lowered or "no such object" in lowered


def _strip_startup_marker(stderr: str, marker: str) -> str:
    lines = stderr.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.rstrip("\r\n") == marker:
            del lines[index]
            break
    return "".join(lines)
