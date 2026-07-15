"""Sandbox execution boundary for the `code_interpreter` toolkit (§8, §10.1).

`code_interpreter` runs agent-authored code — the highest-blast-radius thing in
the system — so execution is isolated behind a `SandboxRunner` the handler is
handed. The Local/Embedded tier (§10.1) backs it with a local Docker container;
hosted tiers swap in a managed provider (E2B/Modal) behind this same interface
(§8 "hidden behind the manifest, a reversible implementation detail").

The security posture is enforced in `build_docker_argv` (a pure function, so it
is exhaustively unit-testable without a Docker daemon): §8's mandate — no
default network egress, a bind mount scoped strictly to that run's workspace and
nothing else, hard per-invocation resource quotas, and a filesystem never
treated as durable (`--rm`). `DockerSandboxRunner` is the thin process wrapper
that actually spawns it; tests inject a fake `SandboxRunner`, so nothing here
requires Docker to be present.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
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
    """The sandbox itself could not run the code — a Docker daemon that's absent
    or unreachable, an image that won't pull, a spawn failure. This is
    infrastructure (§3.6 "sandbox cold-start"), NOT the agent's code failing;
    the handler maps it to a TRANSIENT ToolkitError so the engine retries."""


class SandboxOutcomeUnknown(SandboxError):
    """The sandbox started, but cleanup/output failure hid its final outcome.

    Retrying this invocation is unsafe because agent code may already have
    mutated the workspace.
    """


class SandboxRunner(Protocol):
    async def run(self, spec: SandboxSpec) -> SandboxResult: ...


# Root filesystem is read-only; only the workspace mount and a small tmpfs are
# writable, so nothing the code does persists beyond the workspace or the run.
_TMPFS_SIZE = "64m"


def build_docker_argv(
    spec: SandboxSpec, *, name: str, startup_marker: str
) -> list[str]:
    """The `docker run` argv enforcing §8's isolation. Pure and deterministic so
    every security flag is asserted in tests without invoking Docker:

    - `--rm`: the container filesystem is never durable (§8).
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
    limits = spec.limits
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
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


_OUTPUT_READ_CHUNK = 64 * 1024
_DEFAULT_CLEANUP_TIMEOUT_SECONDS = 10.0


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
    """Runs a SandboxSpec in a local Docker container (§10.1's Local tier). The
    wall-clock timeout is enforced from OUTSIDE the container — on expiry the
    container is force-removed by name, because killing the `docker run`
    client process alone would leave the container running."""

    def __init__(
        self,
        *,
        docker: str = "docker",
        cleanup_timeout_seconds: float = _DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    ) -> None:
        if cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be positive")
        self._docker = docker
        self._cleanup_timeout_seconds = cleanup_timeout_seconds

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        if shutil.which(self._docker) is None:
            raise SandboxError(f"'{self._docker}' not found on PATH — the Local tier needs Docker for code_interpreter")
        name = f"ravana-ci-{_short_id()}"
        startup_marker = f"ravana-started-{uuid.uuid4().hex}"
        return await self._run_container(spec, name, startup_marker)

    async def _run_container(
        self, spec: SandboxSpec, name: str, startup_marker: str
    ) -> SandboxResult:
        argv = build_docker_argv(
            spec, name=name, startup_marker=startup_marker
        )
        argv[0] = self._docker
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        )
        try:
            # Shield process creation so cancellation cannot land after the OS
            # spawned Docker but before Ravana receives the handle needed to
            # clean it up.
            proc = await asyncio.shield(spawn_task)
        except asyncio.CancelledError as exc:
            try:
                proc = await self._await_spawn_uninterruptibly(spawn_task)
            except BaseException as spawn_error:
                exc.add_note(
                    f"sandbox spawn did not return a process handle ({type(spawn_error).__name__})"
                )
                raise exc

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
                proc, process_waiter, name, readers
            )
            errors = [error for error in (cleanup_error, output_error) if error]
            if errors:
                exc.add_note(f"sandbox spawn cancellation cleanup failed: {'; '.join(errors)}")
            raise
        except OSError as exc:
            raise SandboxError(f"failed to spawn the sandbox ({type(exc).__name__})") from exc

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
        try:
            await asyncio.wait_for(
                asyncio.shield(process_waiter),
                timeout=spec.limits.timeout_seconds,
            )
        except asyncio.TimeoutError:
            cleanup_error, output_error, was_cancelled = await self._finish_uninterruptibly(
                proc, process_waiter, name, readers
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
            timeout_message = f"timed out after {spec.limits.timeout_seconds}s"
            captured_stderr = _strip_startup_marker(stderr.text(), startup_marker)
            return SandboxResult(
                exit_code=124,
                stdout=stdout.text(),
                stderr=f"{captured_stderr}\n{timeout_message}" if captured_stderr else timeout_message,
                timed_out=True,
            )
        except asyncio.CancelledError as exc:
            cleanup_error, output_error, _ = await self._finish_uninterruptibly(
                proc, process_waiter, name, readers
            )
            errors = [error for error in (cleanup_error, output_error) if error]
            if errors:
                exc.add_note(f"sandbox cancellation cleanup failed: {'; '.join(errors)}")
            raise
        except BaseException:
            await self._finish_uninterruptibly(proc, process_waiter, name, readers)
            raise

        output_error = await self._settle_readers(
            readers, self._new_cleanup_deadline()
        )
        if output_error is not None:
            raise SandboxOutcomeUnknown(
                f"sandbox finished but its output could not be captured: {output_error}"
            )
        return_code = proc.returncode if proc.returncode is not None else -1
        captured_stdout = stdout.text()
        captured_stderr = _strip_startup_marker(stderr.text(), startup_marker)
        if return_code == 125 and not stderr.sentinel_seen:
            detail = f": {captured_stderr.strip()}" if captured_stderr.strip() else ""
            raise SandboxError(f"docker run failed before sandbox start (exit 125){detail}")
        return SandboxResult(
            exit_code=return_code,
            stdout=captured_stdout,
            stderr=captured_stderr,
        )

    async def _cleanup_process(
        self,
        proc: asyncio.subprocess.Process,
        process_waiter: asyncio.Task[int],
        name: str,
        deadline: float,
    ) -> str | None:
        errors: list[str] = []
        client_error = await self._stop_client(proc, process_waiter, deadline)
        if client_error is not None:
            errors.append(client_error)

        # Stop the client first. During image pull/create, removing by name
        # before the client exits can report "no such container" and then let a
        # late daemon-side create escape the timeout boundary.
        remove_error = await self._force_remove_container(name, deadline)
        absent, verify_error = await self._container_is_absent(name, deadline)
        if not absent and verify_error is None:
            # A create request already accepted by the daemon may finish just
            # after the first remove. Remove once more, then verify the final
            # state rather than trusting the race-prone first lookup.
            retry_error = await self._force_remove_container(name, deadline)
            remove_error = retry_error or remove_error
            absent, verify_error = await self._container_is_absent(name, deadline)

        if not absent:
            if remove_error is not None:
                errors.append(remove_error)
            if verify_error is not None:
                errors.append(verify_error)
            else:
                errors.append("container still exists after forced cleanup")
        return "; ".join(errors) or None

    async def _force_remove_container(
        self, name: str, deadline: float
    ) -> str | None:
        # Removing by name stops the daemon-owned container too. Merely killing
        # the `docker run` client can leave the agent code running in Docker.
        return_code, detail, command_error = await self._run_docker_control(
            "rm", "--force", name, deadline=deadline
        )
        if command_error is not None:
            return f"docker cleanup {command_error}"
        if return_code != 0 and not _is_missing_container_error(detail):
            suffix = f": {detail}" if detail else ""
            return f"docker cleanup exited {return_code}{suffix}"
        return None

    async def _container_is_absent(
        self, name: str, deadline: float
    ) -> tuple[bool, str | None]:
        return_code, detail, command_error = await self._run_docker_control(
            "container", "inspect", name, deadline=deadline
        )
        if command_error is not None:
            return False, f"docker cleanup verification {command_error}"
        if return_code == 0:
            return False, None
        if _is_missing_container_error(detail):
            return True, None
        suffix = f": {detail}" if detail else ""
        return False, f"docker cleanup verification exited {return_code}{suffix}"

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
        return "docker client did not exit after kill"

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

    @staticmethod
    async def _await_spawn_uninterruptibly(
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
    ) -> asyncio.subprocess.Process:
        while True:
            try:
                return await asyncio.shield(spawn_task)
            except asyncio.CancelledError:
                if spawn_task.cancelled():
                    raise

    async def _finish_uninterruptibly(
        self,
        proc: asyncio.subprocess.Process,
        process_waiter: asyncio.Task[int],
        name: str,
        readers: tuple[asyncio.Task[None], ...],
    ) -> tuple[str | None, str | None, bool]:
        deadline = self._new_cleanup_deadline()

        async def finish() -> tuple[str | None, str | None]:
            cleanup_error = await self._cleanup_process(
                proc, process_waiter, name, deadline
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
