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


@dataclass(frozen=True)
class SandboxSpec:
    image: str
    argv: list[str]  # the command run INSIDE the container (e.g. ["python", "main.py"])
    workspace: Path  # host dir bind-mounted read-write at /workspace — the ONLY mount
    limits: SandboxLimits = SandboxLimits()
    network: bool = False  # §8: no default egress; True only for an explicit allow-list (future)


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


class SandboxRunner(Protocol):
    async def run(self, spec: SandboxSpec) -> SandboxResult: ...


# Root filesystem is read-only; only the workspace mount and a small tmpfs are
# writable, so nothing the code does persists beyond the workspace or the run.
_TMPFS_SIZE = "64m"


def build_docker_argv(spec: SandboxSpec, *, name: str) -> list[str]:
    """The `docker run` argv enforcing §8's isolation. Pure and deterministic so
    every security flag is asserted in tests without invoking Docker:

    - `--rm`: the container filesystem is never durable (§8).
    - `--network none` (unless `spec.network`): no default egress (§8).
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
        "bridge" if spec.network else "none",
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
        *spec.argv,
    ]
    return argv


class DockerSandboxRunner:
    """Runs a SandboxSpec in a local Docker container (§10.1's Local tier). The
    wall-clock timeout is enforced from OUTSIDE the container — on expiry the
    container is `docker kill`ed by name, because killing the `docker run`
    client process alone would leave the container running."""

    def __init__(self, *, docker: str = "docker") -> None:
        self._docker = docker

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        if shutil.which(self._docker) is None:
            raise SandboxError(f"'{self._docker}' not found on PATH — the Local tier needs Docker for code_interpreter")
        name = f"ravana-ci-{_short_id()}"
        argv = build_docker_argv(spec, name=name)
        argv[0] = self._docker
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            raise SandboxError(f"failed to spawn the sandbox ({type(exc).__name__})") from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=spec.limits.timeout_seconds)
        except asyncio.TimeoutError:
            await self._kill(name)
            with contextlib.suppress(Exception):
                await proc.wait()
            return SandboxResult(exit_code=124, stdout="", stderr=f"timed out after {spec.limits.timeout_seconds}s", timed_out=True)
        return SandboxResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
        )

    async def _kill(self, name: str) -> None:
        # Best-effort: killing the docker CLIENT alone leaves the container up,
        # so kill it by name; a failure here must not mask the timeout result.
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                self._docker, "kill", name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await asyncio.wait_for(killer.wait(), timeout=10)


def _short_id() -> str:
    return uuid.uuid4().hex[:12]
