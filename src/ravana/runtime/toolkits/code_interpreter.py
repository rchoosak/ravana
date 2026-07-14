"""`code_interpreter` toolkit (§1.7, §8, §10.1). The agent submits code; the
handler writes it into THIS run's isolated workspace and executes it in a
sandbox with §8's isolation (no network, scoped mount, hard quotas).

Security is layered:
  - The workspace is `runs/<run_id>/workspace` — per run, never the parent
    project (§10.1). The handler locates it from `run_id` at call time (the
    registry/gateway is built before the run exists) and refuses a path that
    escapes the runs dir.
  - The agent's `filename` is forced to a bare name (no `/`, no `..`), so the
    written script can't land outside the workspace.
  - Actual isolation (network/mount/quotas) is the SandboxRunner's job
    (sandbox.py) — injectable, so this handler is testable without Docker.

The code *running* is a side effect on the workspace filesystem, so the call is
side-effecting: RavanaToolExecutor dedupes a retried invocation on its logical
key rather than re-running it. A non-zero exit (or a timeout) is a normal
RESULT fed back to the model — only the sandbox being unable to run at all
(SandboxError) is a TRANSIENT ToolkitError the engine retries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.toolkits.sandbox import (
    DockerSandboxRunner,
    SandboxError,
    SandboxLimits,
    SandboxResult,
    SandboxRunner,
    SandboxSpec,
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
_OUTPUT_LIMIT = 10_000  # per stream, chars — a runaway print mustn't flood the transcript


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
        self._limits = SandboxLimits(
            memory_mb=_clamp_int(config.get("memory_mb", 2048), 64, _MAX_MEMORY_MB),
            cpus=_clamp_float(config.get("cpus", 2.0), 0.1, _MAX_CPUS),
            timeout_seconds=_clamp_int(config.get("timeout_seconds", 60), 1, _MAX_TIMEOUT_S),
        )
        # §8: no default egress. Only an explicit config opt-in flips it (a real
        # per-host allow-list is a later enhancement; today it's all-or-nothing).
        self._network = bool(config.get("network", False))
        self._runs_dir = runs_dir
        self._runner = runner or DockerSandboxRunner()
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
        (workspace / filename).write_text(str(arguments["code"]), encoding="utf-8")

        args = [str(a) for a in arguments.get("args", [])]
        spec = SandboxSpec(
            image=self._image,
            argv=[*self._interp, filename, *args],
            workspace=workspace,
            limits=self._limits,
            network=self._network,
        )
        try:
            result = await self._runner.run(spec)
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
        runs_dir = self._runs_dir.resolve()
        workspace = (runs_dir / run_id / "workspace").resolve()
        # §10.1: the mount must stay under this run's dir — never the parent
        # project. run_id is engine-generated (a UUID), but verify rather than
        # trust, so a bad id can't walk the workspace outside runs_dir.
        if not workspace.is_relative_to(runs_dir):
            raise ToolkitError(f"code_interpreter: refusing a workspace path outside the runs dir ({run_id!r})", kind=ToolFailureKind.FATAL)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace


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


def _format_result(result: SandboxResult) -> str:
    parts = [f"exit_code: {result.exit_code}"]
    if result.timed_out:
        parts.append("status: timed out")
    if result.stdout:
        parts.append("stdout:\n" + _truncate(result.stdout))
    if result.stderr:
        parts.append("stderr:\n" + _truncate(result.stderr))
    return "\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[:_OUTPUT_LIMIT] + f"\n… [truncated, {len(text) - _OUTPUT_LIMIT} more chars]"


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
