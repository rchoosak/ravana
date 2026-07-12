"""Secret resolution for `auth_ref` (toolkit) and `llm.api_key_ref` (agent),
which are always pointers, never raw secrets (§8). Phase 0b/Local tier reads
them from the environment; §8's Vault/KMS backing is a Phase 2 item. The
`secrets://name` scheme maps to the env var `RAVANA_SECRET_<NAME>`.

Resolved secrets are never returned in a form intended for logging — callers
inject them into a request and drop them; §8's log-redaction backstop lives
at the logging layer.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, TypeVar, cast

_SCHEME = "secrets://"
_T = TypeVar("_T")


class SecretResolver(Protocol):
    def resolve(self, ref: str) -> "ResolvedSecret": ...


class SecretNotFound(Exception):
    pass


class SecretLeakError(Exception):
    """Credential material crossed an output seam; fail closed before persist."""


class ResolvedSecret:
    """A resolved credential as a pure value object — the ONE place the rules
    about a plaintext credential live, instead of scattering them per call
    site:

    - validates non-empty at construction (an empty credential would slip
      past truthiness gates and silently swap in a different ambient key);
    - never reveals itself: repr/str show a redaction marker, so a debug log
      or pytest assertion diff cannot leak it (§8);
    - equality/hash by value, so client caches keyed on the credential behave
      correctly across re-resolution;
    - `.value()` is the single, intentional access to the plaintext.

    Deliberately holds NO process-global state: it does not register itself in
    a redaction set (the old design coupled the value object to logging, gave
    tests hidden shared state, and pinned rotated keys in memory forever). §8's
    active backstop is `redact_secrets`, which works on *patterns*, not a
    registry of every value ever resolved."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        if not value or not value.strip():
            raise ValueError("refusing an empty credential")
        self._value = value

    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "ResolvedSecret('***')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ResolvedSecret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


# §8's active redaction backstop: "logging must actively redact anything
# matching a known secret PATTERN as a backstop." Pattern-based (not a registry
# of resolved values) so it also catches credentials the runtime never
# resolved itself — an SDK's own env key echoed in an exception, a token in a
# resolver error raised before any ResolvedSecret was built. Longest
# alternatives first (sk-ant- before sk-) so the more specific match wins.
_SECRET_PATTERNS = re.compile(
    r"Bearer\s+[A-Za-z0-9._~+/=\-]+"           # Authorization: Bearer <token>
    r"|sk-ant-[A-Za-z0-9_\-]+"                  # Anthropic
    r"|sk-[A-Za-z0-9_\-]+"                      # OpenAI
    r"|gh[posur]_[A-Za-z0-9]+"                  # GitHub PAT/OAuth/…
    r"|github_pat_[A-Za-z0-9_]+"                # GitHub fine-grained PAT
)


def redact_secrets(text: str, *, values: tuple[str, ...] = ()) -> str:
    """Scrub secrets from free-form text bound for a log or persisted error.
    Two layers: (1) any explicitly-known `values` — a caller with the exact
    credential in scope (an adapter wrapping an SDK exception) passes it, and
    they are replaced longest-first so a value that is a substring of another
    can't leave a partial leak; (2) the §8 known-pattern sweep, which covers
    credentials no caller had in hand. A value with no recognizable pattern
    that no caller declares is the documented gap of a pattern backstop."""
    for value in sorted({v for v in values if v}, key=len, reverse=True):
        text = text.replace(value, "***REDACTED***")
    return _SECRET_PATTERNS.sub("***REDACTED***", text)


def redact_data(value: _T, *, values: tuple[str, ...] = ()) -> _T:
    """Recursively scrub strings in JSON-like structured data."""
    if isinstance(value, str):
        return cast(_T, redact_secrets(value, values=values))
    if isinstance(value, dict):
        return cast(
            _T,
            {
                redact_data(key, values=values): redact_data(item, values=values)
                for key, item in value.items()
            },
        )
    if isinstance(value, list):
        return cast(_T, [redact_data(item, values=values) for item in value])
    if isinstance(value, tuple):
        return cast(_T, tuple(redact_data(item, values=values) for item in value))
    return value


def contains_secret(value: Any, *, values: tuple[str, ...] = ()) -> bool:
    """Whether a JSON-like value contains an exact or known-pattern secret."""
    if isinstance(value, str):
        return redact_secrets(value, values=values) != value
    if isinstance(value, dict):
        return any(
            contains_secret(key, values=values) or contains_secret(item, values=values)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_secret(item, values=values) for item in value)
    return False


def ensure_secret_free(value: Any, *, context: str, values: tuple[str, ...] = ()) -> None:
    """Reject credential-bearing output without changing its domain value."""
    if contains_secret(value, values=values):
        raise SecretLeakError(f"{context} contained credential material; output rejected")


def redact_record(record: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact a structured log record, including nested extras."""
    return redact_data(record)


class EnvSecretResolver:
    def __init__(self, environ: dict[str, str] | None = None):
        # Injectable for tests; defaults to os.environ at call time.
        self._environ = environ

    def resolve(self, ref: str) -> ResolvedSecret:
        if not ref.startswith(_SCHEME):
            raise SecretNotFound(f"auth_ref '{ref}' must use the {_SCHEME} scheme")
        name = ref[len(_SCHEME):]
        env_key = f"RAVANA_SECRET_{name.upper()}"
        environ = self._environ if self._environ is not None else _os_environ()
        if env_key not in environ:
            raise SecretNotFound(
                f"secret '{ref}' not found — set {env_key} in the environment "
                "(Vault/KMS backing is a Phase 2 item, §8)"
            )
        try:
            # ResolvedSecret owns the non-empty invariant (set-but-empty would
            # make truthiness-gated consumers silently swap in a DIFFERENT
            # ambient credential); this just adds the env context to the error.
            return ResolvedSecret(environ[env_key])
        except ValueError as exc:
            raise SecretNotFound(f"secret '{ref}' is set but empty ({env_key}) — refusing an empty credential") from exc


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)
