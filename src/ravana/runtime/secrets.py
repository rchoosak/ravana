"""Secret resolution for `auth_ref` (toolkit) and `llm.api_key_ref` (agent),
which are always pointers, never raw secrets (§8). Phase 0b/Local tier reads
them from the environment; §8's Vault/KMS backing is a Phase 2 item. The
`secrets://name` scheme maps to the env var `RAVANA_SECRET_<NAME>`.

Resolved secrets are never returned in a form intended for logging — callers
inject them into a request and drop them; §8's log-redaction backstop lives
at the logging layer.
"""

from __future__ import annotations

from typing import Protocol

_SCHEME = "secrets://"

# §8's redaction backstop: every secret value that enters the process through
# ResolvedSecret is remembered here, so redact_secrets() can scrub it from any
# text bound for persistence or logs — an SDK exception that embeds the key we
# injected into it, say. Process-local; the values already live in memory on
# the clients themselves.
_KNOWN_SECRET_VALUES: set[str] = set()


class SecretResolver(Protocol):
    def resolve(self, ref: str) -> "ResolvedSecret": ...


class SecretNotFound(Exception):
    pass


class ResolvedSecret:
    """A resolved credential as a value object — the ONE place the rules
    about plaintext secrets live, instead of scattering them per call site:

    - validates non-empty at construction (an empty credential would slip
      past truthiness gates and silently swap in a different ambient key);
    - never reveals itself: repr/str show a redaction marker, so a debug log
      or pytest assertion diff cannot leak it (§8);
    - registers its value for redact_secrets(), §8's log/persistence backstop;
    - equality/hash by value, so client caches keyed on the credential
      behave correctly across re-resolution;
    - `.value()` is the single, intentional access to the plaintext.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str):
        if not value or not value.strip():
            raise ValueError("refusing an empty credential")
        self._value = value
        _KNOWN_SECRET_VALUES.add(value)

    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "ResolvedSecret('***')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ResolvedSecret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


def redact_secrets(text: str) -> str:
    """§8's ACTIVE redaction backstop ("logging must actively redact ...").
    Replaces every known resolved-secret value appearing in `text` — applied
    wherever free-form error text crosses into persistence (node_execution
    .error, the tool ledger) or the log stream, because an SDK/HTTP exception
    can echo the very credential the runtime injected into it."""
    for value in _KNOWN_SECRET_VALUES:
        if value in text:
            text = text.replace(value, "***REDACTED***")
    return text


def ensure_resolved(value: "ResolvedSecret | str") -> "ResolvedSecret":
    """Normalize a resolver's return: the protocol says ResolvedSecret, but a
    custom resolver may still hand back a raw str — wrap it so the invariants
    (non-empty, redaction registration) hold regardless."""
    return value if isinstance(value, ResolvedSecret) else ResolvedSecret(value)


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
