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


class SecretResolver(Protocol):
    def resolve(self, ref: str) -> str: ...


class SecretNotFound(Exception):
    pass


class EnvSecretResolver:
    def __init__(self, environ: dict[str, str] | None = None):
        # Injectable for tests; defaults to os.environ at call time.
        self._environ = environ

    def resolve(self, ref: str) -> str:
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
        value = environ[env_key]
        if not value.strip():
            # Set-but-empty is NOT a usable secret. Returning "" would make
            # truthiness-gated consumers silently swap in a DIFFERENT
            # credential (the SDK's ambient env key) or send unauthenticated
            # requests — fail closed instead, same as missing.
            raise SecretNotFound(f"secret '{ref}' is set but empty ({env_key}) — refusing an empty credential")
        return value


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)
