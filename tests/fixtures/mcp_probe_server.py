"""A real MCP stdio server for the toolkit tests.

Real subprocess rather than a mocked session: the things worth testing here —
tool discovery, pinning, the anyio task-affinity constraint, subprocess
shutdown — are exactly the ones a mock would define away.
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ravana-probe")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def echo_env(name: str) -> str:
    """Echo an environment variable the client passed to this server."""
    return os.environ.get(name, "(unset)")


@mcp.tool()
def auth_is_set() -> bool:
    """Report whether the dispatch credential reached the child process."""
    return bool(os.environ.get("MCP_AUTH_TOKEN"))


@mcp.tool()
def current_directory() -> str:
    """Report the server process working directory."""
    return os.getcwd()


@mcp.tool()
def explode() -> str:
    """Always fail, to exercise the tool-error path."""
    raise ValueError("tool blew up")


if __name__ == "__main__":
    if os.environ.get("RAVANA_PROBE_REFUSE_START") == "1":
        sys.exit(3)
    if os.environ.get("RAVANA_PROBE_ECHO_AUTH_STDERR") == "1":
        print(os.environ.get("MCP_AUTH_TOKEN", ""), file=sys.stderr, flush=True)
    mcp.run()
