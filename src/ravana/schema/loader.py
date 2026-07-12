from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ravana.schema.models import WorkflowDoc


class WorkflowValidationError(Exception):
    """A workflow file failed schema validation. Deliberately NOT the pydantic
    ValidationError: that object's structured surfaces (`.errors()`, `.json()`)
    embed the raw `input` values even when `hide_input_in_errors` suppresses
    them in `str()` — and a workflow file can have a raw secret pasted exactly
    where the §8 pointer validators reject it. This wrapper carries only the
    already-redacted message (field paths + reasons), so no consumer of the
    load boundary can reach the pasted value."""


def load_workflow_yaml(path: str | Path) -> WorkflowDoc:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # str(exc) honors hide_input_in_errors (WorkflowDoc sets it), so the
    # message names the offending field without echoing its value. Capture the
    # message inside the except, but RAISE outside it: raising within the
    # handler would set the wrapper's __context__ to the ValidationError (whose
    # .errors()/.json() still hold the raw input) — `from None` only clears
    # __cause__, not __context__. Raising after the block leaves both None.
    message: str | None = None
    try:
        return WorkflowDoc.model_validate(raw)
    except ValidationError as exc:
        message = str(exc)
    raise WorkflowValidationError(message)
