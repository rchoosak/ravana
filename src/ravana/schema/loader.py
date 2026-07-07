from __future__ import annotations

from pathlib import Path

import yaml

from ravana.schema.models import WorkflowDoc


def load_workflow_yaml(path: str | Path) -> WorkflowDoc:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return WorkflowDoc.model_validate(raw)
