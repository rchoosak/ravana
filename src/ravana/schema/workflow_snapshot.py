"""Canonical, omission-preserving serialization for validated workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ravana.schema.models import WorkflowDoc
from ravana.schema.util import dumps, loads


class WorkflowSnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class WorkflowSnapshot:
    payload: dict[str, Any]

    @classmethod
    def from_doc(cls, doc: WorkflowDoc) -> "WorkflowSnapshot":
        # exclude_unset preserves the compiler's distinction between an omitted
        # node policy (inherit from the agent) and an explicit null (clear it).
        return cls(doc.model_dump(mode="json", by_alias=True, exclude_unset=True))

    @classmethod
    def from_json(cls, value: str | None) -> "WorkflowSnapshot":
        if value is None:
            raise WorkflowSnapshotError("workflow snapshot is unavailable")
        payload = loads(value)
        if not isinstance(payload, dict):
            raise WorkflowSnapshotError("workflow snapshot must be a JSON object")
        # Validate at the durable boundary so corrupted rows never become a graph.
        try:
            doc = WorkflowDoc.model_validate(payload)
        except Exception as exc:
            raise WorkflowSnapshotError("workflow snapshot failed schema validation") from exc
        return cls.from_doc(doc)

    def to_json(self) -> str:
        value = dumps(self.payload)
        assert value is not None
        return value

    def to_doc(self) -> WorkflowDoc:
        return WorkflowDoc.model_validate(self.payload)

    def compile(self):
        from ravana.compiler.graph import compile_workflow

        return compile_workflow(self.to_doc())

    def equivalent_to(self, doc: WorkflowDoc) -> bool:
        return self == self.from_doc(doc)
