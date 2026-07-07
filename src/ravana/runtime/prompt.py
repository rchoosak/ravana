"""Prompt Assembler (§1.1, §1.6). Builds the system prompt an agent turn runs
with: the agent's persona, its attached Skills (always-on concatenation —
§1.6's Phase-0b strategy, not progressive disclosure yet), and the current
Global Workspace state injected as context. Kept separate from the gateway
so the "what the model is told" logic is testable without any provider.
"""

from __future__ import annotations

import json
from typing import Any

from ravana.schema.models import AgentConfig, SkillConfig


def assemble_system_prompt(
    agent: AgentConfig,
    skills_by_id: dict[str, SkillConfig],
    shared_state: dict[str, Any],
) -> str:
    parts: list[str] = [agent.system_prompt.strip()]

    for skill_id in agent.skills:
        skill = skills_by_id.get(skill_id)
        if skill is None:
            continue
        parts.append(f"## Skill: {skill.description}\n{skill.instructions.strip()}")

    # The blackboard (§1.3) the agent reads from. JSON so a value that is
    # itself structured (qa_report, milestone_plan) round-trips faithfully.
    parts.append("## Current shared state\n" + json.dumps(shared_state, indent=2, sort_keys=True))

    return "\n\n".join(parts)
