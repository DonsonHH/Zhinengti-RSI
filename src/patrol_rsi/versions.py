"""Versioned, bounded outer-loop proposals. No autonomous RSI claim yet."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class AgentVersion:
    estimator_samples: int = 1
    estimator_refresh_interval: int = 1
    coordination_order: str = "q_priority"
    diagnostic_instruction: str = "Use development logs to identify a reusable patrolling failure."

    def __post_init__(self):
        for key, value, upper in [("estimator_samples", self.estimator_samples, 8),
                                  ("estimator_refresh_interval", self.estimator_refresh_interval, 8)]:
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{key} must be an integer in [1,{upper}]")
        if self.coordination_order not in ("q_priority", "cyclic"):
            raise ValueError("Unknown coordination order")
        if not isinstance(self.diagnostic_instruction, str) or not 1 <= len(self.diagnostic_instruction) <= 2000:
            raise ValueError("Invalid diagnostic instruction")

    @property
    def version_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def proposed(self, patch: dict) -> "AgentVersion":
        unknown = set(patch) - set(asdict(self))
        if unknown:
            raise ValueError(f"Protected or unknown proposal fields: {sorted(unknown)}")
        return AgentVersion(**(asdict(self) | patch))


def record_proposal(directory: Path, parent: AgentVersion, patch: dict,
                    development_evaluation: dict, reason: str, accepted: bool,
                    origin: str) -> Path:
    if development_evaluation.get("split") != "development":
        raise ValueError("Proposals may be selected from development evaluation only")
    if origin not in ("human_fixture", "language_model", "search_baseline"):
        raise ValueError("Proposal origin is required")
    candidate = parent.proposed(patch)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{parent.version_id}-{candidate.version_id}.json"
    record = {"parent": parent.version_id, "candidate": candidate.version_id,
              "config": asdict(candidate), "patch": patch, "evaluation": development_evaluation,
              "reason": reason, "accepted": bool(accepted), "origin": origin}
    # Repeated evaluation must not silently overwrite the earlier evidence.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    return path
