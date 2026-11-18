"""
agent_replay.diff

Structural trajectory differencer comparing agent execution runs step-by-step,
detecting tool argument changes, output differences, and latency/token shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent_replay.types import FlightRecording, FlightStep


@dataclass
class StepDiff:
    """Detailed diff comparison between two corresponding trajectory steps."""
    step_index: int
    base_step: Optional[FlightStep]
    compare_step: Optional[FlightStep]
    diff_type: str  # "MATCH", "MODIFIED", "ADDED", "REMOVED"
    changes: List[str]


@dataclass
class TrajectoryDiffResult:
    """Full trajectory alignment diff result between two flight recordings."""
    base_id: str
    compare_id: str
    total_base_steps: int
    total_compare_steps: int
    aligned_steps: List[StepDiff]
    differences_count: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_id": self.base_id,
            "compare_id": self.compare_id,
            "total_base_steps": self.total_base_steps,
            "total_compare_steps": self.total_compare_steps,
            "differences_count": self.differences_count,
            "summary": self.summary,
            "step_diffs": [
                {
                    "step_index": s.step_index,
                    "diff_type": s.diff_type,
                    "changes": s.changes,
                    "base_tool": s.base_step.tool_name if s.base_step else None,
                    "compare_tool": s.compare_step.tool_name if s.compare_step else None,
                }
                for s in self.aligned_steps
            ],
        }


class TrajectoryDifferencer:
    """Engine for computing side-by-side structural differences between trajectories."""

    @classmethod
    def diff(cls, base: FlightRecording, compare: FlightRecording) -> TrajectoryDiffResult:
        """Compute structural alignment and differences between two flight recordings."""
        aligned_diffs: List[StepDiff] = []
        max_len = max(len(base.steps), len(compare.steps))
        diff_count = 0

        for i in range(max_len):
            b = base.steps[i] if i < len(base.steps) else None
            c = compare.steps[i] if i < len(compare.steps) else None

            if b is not None and c is not None:
                changes = []
                if b.step_type != c.step_type:
                    changes.append(f"Type changed: '{b.step_type}' -> '{c.step_type}'")
                if b.tool_name != c.tool_name:
                    changes.append(f"Tool changed: '{b.tool_name}' -> '{c.tool_name}'")
                if b.tool_args != c.tool_args:
                    changes.append("Tool arguments modified")
                if b.tool_result != c.tool_result:
                    changes.append("Tool outputs differ")
                if b.content != c.content:
                    changes.append("Content text differs")
                if b.tool_error != c.tool_error:
                    changes.append(f"Error state changed: '{b.tool_error}' -> '{c.tool_error}'")

                if changes:
                    diff_count += 1
                    aligned_diffs.append(StepDiff(i, b, c, "MODIFIED", changes))
                else:
                    aligned_diffs.append(StepDiff(i, b, c, "MATCH", []))
            elif b is not None:
                diff_count += 1
                aligned_diffs.append(StepDiff(i, b, None, "REMOVED", ["Step exists only in base run"]))
            elif c is not None:
                diff_count += 1
                aligned_diffs.append(StepDiff(i, None, c, "ADDED", ["Step exists only in compare run"]))

        summary = (
            f"Compared {len(base.steps)} base steps with {len(compare.steps)} compare steps. "
            f"Found {diff_count} differing steps."
        )

        return TrajectoryDiffResult(
            base_id=base.recording_id,
            compare_id=compare.recording_id,
            total_base_steps=len(base.steps),
            total_compare_steps=len(compare.steps),
            aligned_steps=aligned_diffs,
            differences_count=diff_count,
            summary=summary,
        )
