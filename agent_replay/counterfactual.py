"""
agent_replay.counterfactual

Counterfactual branching engine for forking agent trajectories at any historic step,
injecting alternative observations, tool results or prompts, and analyzing divergence.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import (
    DivergenceReport,
    ExecutionStatus,
    FlightRecording,
    FlightStep,
    StepType,
)


@dataclass
class Intervention:
    """Deliberate counterfactual intervention applied to an execution branch."""
    intervention_id: str = field(default_factory=lambda: f"intv_{uuid.uuid4().hex[:6]}")
    description: str = ""
    override_tool_result: Optional[Any] = None
    override_tool_error: Optional[str] = None
    override_observation: Optional[str] = None
    override_thought: Optional[str] = None
    override_state_delta: Optional[Dict[str, Any]] = None
    injected_user_prompt: Optional[str] = None


class CounterfactualBranchingEngine:
    """Engine for creating and analyzing counterfactual execution branches."""

    @classmethod
    def fork_branch(
        cls,
        base_recording: FlightRecording,
        at_step: int,
        branch_name: str,
        intervention: Optional[Intervention] = None,
        description: str = "",
    ) -> FlightRecorder:
        """Fork an existing flight recording at step `at_step` and apply an intervention."""
        if at_step < 0 or at_step >= len(base_recording.steps):
            raise IndexError(
                f"Fork step {at_step} is invalid for recording with {len(base_recording.steps)} steps"
            )

        new_recording_id = f"fork_{branch_name}_{uuid.uuid4().hex[:6]}"
        recorder = FlightRecorder(
            agent_name=base_recording.agent_name,
            session_id=base_recording.session_id,
            framework=base_recording.framework,
            metadata={
                **base_recording.metadata,
                "fork_origin_id": base_recording.recording_id,
                "fork_step_index": at_step,
                "branch_name": branch_name,
                "fork_description": description,
            },
        )
        recorder.recording.recording_id = new_recording_id
        recorder.recording.parent_recording_id = base_recording.recording_id
        recorder.recording.fork_step_index = at_step
        recorder.recording.branch_name = branch_name

        # Copy steps up to and including at_step
        recorder.recording.steps = []
        last_digest = ""
        for i in range(at_step + 1):
            src_step = base_recording.steps[i]
            copied_step = FlightStep(
                step_index=src_step.step_index,
                step_type=src_step.step_type,
                agent_id=src_step.agent_id,
                role=src_step.role,
                content=src_step.content,
                tool_name=src_step.tool_name,
                tool_args=src_step.tool_args,
                tool_result=src_step.tool_result,
                tool_error=src_step.tool_error,
                state_delta=src_step.state_delta,
                memory_snapshot=src_step.memory_snapshot,
                tokens_prompt=src_step.tokens_prompt,
                tokens_completion=src_step.tokens_completion,
                latency_ms=src_step.latency_ms,
                timestamp_ns=src_step.timestamp_ns,
                environment_hash=src_step.environment_hash,
                parent_digest=last_digest,
            ).seal()
            last_digest = copied_step.digest
            recorder.recording.steps.append(copied_step)

        recorder._current_step_index = at_step + 1
        recorder._last_digest = last_digest

        # Apply counterfactual intervention if provided
        if intervention:
            if intervention.override_tool_result is not None or intervention.override_tool_error is not None:
                recorder.record_step(
                    step_type=StepType.TOOL_RESULT,
                    tool_name="intervention_override",
                    tool_result=intervention.override_tool_result,
                    tool_error=intervention.override_tool_error,
                    content=f"Counterfactual intervention: {intervention.description or 'Overridden tool output'}",
                    state_delta=intervention.override_state_delta,
                )
            elif intervention.override_observation is not None:
                recorder.record_observation(
                    observation=intervention.override_observation,
                    source="counterfactual_intervention",
                )
            elif intervention.injected_user_prompt is not None:
                recorder.record_step(
                    step_type=StepType.USER_INPUT,
                    role="user",
                    content=intervention.injected_user_prompt,
                )
            elif intervention.override_thought is not None:
                recorder.record_thought(thought=intervention.override_thought)

        return recorder

    @classmethod
    def compare_trajectories(
        cls,
        base: FlightRecording,
        fork: FlightRecording,
    ) -> DivergenceReport:
        """Compare two trajectories and compute divergence statistics."""
        min_steps = min(len(base.steps), len(fork.steps))
        divergence_idx: Optional[int] = None

        for i in range(min_steps):
            b_step = base.steps[i]
            f_step = fork.steps[i]
            if (
                b_step.step_type != f_step.step_type
                or b_step.tool_name != f_step.tool_name
                or b_step.content != f_step.content
                or b_step.tool_args != f_step.tool_args
                or b_step.tool_result != f_step.tool_result
            ):
                divergence_idx = i
                break

        if divergence_idx is None and len(base.steps) != len(fork.steps):
            divergence_idx = min_steps

        # Compute tool similarity
        base_tools = [s.tool_name for s in base.steps if s.tool_name]
        fork_tools = [s.tool_name for s in fork.steps if s.tool_name]
        
        matches = sum(1 for bt, ft in zip(base_tools, fork_tools) if bt == ft)
        max_tools = max(len(base_tools), len(fork_tools), 1)
        tool_similarity = matches / max_tools

        # State keys difference
        base_state_keys = set()
        for s in base.steps:
            if s.state_delta:
                base_state_keys.update(s.state_delta.keys())
        fork_state_keys = set()
        for s in fork.steps:
            if s.state_delta:
                fork_state_keys.update(s.state_delta.keys())

        state_diff = sorted(list(base_state_keys.symmetric_difference(fork_state_keys)))

        # Token & latency deltas
        _, _, base_tokens = base.total_tokens()
        _, _, fork_tokens = fork.total_tokens()
        token_delta = fork_tokens - base_tokens
        latency_delta = fork.total_latency_ms() - base.total_latency_ms()

        # Tool deltas
        b_freq = base.tool_usage_frequency()
        f_freq = fork.tool_usage_frequency()
        all_tools = sorted(list(set(b_freq.keys()).union(f_freq.keys())))
        tool_deltas: Dict[str, Dict[str, int]] = {}
        for t in all_tools:
            b_count = b_freq.get(t, 0)
            f_count = f_freq.get(t, 0)
            tool_deltas[t] = {"base": b_count, "fork": f_count, "delta": f_count - b_count}

        verdict = (
            f"Diverged at step {divergence_idx}"
            if divergence_idx is not None
            else "Identical trajectories (0 divergence)"
        )

        return DivergenceReport(
            base_recording_id=base.recording_id,
            fork_recording_id=fork.recording_id,
            divergence_step_index=divergence_idx,
            total_steps_base=len(base.steps),
            total_steps_fork=len(fork.steps),
            tool_sequence_similarity=round(tool_similarity, 4),
            state_divergence_keys=state_diff,
            token_delta=token_delta,
            latency_delta_ms=round(latency_delta, 2),
            tool_deltas=tool_deltas,
            verdict=verdict,
        )
