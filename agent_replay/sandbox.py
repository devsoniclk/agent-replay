"""
agent_replay.sandbox

Deterministic replay sandbox for validating agent trajectories under controlled
or mocked conditions, asserting safety invariants, and measuring state drift.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agent_replay.types import FlightRecording, FlightStep, ReplayVerificationResult, StepType


class ReplaySandbox:
    """Deterministic execution sandbox verifying historical agent trajectories."""

    @classmethod
    def verify_recording(
        cls,
        recording: FlightRecording,
        invariants: Optional[List[Callable[[FlightStep, Dict[str, Any]], Optional[str]]]] = None,
    ) -> ReplayVerificationResult:
        """Verify cryptographic integrity and invariant compliance of a flight recording."""
        is_valid_chain, bad_step, chain_msg = recording.verify_integrity()
        tampered = not is_valid_chain

        violations: List[Dict[str, Any]] = []
        divergent_steps: List[int] = []

        if bad_step is not None:
            divergent_steps.append(bad_step)

        # Roll state and check invariants
        current_state: Dict[str, Any] = {}
        for step in recording.steps:
            if step.memory_snapshot:
                current_state.update(step.memory_snapshot)
            elif step.state_delta:
                current_state.update(step.state_delta)

            if invariants:
                for inv in invariants:
                    try:
                        err = inv(step, current_state)
                        if err:
                            violations.append({
                                "step_index": step.step_index,
                                "tool_name": step.tool_name,
                                "violation": err,
                            })
                    except Exception as e:
                        violations.append({
                            "step_index": step.step_index,
                            "violation": f"Invariant check exception: {str(e)}",
                        })

        is_deterministic = not tampered and len(violations) == 0
        details = (
            f"Verified {len(recording.steps)} steps. "
            f"Integrity: {'OK' if not tampered else 'FAILED (' + chain_msg + ')'}. "
            f"Invariant violations: {len(violations)}."
        )

        return ReplayVerificationResult(
            recording_id=recording.recording_id,
            is_deterministic=is_deterministic,
            tampered=tampered,
            matched_steps=len(recording.steps) - len(divergent_steps),
            total_steps=len(recording.steps),
            divergent_step_indices=divergent_steps,
            invariant_violations=violations,
            details=details,
        )

    # Standard Invariant Builders
    @staticmethod
    def max_token_invariant(max_tokens_per_step: int) -> Callable[[FlightStep, Dict[str, Any]], Optional[str]]:
        def check(step: FlightStep, state: Dict[str, Any]) -> Optional[str]:
            tokens = step.tokens_prompt + step.tokens_completion
            if tokens > max_tokens_per_step:
                return f"Step exceeded max token limit ({tokens} > {max_tokens_per_step})"
            return None
        return check

    @staticmethod
    def disallowed_tools_invariant(disallowed_tools: List[str]) -> Callable[[FlightStep, Dict[str, Any]], Optional[str]]:
        def check(step: FlightStep, state: Dict[str, Any]) -> Optional[str]:
            if step.tool_name and step.tool_name in disallowed_tools:
                return f"Invocation of disallowed tool '{step.tool_name}' detected"
            return None
        return check

    @staticmethod
    def no_unhandled_errors_invariant() -> Callable[[FlightStep, Dict[str, Any]], Optional[str]]:
        def check(step: FlightStep, state: Dict[str, Any]) -> Optional[str]:
            if step.step_type == StepType.ERROR:
                return f"Unhandled error recorded: {step.content}"
            if step.tool_error and not step.tool_result:
                return f"Tool error recorded on '{step.tool_name}': {step.tool_error}"
            return None
        return check
