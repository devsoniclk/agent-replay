"""Tests for agent_replay.types."""

import pytest
from agent_replay.types import (
    Breakpoint,
    ExecutionStatus,
    FlightRecording,
    FlightStep,
    StepType,
)


def test_step_type_and_execution_status_enums():
    assert StepType.THOUGHT == "thought"
    assert StepType.TOOL_CALL == "tool_call"
    assert StepType.TOOL_RESULT == "tool_result"
    assert ExecutionStatus.RECORDING == "recording"
    assert ExecutionStatus.COMPLETED == "completed"


def test_flight_step_digest_and_seal():
    step = FlightStep(
        step_index=0,
        step_type=StepType.USER_INPUT,
        agent_id="test_agent",
        role="user",
        content="Analyze the quarterly report",
    )
    assert step.digest == ""
    step.seal()
    assert len(step.digest) == 64
    original_digest = step.digest

    # Serialization roundtrip
    d = step.to_dict()
    reconstructed = FlightStep.from_dict(d)
    assert reconstructed.step_index == 0
    assert reconstructed.step_type == StepType.USER_INPUT
    assert reconstructed.digest == original_digest


def test_flight_recording_integrity_and_merkle_root():
    rec = FlightRecording(
        recording_id="rec_001",
        session_id="sess_001",
        agent_name="agent_alpha",
    )
    s0 = FlightStep(
        step_index=0,
        step_type=StepType.USER_INPUT,
        agent_id="agent_alpha",
        role="user",
        content="Hello",
    ).seal()

    s1 = FlightStep(
        step_index=1,
        step_type=StepType.THOUGHT,
        agent_id="agent_alpha",
        role="assistant",
        content="Thinking...",
        parent_digest=s0.digest,
    ).seal()

    rec.steps = [s0, s1]
    rec.compute_merkle_root()
    assert len(rec.merkle_root) == 64

    is_valid, bad_idx, msg = rec.verify_integrity()
    assert is_valid is True
    assert bad_idx is None

    # Test broken chain
    s1.parent_digest = "tampered_parent"
    is_valid, bad_idx, msg = rec.verify_integrity()
    assert is_valid is False
    assert bad_idx == 1
    assert "Hash chain broken" in msg


def test_flight_recording_tampered_step_digest():
    rec = FlightRecording(recording_id="rec_002", session_id="s2", agent_name="agent")
    s0 = FlightStep(step_index=0, step_type=StepType.USER_INPUT, agent_id="a", role="u", content="query").seal()
    rec.steps = [s0]
    rec.compute_merkle_root()

    # Tamper with content without updating digest
    s0.content = "hacked query"
    is_valid, bad_idx, msg = rec.verify_integrity()
    assert is_valid is False
    assert bad_idx == 0
    assert "tampered" in msg


def test_breakpoint_matching():
    step = FlightStep(
        step_index=4,
        step_type=StepType.TOOL_CALL,
        agent_id="agent",
        role="assistant",
        tool_name="bash_exec",
        tool_args={"cmd": "rm -rf /tmp"},
    )
    state = {"workspace": "/tmp", "safe_mode": False}

    bp_index = Breakpoint("bp1", step_index=4)
    assert bp_index.matches(step, state) is True

    bp_tool = Breakpoint("bp2", tool_name="bash_exec")
    assert bp_tool.matches(step, state) is True

    bp_tool_miss = Breakpoint("bp3", tool_name="read_file")
    assert bp_tool_miss.matches(step, state) is False

    bp_cond = Breakpoint("bp4", condition_expr="tool == 'bash_exec' and not state.get('safe_mode')")
    assert bp_cond.matches(step, state) is True

    bp_disabled = Breakpoint("bp5", step_index=4, enabled=False)
    assert bp_disabled.matches(step, state) is False
