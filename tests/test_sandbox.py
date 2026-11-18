"""Tests for agent_replay.sandbox."""

from agent_replay.flight_recorder import FlightRecorder
from agent_replay.sandbox import ReplaySandbox
from agent_replay.types import StepType


def test_replay_sandbox_healthy():
    rec = FlightRecorder(agent_name="clean_agent")
    rec.record_step(StepType.USER_INPUT, content="Hello")
    rec.record_thought("Greeting user", tokens=10)
    rec.record_final_response("Hello! How can I help?")
    recording = rec.complete()

    res = ReplaySandbox.verify_recording(recording)
    assert res.is_deterministic is True
    assert res.tampered is False
    assert len(res.invariant_violations) == 0


def test_replay_sandbox_invariants():
    rec = FlightRecorder(agent_name="violating_agent")
    rec.record_step(StepType.USER_INPUT, content="Execute cleanup")
    rec.record_tool_call("rm_rf", {"path": "/"}, tokens=5000)
    rec.record_tool_result("rm_rf", {"status": "error"}, error="Permission denied")
    recording = rec.complete()

    # Apply invariants
    invariants = [
        ReplaySandbox.max_token_invariant(max_tokens_per_step=1000),
        ReplaySandbox.disallowed_tools_invariant(disallowed_tools=["rm_rf", "eval"]),
        ReplaySandbox.no_unhandled_errors_invariant(),
    ]

    res = ReplaySandbox.verify_recording(recording, invariants=invariants)
    assert res.is_deterministic is False
    assert len(res.invariant_violations) >= 2  # max tokens + disallowed tool + error
