"""Tests for agent_replay.adapters."""

import pytest
from agent_replay.adapters import record_agent_step, record_tool
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import StepType


def test_record_agent_step_decorator():
    recorder = FlightRecorder(agent_name="decorated_agent")

    @record_agent_step(recorder, step_type=StepType.THOUGHT)
    def compute_analysis(data):
        return len(data) * 2

    res = compute_analysis([1, 2, 3])
    assert res == 6
    assert len(recorder.recording.steps) == 1
    assert recorder.recording.steps[0].step_type == StepType.THOUGHT
    assert "compute_analysis" in recorder.recording.steps[0].content


def test_record_tool_decorator():
    recorder = FlightRecorder(agent_name="tool_agent")

    @record_tool(recorder, tool_name="calculator_add")
    def add(a: int, b: int) -> int:
        return a + b

    val = add(10, 25)
    assert val == 35
    assert len(recorder.recording.steps) == 2  # call + result
    call_step = recorder.recording.steps[0]
    result_step = recorder.recording.steps[1]

    assert call_step.step_type == StepType.TOOL_CALL
    assert call_step.tool_name == "calculator_add"
    assert result_step.step_type == StepType.TOOL_RESULT
    assert result_step.tool_result == 35


def test_record_tool_decorator_exception():
    recorder = FlightRecorder(agent_name="tool_agent")

    @record_tool(recorder, tool_name="failing_tool")
    def fail_op():
        raise RuntimeError("Network failure")

    with pytest.raises(RuntimeError):
        fail_op()

    assert len(recorder.recording.steps) == 2
    assert recorder.recording.steps[1].tool_error == "Network failure"
