"""Tests for agent_replay.debugger."""

import pytest
from agent_replay.debugger import TimeTravelDebugger
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import StepType


@pytest.fixture
def sample_recording():
    rec = FlightRecorder(agent_name="debugger_agent")
    rec.record_step(StepType.USER_INPUT, content="Execute transaction", role="user")
    rec.record_thought("Checking balance", tokens=20)
    rec.record_tool_call("get_balance", {"account": "0x123"})
    rec.record_tool_result("get_balance", {"balance_sol": 15.5})
    rec.record_state_mutation({"balance": 15.5, "verified": True})
    rec.record_tool_call("transfer", {"to": "0x456", "amount": 2.0})
    rec.record_tool_result("transfer", {"tx_hash": "5abc..."})
    rec.record_state_mutation({"balance": 13.5, "tx_count": 1})
    rec.record_final_response("Transferred 2.0 SOL")
    return rec.complete()


def test_time_travel_stepping(sample_recording):
    dbg = TimeTravelDebugger(sample_recording)
    assert dbg.cursor == -1
    assert dbg.current_step is None

    # Step forward
    step0 = dbg.step_forward()
    assert step0 is not None
    assert step0.step_index == 0
    assert dbg.cursor == 0

    step1 = dbg.step_forward()
    assert step1.step_index == 1
    assert dbg.cursor == 1

    # Step backward
    back0 = dbg.step_backward()
    assert back0.step_index == 0
    assert dbg.cursor == 0

    back_none = dbg.step_backward()
    assert back_none is None
    assert dbg.cursor == -1

    # Goto
    step5 = dbg.goto(5)
    assert step5.tool_name == "transfer"
    assert dbg.cursor == 5

    # Jump to end
    end_step = dbg.jump_to_end()
    assert end_step.step_type == StepType.FINAL_RESPONSE
    assert dbg.cursor == len(sample_recording.steps) - 1


def test_time_travel_state_reconstruction(sample_recording):
    dbg = TimeTravelDebugger(sample_recording)
    
    # State before any mutation
    dbg.goto(2)
    state2 = dbg.get_state_at_cursor()
    assert state2 == {}

    # State after first mutation (step 4)
    dbg.goto(4)
    state4 = dbg.get_state_at_cursor()
    assert state4.get("balance") == 15.5
    assert state4.get("verified") is True
    assert "tx_count" not in state4

    # State after second mutation (step 7)
    dbg.goto(7)
    state7 = dbg.get_state_at_cursor()
    assert state7.get("balance") == 13.5
    assert state7.get("tx_count") == 1


def test_time_travel_breakpoints(sample_recording):
    dbg = TimeTravelDebugger(sample_recording)
    bp = dbg.add_breakpoint(tool_name="transfer")
    assert bp.hit_count == 0

    # First hit on step 5 (tool_call transfer)
    hit_step = dbg.continue_execution()
    assert hit_step is not None
    assert hit_step.tool_name == "transfer"
    assert hit_step.step_type == StepType.TOOL_CALL
    assert hit_step.step_index == 5
    assert bp.hit_count == 1

    # Second hit on step 6 (tool_result transfer)
    hit_step2 = dbg.continue_execution()
    assert hit_step2 is not None
    assert hit_step2.tool_name == "transfer"
    assert hit_step2.step_type == StepType.TOOL_RESULT
    assert hit_step2.step_index == 6
    assert bp.hit_count == 2

    # Continue to end
    end_step = dbg.continue_execution()
    assert end_step.step_type == StepType.FINAL_RESPONSE


def test_time_travel_reverse_continue(sample_recording):
    dbg = TimeTravelDebugger(sample_recording)
    dbg.jump_to_end()
    bp = dbg.add_breakpoint(tool_name="get_balance")

    hit_step = dbg.reverse_continue()
    assert hit_step is not None
    assert hit_step.tool_name == "get_balance"
    assert bp.hit_count == 1
