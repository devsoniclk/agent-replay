"""Tests for agent_replay.counterfactual."""

import pytest
from agent_replay.counterfactual import CounterfactualBranchingEngine, Intervention
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import StepType


@pytest.fixture
def base_mission():
    rec = FlightRecorder(agent_name="trader")
    rec.record_step(StepType.USER_INPUT, content="Check oracle price", role="user")
    rec.record_thought("Fetching Pyth oracle price")
    rec.record_tool_call("pyth_price", {"feed": "SOL/USD"})
    rec.record_tool_result("pyth_price", {"price": 180.5, "conf": 0.05})
    rec.record_thought("Price is favorable, placing long order")
    rec.record_tool_call("place_order", {"side": "buy", "size": 10})
    rec.record_tool_result("place_order", {"status": "filled", "order_id": "ord_101"})
    rec.record_final_response("Long order filled at 180.5")
    return rec.complete()


def test_counterfactual_branch_fork(base_mission):
    # Fork at step 2 (after tool call, before favorable result)
    # Intervene: what if price crashed to 120.0?
    intervention = Intervention(
        description="Oracle price flash drop to 120.0",
        override_tool_result={"price": 120.0, "conf": 0.05},
    )

    forked_recorder = CounterfactualBranchingEngine.fork_branch(
        base_recording=base_mission,
        at_step=2,
        branch_name="flash_crash_branch",
        intervention=intervention,
    )

    # In the forked world, agent sees low price and decides NOT to buy
    forked_recorder.record_thought("Price dropped sharply, aborting trade!")
    forked_recorder.record_final_response("Trade aborted due to adverse price.")
    forked_mission = forked_recorder.complete()

    assert forked_mission.branch_name == "flash_crash_branch"
    assert forked_mission.parent_recording_id == base_mission.recording_id
    assert forked_mission.fork_step_index == 2

    # Verify cryptographic integrity of forked branch
    is_valid, bad_idx, _ = forked_mission.verify_integrity()
    assert is_valid is True

    # Compare trajectories
    report = CounterfactualBranchingEngine.compare_trajectories(base_mission, forked_mission)
    assert report.divergence_step_index == 3
    assert "Diverged at step 3" in report.verdict
    assert report.tool_deltas["place_order"]["fork"] == 0
    assert report.tool_deltas["place_order"]["base"] == 2
