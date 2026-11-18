"""Tests for agent_replay.diff."""

from agent_replay.diff import TrajectoryDifferencer
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import StepType


def test_trajectory_diff_identical():
    rec_a = FlightRecorder(agent_name="agent")
    rec_a.record_thought("Thought 1")
    rec_a.record_final_response("Done")
    a = rec_a.complete()

    rec_b = FlightRecorder(agent_name="agent")
    rec_b.record_thought("Thought 1")
    rec_b.record_final_response("Done")
    b = rec_b.complete()

    diff_res = TrajectoryDifferencer.diff(a, b)
    assert diff_res.differences_count == 0
    assert len(diff_res.aligned_steps) == 2
    assert all(s.diff_type == "MATCH" for s in diff_res.aligned_steps)


def test_trajectory_diff_modified_and_added():
    rec_a = FlightRecorder(agent_name="agent")
    rec_a.record_thought("Initial plan")
    rec_a.record_tool_call("web_search", {"q": "cats"})
    a = rec_a.complete()

    rec_b = FlightRecorder(agent_name="agent")
    rec_b.record_thought("Initial plan")
    rec_b.record_tool_call("web_search", {"q": "dogs"})  # args modified
    rec_b.record_final_response("Search finished")        # added step
    b = rec_b.complete()

    diff_res = TrajectoryDifferencer.diff(a, b)
    assert diff_res.differences_count == 2
    assert diff_res.aligned_steps[1].diff_type == "MODIFIED"
    assert diff_res.aligned_steps[2].diff_type == "ADDED"

    d = diff_res.to_dict()
    assert d["differences_count"] == 2
