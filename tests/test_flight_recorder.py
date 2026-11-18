"""Tests for agent_replay.flight_recorder."""

import tempfile
from pathlib import Path
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import ExecutionStatus, StepType


def test_flight_recorder_full_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = FlightRecorder(
            agent_name="researcher",
            framework="openbot",
            storage_dir=tmpdir,
        )

        s0 = recorder.record_step(StepType.USER_INPUT, content="Find AI papers", role="user")
        s1 = recorder.record_thought("Searching arXiv for recent papers", tokens=45, latency_ms=120.0)
        s2 = recorder.record_tool_call("arxiv_search", {"query": "agent time travel"}, tokens=15, latency_ms=10.0)
        s3 = recorder.record_tool_result("arxiv_search", {"results": ["Paper 1", "Paper 2"]}, latency_ms=450.0)
        s4 = recorder.record_state_mutation({"last_query": "agent time travel", "count": 2})
        s5 = recorder.record_final_response("Found 2 papers.", tokens_prompt=200, tokens_completion=50, latency_ms=80.0)

        recording = recorder.complete(ExecutionStatus.COMPLETED)

        assert len(recording.steps) == 6
        assert recording.status == ExecutionStatus.COMPLETED
        assert len(recording.merkle_root) == 64

        p_tok, c_tok, t_tok = recording.total_tokens()
        assert t_tok == 45 + 15 + 200 + 50
        assert recording.total_latency_ms() > 600.0

        freq = recording.tool_usage_frequency()
        assert freq.get("arxiv_search") == 2  # call + result

        # Verify hash integrity
        is_valid, bad_idx, _ = recording.verify_integrity()
        assert is_valid is True

        # Test Save & Load (.flight)
        saved_path = recorder.save()
        assert Path(saved_path).exists()

        loaded_rec = FlightRecorder.load(saved_path)
        assert loaded_rec.recording_id == recording.recording_id
        assert len(loaded_rec.steps) == 6
        assert loaded_rec.merkle_root == recording.merkle_root


def test_flight_recorder_context_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        with FlightRecorder(agent_name="context_agent", storage_dir=tmpdir) as rec:
            rec.record_thought("Step 1")
            rec.record_final_response("Done")

        assert rec.recording.status == ExecutionStatus.COMPLETED
        assert len(rec.recording.steps) == 2


def test_flight_recorder_exception_handling():
    try:
        with FlightRecorder(agent_name="failing_agent") as rec:
            rec.record_thought("Attempting risky operation")
            raise ValueError("Database connection timeout")
    except ValueError:
        pass

    assert rec.recording.status == ExecutionStatus.FAILED
    assert rec.recording.steps[-1].step_type == StepType.ERROR
    assert "Database connection timeout" in rec.recording.steps[-1].content
