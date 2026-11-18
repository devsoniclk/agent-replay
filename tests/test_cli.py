import json
from pathlib import Path

import pytest
from agent_replay import FlightRecorder
from agent_replay.cli import main
from agent_replay.types import ExecutionStatus


@pytest.fixture
def sample_flight_file(tmp_path: Path) -> str:
    rec = FlightRecorder(agent_name="cli-test-agent", framework="langchain")
    rec.record_thought("Analyzing requirements", latency_ms=12.0)
    rec.record_tool_call(
        tool_name="file_search",
        tool_args={"pattern": "*.py"},
        latency_ms=25.0,
    )
    rec.record_tool_result(
        tool_name="file_search",
        tool_result=["main.py", "test.py"],
        latency_ms=10.0,
    )
    rec.record_state_mutation({"active_file": "main.py"})
    rec.record_final_response(
        response="Operation completed successfully.",
        prompt_tokens=100,
        completion_tokens=50,
        status=ExecutionStatus.SUCCESS,
    )
    flight_path = tmp_path / "test_session.flight"
    rec.save(str(flight_path))
    return str(flight_path)


def test_cli_doctor(capsys):
    ret = main(["doctor"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Agent Replay System Doctor" in captured.out
    assert "Status:       All subsystems operational [PASS]" in captured.out


def test_cli_inspect_text(sample_flight_file, capsys):
    ret = main(["inspect", sample_flight_file])
    assert ret == 0
    captured = capsys.readouterr()
    assert "cli-test-agent" in captured.out
    assert "file_search" in captured.out


def test_cli_inspect_json(sample_flight_file, capsys):
    ret = main(["inspect", sample_flight_file, "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["agent_name"] == "cli-test-agent"
    assert data["total_steps"] == 5
    assert data["tokens"]["total"] == 150


def test_cli_travel(sample_flight_file, capsys):
    ret = main(["travel", sample_flight_file, "--step", "1"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Step [1/5]" in captured.out
    assert "file_search" in captured.out


def test_cli_verify(sample_flight_file, capsys):
    ret = main(["verify", sample_flight_file])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Deterministic:  YES [PASS]" in captured.out


def test_cli_diff(sample_flight_file, tmp_path, capsys):
    rec2 = FlightRecorder(agent_name="cli-test-agent-diff")
    rec2.record_thought("Divergent thought")
    rec2.record_final_response("Diff response")
    compare_path = tmp_path / "compare.flight"
    rec2.save(str(compare_path))

    ret = main(["diff", sample_flight_file, str(compare_path)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Trajectory Diff Report" in captured.out


def test_cli_report(sample_flight_file, tmp_path):
    out_html = tmp_path / "dossier.html"
    ret = main(["report", sample_flight_file, "--out", str(out_html)])
    assert ret == 0
    assert out_html.exists()
    assert "Agent Replay" in out_html.read_text()
