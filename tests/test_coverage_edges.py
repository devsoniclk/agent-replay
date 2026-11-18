"""Edge cases and error boundary coverage tests for agent-replay."""

import io
import json
import tempfile
from pathlib import Path
import pytest
from agent_replay.cli import main
from agent_replay.debugger import TimeTravelDebugger
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.mcp_server import AgentReplayMCPServer
from agent_replay.sandbox import ReplaySandbox
from agent_replay.types import FlightRecording, FlightStep, StepType


def test_flight_recorder_file_not_found():
    with pytest.raises(FileNotFoundError):
        FlightRecorder.load("/nonexistent/path/to/flight.file")


def test_debugger_index_out_of_bounds():
    rec = FlightRecording("rec_bounds", "s", "a")
    s0 = FlightStep(0, StepType.THOUGHT, "a", "role", content="thought").seal()
    rec.steps = [s0]
    dbg = TimeTravelDebugger(rec)

    with pytest.raises(IndexError):
        dbg.goto(10)
    with pytest.raises(IndexError):
        dbg.goto(-2)


def test_debugger_clear_and_remove_breakpoint():
    rec = FlightRecording("rec_bp", "s", "a")
    dbg = TimeTravelDebugger(rec)
    bp = dbg.add_breakpoint(step_index=0)
    assert bp.breakpoint_id in dbg.breakpoints

    assert dbg.remove_breakpoint(bp.breakpoint_id) is True
    assert dbg.remove_breakpoint("nonexistent") is False

    dbg.add_breakpoint(tool_name="tool_a")
    dbg.add_breakpoint(tool_name="tool_b")
    assert len(dbg.breakpoints) == 2
    dbg.clear_breakpoints()
    assert len(dbg.breakpoints) == 0


def test_debugger_inspect_frame_empty_recording():
    rec = FlightRecording("rec_empty", "s", "a")
    dbg = TimeTravelDebugger(rec)
    frame = dbg.inspect_frame()
    assert frame["cursor"] == -1
    assert frame["total_steps"] == 0
    assert frame["step"] is None
    assert frame["state"] == {}
    assert dbg.get_tool_history() == []


def test_mcp_server_error_paths():
    server = AgentReplayMCPServer()

    # Step on missing recording
    err1 = server.record_step("invalid_id", step_type="thought")
    assert err1["status"] == "error"

    # Export on missing recording
    err2 = server.export_flight("invalid_id", "/tmp/out.flight")
    assert err2["status"] == "error"

    # Load bad file
    err3 = server.load_flight("/tmp/totally_missing.flight")
    assert err3["status"] == "error"

    # Time travel on missing debugger
    err4 = server.time_travel("invalid_id", action="forward")
    assert err4["status"] == "error"

    # Time travel with invalid action
    # First create valid recording
    r = server.create_recording("bot")
    rid = r["recording_id"]
    server.active_debuggers[rid] = TimeTravelDebugger(server.active_recorders[rid].recording)
    err5 = server.time_travel(rid, action="teleport")
    assert err5["status"] == "error"

    # Goto missing step_index
    err6 = server.time_travel(rid, action="goto")
    assert err6["status"] == "error"

    # Unknown tool dispatch
    err7 = server._dispatch_tool("unknown_tool", {})
    assert err7["status"] == "error"


def test_cli_no_command(capsys):
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "agent-replay" in captured.out
