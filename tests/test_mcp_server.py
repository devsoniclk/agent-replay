"""Tests for agent_replay.mcp_server."""

import tempfile
from pathlib import Path
from agent_replay.mcp_server import AgentReplayMCPServer


def test_mcp_server_full_dispatch_flow():
    server = AgentReplayMCPServer()

    # 1. Create recording
    r1 = server.create_recording(agent_name="mcp_bot", framework="crewai")
    assert r1["status"] == "success"
    rec_id = r1["recording_id"]

    # 2. Record steps
    s1 = server.record_step(rec_id, step_type="user_input", content="Fetch data")
    assert s1["status"] == "success"
    assert s1["step_index"] == 0

    s2 = server.record_step(
        rec_id,
        step_type="tool_call",
        tool_name="fetch_url",
        tool_args={"url": "https://api.github.com"},
    )
    assert s2["status"] == "success"

    # 3. Export flight
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = str(Path(tmpdir) / "test.flight")
        exp = server.export_flight(rec_id, out_file)
        assert exp["status"] == "success"
        assert Path(out_file).exists()

        # 4. Load flight in debugger
        loaded = server.load_flight(out_file)
        assert loaded["status"] == "success"
        assert loaded["total_steps"] == 2

        # 5. Time travel
        tt_forward = server.time_travel(rec_id, action="forward")
        assert tt_forward["status"] == "success"
        assert tt_forward["cursor"] == 0

        tt_end = server.time_travel(rec_id, action="end")
        assert tt_end["status"] == "success"
        assert tt_end["cursor"] == 1

        # 6. Fork counterfactual
        fork_res = server.fork_counterfactual(
            base_recording_id=rec_id,
            at_step=0,
            branch_name="alt_branch",
            override_tool_result={"mocked": True},
        )
        assert fork_res["status"] == "success"

        # 7. Verify determinism
        ver = server.verify_determinism(rec_id)
        assert ver["status"] == "success"
        assert ver["verification"]["is_deterministic"] is True

        # 8. Render HTML
        html_out = str(Path(tmpdir) / "dossier.html")
        rend = server.render_html(rec_id, html_out)
        assert rend["status"] == "success"
        assert Path(html_out).exists()
