"""Tests for agent_replay.visualizer."""

import tempfile
from pathlib import Path
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import StepType
from agent_replay.visualizer import HTMLVisualizer


def test_html_visualizer_render():
    rec = FlightRecorder(agent_name="vis_agent", framework="langgraph")
    rec.record_step(StepType.USER_INPUT, content="Build a landing page", role="user")
    rec.record_thought("Designing component layout")
    rec.record_tool_call("create_component", {"name": "HeroSection"})
    rec.record_tool_result("create_component", {"status": "created", "lines": 42})
    rec.record_final_response("Landing page ready.")
    recording = rec.complete()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "dossier.html"
        content = HTMLVisualizer.render(recording, output_path=out_html)

        assert out_html.exists()
        assert "Flight Recording Dossier" in content
        assert "vis_agent" in content
        assert "HeroSection" in content
        assert recording.merkle_root in content
