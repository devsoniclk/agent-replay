"""
agent_replay.mcp_server

FastMCP and JSON-RPC 2.0 Model Context Protocol server exposing deterministic
flight recording, time-travel stepping, counterfactual branching, and diffing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_replay.counterfactual import CounterfactualBranchingEngine, Intervention
from agent_replay.debugger import TimeTravelDebugger
from agent_replay.diff import TrajectoryDifferencer
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.sandbox import ReplaySandbox
from agent_replay.types import ExecutionStatus, FlightRecording, StepType
from agent_replay.visualizer import HTMLVisualizer


class AgentReplayMCPServer:
    """In-memory state and tool dispatcher for Agent Replay MCP server."""

    def __init__(self) -> None:
        self.active_recorders: Dict[str, FlightRecorder] = {}
        self.active_debuggers: Dict[str, TimeTravelDebugger] = {}

    def create_recording(
        self,
        agent_name: str = "agent",
        framework: str = "mcp",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rec = FlightRecorder(agent_name=agent_name, framework=framework, session_id=session_id)
        self.active_recorders[rec.recording_id] = rec
        return {
            "status": "success",
            "recording_id": rec.recording_id,
            "session_id": rec.session_id,
            "message": f"Flight recording initialized for {agent_name}",
        }

    def record_step(
        self,
        recording_id: str,
        step_type: str,
        content: str = "",
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Any] = None,
        tool_error: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:
        rec = self.active_recorders.get(recording_id)
        if not rec:
            return {"status": "error", "error": f"Recording ID '{recording_id}' not found"}

        step = rec.record_step(
            step_type=step_type,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_error=tool_error,
            latency_ms=latency_ms,
        )
        return {
            "status": "success",
            "step_index": step.step_index,
            "digest": step.digest,
            "total_steps": len(rec.recording.steps),
        }

    def export_flight(self, recording_id: str, output_path: str) -> Dict[str, Any]:
        rec = self.active_recorders.get(recording_id)
        if not rec:
            return {"status": "error", "error": f"Recording ID '{recording_id}' not found"}

        path = rec.save(output_path)
        return {
            "status": "success",
            "recording_id": recording_id,
            "saved_path": path,
            "merkle_root": rec.recording.merkle_root,
        }

    def load_flight(self, filepath: str) -> Dict[str, Any]:
        try:
            recording = FlightRecorder.load(filepath)
            debugger = TimeTravelDebugger(recording)
            self.active_debuggers[recording.recording_id] = debugger
            return {
                "status": "success",
                "recording_id": recording.recording_id,
                "agent_name": recording.agent_name,
                "total_steps": debugger.total_steps,
                "merkle_root": recording.merkle_root,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def time_travel(self, recording_id: str, action: str, step_index: Optional[int] = None) -> Dict[str, Any]:
        dbg = self.active_debuggers.get(recording_id)
        if not dbg:
            return {"status": "error", "error": f"Debugger session for '{recording_id}' not active. Call load_flight first."}

        if action == "forward":
            step = dbg.step_forward()
        elif action == "backward":
            step = dbg.step_backward()
        elif action == "goto":
            if step_index is None:
                return {"status": "error", "error": "step_index required for goto"}
            step = dbg.goto(step_index)
        elif action == "reset":
            dbg.reset()
            step = None
        elif action == "end":
            step = dbg.jump_to_end()
        else:
            return {"status": "error", "error": f"Unknown time-travel action '{action}'"}

        return {
            "status": "success",
            "cursor": dbg.cursor,
            "total_steps": dbg.total_steps,
            "step": step.to_dict() if step else None,
            "state": dbg.get_state_at_cursor(),
        }

    def fork_counterfactual(
        self,
        base_recording_id: str,
        at_step: int,
        branch_name: str,
        override_tool_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        dbg = self.active_debuggers.get(base_recording_id)
        base_rec = dbg.recording if dbg else (self.active_recorders[base_recording_id].recording if base_recording_id in self.active_recorders else None)
        if not base_rec:
            return {"status": "error", "error": f"Base recording '{base_recording_id}' not found."}

        intervention = Intervention(override_tool_result=override_tool_result) if override_tool_result is not None else None
        forked_recorder = CounterfactualBranchingEngine.fork_branch(
            base_recording=base_rec,
            at_step=at_step,
            branch_name=branch_name,
            intervention=intervention,
        )
        self.active_recorders[forked_recorder.recording_id] = forked_recorder
        return {
            "status": "success",
            "fork_recording_id": forked_recorder.recording_id,
            "branch_name": branch_name,
            "fork_step_index": at_step,
            "current_steps": len(forked_recorder.recording.steps),
        }

    def diff_trajectories(self, base_id: str, compare_id: str) -> Dict[str, Any]:
        rec_a = self._resolve_recording(base_id)
        rec_b = self._resolve_recording(compare_id)
        if not rec_a or not rec_b:
            return {"status": "error", "error": "One or both recording IDs could not be resolved."}

        diff_res = TrajectoryDifferencer.diff(rec_a, rec_b)
        return {"status": "success", "diff": diff_res.to_dict()}

    def verify_determinism(self, recording_id: str) -> Dict[str, Any]:
        rec = self._resolve_recording(recording_id)
        if not rec:
            return {"status": "error", "error": f"Recording '{recording_id}' not found."}

        result = ReplaySandbox.verify_recording(rec)
        return {"status": "success", "verification": result.to_dict()}

    def render_html(self, recording_id: str, output_path: str) -> Dict[str, Any]:
        rec = self._resolve_recording(recording_id)
        if not rec:
            return {"status": "error", "error": f"Recording '{recording_id}' not found."}

        HTMLVisualizer.render(rec, output_path)
        return {"status": "success", "output_path": output_path}

    def _resolve_recording(self, rec_id: str) -> Optional[FlightRecording]:
        if rec_id in self.active_debuggers:
            return self.active_debuggers[rec_id].recording
        if rec_id in self.active_recorders:
            return self.active_recorders[rec_id].recording
        return None

    def run_stdio_jsonrpc(self) -> None:
        """Run JSON-RPC 2.0 stdio server loop for direct MCP integration."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                req = json.loads(line.strip())
                req_id = req.get("id")
                method = req.get("method")
                params = req.get("params", {})

                if method == "tools/list":
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {"name": "agentreplay_create_recording", "description": "Create flight recorder"},
                                {"name": "agentreplay_record_step", "description": "Append step to recording"},
                                {"name": "agentreplay_export_flight", "description": "Save flight recording file"},
                                {"name": "agentreplay_load_flight", "description": "Load flight recording"},
                                {"name": "agentreplay_time_travel", "description": "Step forward/backward in time"},
                                {"name": "agentreplay_fork_counterfactual", "description": "Fork branch at historic step"},
                                {"name": "agentreplay_diff_trajectories", "description": "Compare two trajectories"},
                                {"name": "agentreplay_verify_determinism", "description": "Verify hash chain and sandbox replay"},
                                {"name": "agentreplay_render_html", "description": "Render HTML visual dossier"},
                            ]
                        },
                    }
                elif method == "tools/call":
                    tool_name = params.get("name")
                    args = params.get("arguments", {})
                    out = self._dispatch_tool(tool_name, args)
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(out)}]},
                    }
                else:
                    res = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_res = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}}
                sys.stdout.write(json.dumps(err_res) + "\n")
                sys.stdout.flush()

    def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "agentreplay_create_recording":
            return self.create_recording(
                agent_name=args.get("agent_name", "agent"),
                framework=args.get("framework", "mcp"),
            )
        elif name == "agentreplay_record_step":
            return self.record_step(
                recording_id=args["recording_id"],
                step_type=args.get("step_type", "thought"),
                content=args.get("content", ""),
                tool_name=args.get("tool_name"),
                tool_args=args.get("tool_args"),
                tool_result=args.get("tool_result"),
                tool_error=args.get("tool_error"),
            )
        elif name == "agentreplay_export_flight":
            return self.export_flight(args["recording_id"], args["output_path"])
        elif name == "agentreplay_load_flight":
            return self.load_flight(args["filepath"])
        elif name == "agentreplay_time_travel":
            return self.time_travel(args["recording_id"], args["action"], args.get("step_index"))
        elif name == "agentreplay_fork_counterfactual":
            return self.fork_counterfactual(
                args["base_recording_id"],
                int(args["at_step"]),
                args["branch_name"],
                args.get("override_tool_result"),
            )
        elif name == "agentreplay_diff_trajectories":
            return self.diff_trajectories(args["base_id"], args["compare_id"])
        elif name == "agentreplay_verify_determinism":
            return self.verify_determinism(args["recording_id"])
        elif name == "agentreplay_render_html":
            return self.render_html(args["recording_id"], args["output_path"])
        return {"status": "error", "error": f"Unknown tool '{name}'"}
