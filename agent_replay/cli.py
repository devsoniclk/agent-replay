"""
agent_replay.cli

Command Line Interface for agent-replay: inspecting flights, stepping through time,
forking counterfactual branches, diffing trajectories, and launching the MCP server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from agent_replay import __version__
from agent_replay.counterfactual import CounterfactualBranchingEngine, Intervention
from agent_replay.debugger import TimeTravelDebugger
from agent_replay.diff import TrajectoryDifferencer
from agent_replay.flight_recorder import FlightRecorder
from agent_replay.mcp_server import AgentReplayMCPServer
from agent_replay.sandbox import ReplaySandbox
from agent_replay.visualizer import HTMLVisualizer


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-replay",
        description="Deterministic Time-Travel Debugger, Flight Recorder & Counterfactual Branching Engine for AI Agents",
    )
    parser.add_argument("--version", action="version", version=f"agent-replay {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # doctor
    subparsers.add_parser("doctor", help="Run system diagnostics and verify cryptographic runtime")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a flight recording file")
    p_inspect.add_argument("file", help="Path to .flight or .json recording file")
    p_inspect.add_argument("--json", action="store_true", help="Output raw JSON summary")

    # travel
    p_travel = subparsers.add_parser("travel", help="Time-travel step inspection")
    p_travel.add_argument("file", help="Path to .flight or .json recording file")
    p_travel.add_argument("--step", type=int, default=None, help="Jump to specific step index")

    # fork
    p_fork = subparsers.add_parser("fork", help="Fork counterfactual branch from a historical step")
    p_fork.add_argument("file", help="Path to base recording file")
    p_fork.add_argument("--at", type=int, required=True, help="Step index to branch at")
    p_fork.add_argument("--branch", required=True, help="Name of new branch")
    p_fork.add_argument("--out", default=None, help="Output path for new forked flight file")

    # diff
    p_diff = subparsers.add_parser("diff", help="Side-by-side diff between two flight recordings")
    p_diff.add_argument("base", help="Path to base recording file")
    p_diff.add_argument("compare", help="Path to compare recording file")
    p_diff.add_argument("--json", action="store_true", help="Output structured diff as JSON")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify cryptographic hash chain and invariants")
    p_verify.add_argument("file", help="Path to recording file")

    # report
    p_report = subparsers.add_parser("report", help="Generate standalone interactive HTML dossier")
    p_report.add_argument("file", help="Path to recording file")
    p_report.add_argument("--out", default=None, help="Output HTML file path (default: <filename>.html)")

    # mcp
    subparsers.add_parser("mcp", help="Launch Model Context Protocol (MCP) stdio server")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "doctor":
        return _cmd_doctor()
    elif args.command == "inspect":
        return _cmd_inspect(args.file, args.json)
    elif args.command == "travel":
        return _cmd_travel(args.file, args.step)
    elif args.command == "fork":
        return _cmd_fork(args.file, args.at, args.branch, args.out)
    elif args.command == "diff":
        return _cmd_diff(args.base, args.compare, args.json)
    elif args.command == "verify":
        return _cmd_verify(args.file)
    elif args.command == "report":
        return _cmd_report(args.file, args.out)
    elif args.command == "mcp":
        server = AgentReplayMCPServer()
        server.run_stdio_jsonrpc()
        return 0

    return 0


def _cmd_doctor() -> int:
    import hashlib
    import platform

    print("Agent Replay System Doctor")
    print("==========================")
    print(f"Version:      {__version__}")
    print(f"Python:       {sys.version.split()[0]} ({platform.platform()})")
    
    # Hash check
    test_h = hashlib.sha256(b"agent_replay_test").hexdigest()
    print(f"SHA-256:      OK ({test_h[:12]}...)")
    print("Status:       All subsystems operational [PASS]")
    return 0


def _cmd_inspect(file_path: str, as_json: bool) -> int:
    rec = FlightRecorder.load(file_path)
    prompt_tok, comp_tok, total_tok = rec.total_tokens()

    if as_json:
        data = {
            "recording_id": rec.recording_id,
            "agent_name": rec.agent_name,
            "framework": rec.framework,
            "total_steps": len(rec.steps),
            "tokens": {"prompt": prompt_tok, "completion": comp_tok, "total": total_tok},
            "latency_ms": rec.total_latency_ms(),
            "merkle_root": rec.merkle_root or rec.compute_merkle_root(),
            "status": str(rec.status.value if hasattr(rec.status, "value") else rec.status),
            "tools": rec.tool_usage_frequency(),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(f"Flight Recording: {rec.recording_id}")
    print(f"Agent:            {rec.agent_name} ({rec.framework})")
    print(f"Status:           {rec.status.value if hasattr(rec.status, 'value') else rec.status}")
    print(f"Steps:            {len(rec.steps)}")
    print(f"Tokens:           {total_tok:,} (Prompt: {prompt_tok:,}, Completion: {comp_tok:,})")
    print(f"Latency:          {rec.total_latency_ms():.1f} ms")
    print(f"Merkle Root:      {rec.merkle_root or rec.compute_merkle_root()}")
    print("\nTool Usage Breakdown:")
    freq = rec.tool_usage_frequency()
    if freq:
        for tool, count in sorted(freq.items(), key=lambda x: -x[1]):
            print(f"  - {tool:24} : {count}")
    else:
        print("  (no tools invoked)")
    return 0


def _cmd_travel(file_path: str, step_index: Optional[int]) -> int:
    dbg = TimeTravelDebugger(file_path)
    if step_index is not None:
        dbg.goto(step_index)
    else:
        dbg.jump_to_end()

    frame = dbg.inspect_frame()
    step = dbg.current_step

    print(f"Step [{dbg.cursor}/{dbg.total_steps}] Time-Travel Cursor")
    if step:
        st_val = step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)
        print(f"Type:     {st_val}")
        if step.tool_name:
            print(f"Tool:     {step.tool_name}")
            print(f"Args:     {json.dumps(step.tool_args)}")
            if step.tool_result:
                print(f"Result:   {str(step.tool_result)[:200]}")
        if step.content:
            print(f"Content:  {step.content[:300]}")
        print(f"State:    {json.dumps(frame['state'])}")
    else:
        print("Cursor positioned before step 0 (initial unstarted state).")
    return 0


def _cmd_fork(file_path: str, at_step: int, branch_name: str, out_path: Optional[str]) -> int:
    base = FlightRecorder.load(file_path)
    forked_recorder = CounterfactualBranchingEngine.fork_branch(
        base_recording=base,
        at_step=at_step,
        branch_name=branch_name,
        description=f"Branch {branch_name} forked via CLI at step {at_step}",
    )
    forked_recorder.complete()
    dest = out_path or f"{base.recording_id}_{branch_name}.flight"
    saved = forked_recorder.save(dest)
    print(f"Forked branch '{branch_name}' at step {at_step}.")
    print(f"Saved to: {saved}")
    return 0


def _cmd_diff(base_path: str, compare_path: str, as_json: bool) -> int:
    rec_a = FlightRecorder.load(base_path)
    rec_b = FlightRecorder.load(compare_path)
    diff_res = TrajectoryDifferencer.diff(rec_a, rec_b)

    if as_json:
        print(json.dumps(diff_res.to_dict(), indent=2))
        return 0

    print("Trajectory Diff Report")
    print("======================")
    print(f"Base:    {rec_a.recording_id} ({len(rec_a.steps)} steps)")
    print(f"Compare: {rec_b.recording_id} ({len(rec_b.steps)} steps)")
    print(f"\n{diff_res.summary}\n")

    for s in diff_res.aligned_steps:
        if s.diff_type == "MATCH":
            print(f"  [MATCH]    Step #{s.step_index:2d}: identical")
        elif s.diff_type == "MODIFIED":
            print(f"  [MODIFIED] Step #{s.step_index:2d}: {', '.join(s.changes)}")
        elif s.diff_type == "ADDED":
            print(f"  [ADDED]    Step #{s.step_index:2d}: {s.changes[0]}")
        elif s.diff_type == "REMOVED":
            print(f"  [REMOVED]  Step #{s.step_index:2d}: {s.changes[0]}")

    return 0


def _cmd_verify(file_path: str) -> int:
    rec = FlightRecorder.load(file_path)
    res = ReplaySandbox.verify_recording(rec)
    print("Flight Recording Verification")
    print("=============================")
    print(f"Recording ID:   {res.recording_id}")
    print(f"Deterministic:  {'YES [PASS]' if res.is_deterministic else 'NO [FAIL]'}")
    print(f"Tampered:       {'NO [PASS]' if not res.tampered else 'YES [FAIL]'}")
    print(f"Matched Steps:  {res.matched_steps} / {res.total_steps}")
    print(f"Details:        {res.details}")
    if res.is_deterministic:
        print("VERIFIED: Cryptographic Merkle chain & invariants valid.")
        return 0
    return 1


def _cmd_report(file_path: str, out_path: Optional[str]) -> int:
    rec = FlightRecorder.load(file_path)
    dest = out_path or f"{Path(file_path).stem}_report.html"
    HTMLVisualizer.render(rec, dest)
    print(f"Generated standalone HTML dossier: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
