# agent-replay

**Deterministic Time-Travel Debugger, Flight Recorder & Counterfactual Branching Engine for Autonomous AI Agents**

[![CI](https://github.com/devsoniclk/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/devsoniclk/agent-replay/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://python.org)

---

## 1. The Problem

Debugging multi-step autonomous AI agents is notoriously difficult:
- **Non-deterministic drift:** Re-running an agent prompt with temperature produces entirely different tool invocation sequences and reasoning paths.
- **Irreversible side-effects:** Agents mutate filesystems, initiate database writes, and invoke third-party APIs. Attempting to reproduce a crash at step 38 by starting from step 1 corrupts external state.
- **State blindness:** When an agent produces an erroneous conclusion, developers cannot inspect the exact internal state vector, working memory, or tool output payload at intermediate steps.
- **Passive telemetry limits:** Modern observability platforms (LangSmith, Langfuse, Phoenix) act as passive log collectors. They cannot step backwards through time, pause execution at conditional breakpoints, or fork "what-if" counterfactual scenarios from step 12 without re-executing steps 1 through 11.

---

## 2. The Solution: agent-replay

`agent-replay` introduces deterministic flight recording and interactive time-travel execution replay for AI agent workflows:

1. **Cryptographic Flight Recorder:** Captures every reasoning deliberation, prompt, tool parameter, response payload, and memory mutation into an immutable `.flight` archive sealed with SHA-256 parent-hash chains and Merkle root verification.
2. **Deterministic Time-Travel Debugger:** Move backwards and forwards through an agent's execution history (`step_forward()`, `step_backward()`, `goto(k)`). Reconstruct exact memory states at any historic frame with zero live LLM re-sampling.
3. **Counterfactual Branching:** Branch execution at step $k$, inject custom tool outputs or modified observations, and observe how downstream trajectory paths diverge from the historical baseline.
4. **Trajectory Diffing Engine:** Quantify divergences across runs with structural matching (`MATCH`, `MODIFIED`, `ADDED`, `REMOVED`) and token/latency delta metrics.
5. **Universal Framework Adapters & MCP Server:** Plug-and-play decorators for LangChain, CrewAI, AutoGen, and native Python agents, plus an integrated Model Context Protocol (MCP) server for Claude Desktop and OpenBot.

---

## 3. Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Autonomous Agent Run                     │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Decorators / SDK)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  FlightRecorder Subsystem                   │
│   • Step captures (Thought, Tool Call, Result, State Delta) │
│   • SHA-256 Parent Digest Hash Chaining                     │
│   • Merkle Root Generation & Serialization (.flight)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│     Time-Travel Debugger     │    │   Counterfactual Branching   │
│   • Bidirectional Stepping   │    │   • Historical Point Forking │
│   • State Reconstruction     │    │   • Observation Injection    │
│   • Conditional Breakpoints  │    │   • Trajectory Diffing       │
└──────────────┬───────────────┘    └──────────────┬───────────────┘
               │                                   │
               ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│            Visualizer, CLI & MCP Server Interface           │
│   • Interactive HTML Dossier (Timelines, State Trees)       │
│   • CLI: inspect, travel, fork, diff, verify, report        │
│   • JSON-RPC 2.0 stdio Model Context Protocol (MCP) Server  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Key Capabilities

| Capability | What It Delivers |
|---|---|
| **Deterministic Time Travel** | Jump to any step index, step backwards, or reverse-continue to prior breakpoints. |
| **Merkle State Proofs** | Verify whether any step or tool result was altered after recording. |
| **Counterfactual Intervention** | Simulate: *"What would the agent have done if the API returned 403 Forbidden at step 4?"* |
| **Trajectory Diff Analysis** | Compute structural step alignment and token/latency differentials between branches. |
| **Zero External Runtime Dependencies** | Core recorder, debugger, diff engine, and HTML visualizer require only standard library Python. |
| **Integrated MCP Protocol** | Expose time-travel and replay tools directly to Claude Desktop and AI IDEs. |

---

## 5. Installation

Install via pip:

```bash
pip install agent-replay
```

Or install from source with development tools:

```bash
git clone https://github.com/devsoniclk/agent-replay.git
cd agent-replay
pip install -e ".[dev]"
```

---

## 6. Quickstart Tutorial

### 6.1 Recording an Agent Session

```python
from agent_replay import FlightRecorder, ExecutionStatus

# Initialize recorder
recorder = FlightRecorder(agent_name="data-analyst", framework="langchain")

# Record thought deliberation
recorder.record_thought("Querying database for monthly active user metrics.")

# Record tool invocation and response
recorder.record_tool_call("db_query", {"query": "SELECT count(*) FROM users WHERE active = 1"})
recorder.record_tool_result("db_query", {"count": 142050})

# Record memory mutation
recorder.record_state_mutation({"active_users": 142050, "metric_verified": True})

# Record final response and seal flight archive
recorder.record_final_response(
    response="Total monthly active users: 142,050.",
    prompt_tokens=210,
    completion_tokens=42,
    status=ExecutionStatus.SUCCESS,
)

# Save sealed immutable flight recording
flight = recorder.complete()
recorder.save("session.flight")
print(f"Recorded {len(flight.steps)} steps with Merkle root {flight.merkle_root[:16]}...")
```

### 6.2 Time-Travel Debugging

```python
from agent_replay import FlightRecorder, TimeTravelDebugger, StepType

# Load recorded flight
recording = FlightRecorder.load("session.flight")
debugger = TimeTravelDebugger(recording)

# Step backward from the end
debugger.jump_to_end()
step = debugger.step_backward()

# Reconstruct exact state at any historic step
state_at_step_2 = debugger.get_state_at(2)
print("Reconstructed state at step 2:", state_at_step_2)

# Set conditional breakpoint on tool name
debugger.add_breakpoint(tool_name="db_query", step_type=StepType.TOOL_CALL)
debugger.goto(0)
hit = debugger.continue_execution()
print(f"Breakpoint hit at step {hit.step_index}: {hit.tool_name}")
```

### 6.3 Counterfactual Branching & Trajectory Diffing

```python
from agent_replay import (
    FlightRecorder,
    CounterfactualBranchingEngine,
    Intervention,
    TrajectoryDifferencer,
)

base = FlightRecorder.load("session.flight")

# Fork execution at step 1 and inject simulated error
forked_rec = CounterfactualBranchingEngine.fork_branch(
    base_recording=base,
    at_step=1,
    branch_name="db_timeout_scenario",
    intervention=Intervention(
        description="Simulate database network timeout",
        override_tool_error="Connection timed out after 30000ms",
    ),
)
forked_rec.record_thought("Database unreachable; attempting secondary replica query.")
forked_recording = forked_rec.complete()

# Compare trajectories side-by-side
diff_report = TrajectoryDifferencer.diff(base, forked_recording)
print(f"Tool similarity: {diff_report.tool_sequence_similarity * 100:.1f}%")
print(f"Divergence detected at step: {diff_report.divergence_step_index}")
```

---

## 7. Command Line Interface (CLI)

`agent-replay` comes with an integrated CLI:

```bash
# System verification and diagnostics
agent-replay doctor

# Inspect recorded flight details and token metrics
agent-replay inspect session.flight
agent-replay inspect session.flight --json

# Step through execution history interactively
agent-replay travel session.flight --step 2

# Verify Merkle cryptographic integrity and invariant compliance
agent-replay verify session.flight

# Compare two recording runs side-by-side
agent-replay diff baseline.flight forked.flight

# Generate interactive standalone HTML dossier
agent-replay report session.flight --out dossier.html

# Launch Model Context Protocol (MCP) server over stdio
agent-replay mcp
```

---

## 8. Competitor & Landscape Comparison

| Feature | `agent-replay` | LangSmith / Langfuse | Arize Phoenix | Playwright / RRWeb |
|---|:---:|:---:|:---:|:---:|
| **Target Workload** | Multi-step Agent Trajectories | LLM Trace Telemetry | Evaluation & Tracing | Browser DOM Recording |
| **Deterministic Replay** | Yes (No LLM Re-calls) | No (Telemetry only) | No | DOM Only |
| **Time-Travel Step Navigation** | Yes (`goto`, `step_back`) | No | No | Timeline scrubbing |
| **Counterfactual Branching** | Yes (Intervention engine) | No | No | No |
| **Cryptographic Merkle Proofs** | Yes (SHA-256 chain) | No | No | No |
| **Zero Dependencies Core** | Yes | No | No | No |
| **MCP Server Built-in** | Yes | No | No | No |

---

## 9. Model Context Protocol (MCP) Integration

`agent-replay` includes a compliant MCP server exposing replay and inspection tools to AI assistants:

```json
{
  "mcpServers": {
    "agent-replay": {
      "command": "python",
      "args": ["-m", "agent_replay.cli", "mcp"]
    }
  }
}
```

Available MCP tools:
- `replay_inspect`: Summary of recording ID, token costs, latency, and tool counts.
- `replay_step`: Inspect state and tool payloads at specific step indices.
- `replay_diff`: Analyze divergence between two recordings.
- `replay_verify`: Cryptographic integrity validation.

---

## 10. Roadmap

- [x] Immutable `.flight` file format with SHA-256 parent hash chains.
- [x] Time-travel debugger with bidirectional navigation and state reconstruction.
- [x] Counterfactual branching engine with observation/tool overrides.
- [x] Self-contained HTML interactive dossier visualizer.
- [x] Built-in Model Context Protocol (MCP) stdio server.
- [ ] Integration plugins for LangGraph checkpoints.
- [ ] Automated regression test generation from counterfactual divergences.
- [ ] Distributed storage backend adapter for S3 and Cloudflare R2.

---

## 11. Contributing

Contributions are welcome. Please ensure:
1. All changes include corresponding unit tests in `tests/`.
2. Existing test suites pass (`pytest tests/ -v`).
3. Commit messages follow conventional commit formatting.

---

## 12. License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
