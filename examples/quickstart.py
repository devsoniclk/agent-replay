"""
Quickstart demonstration of agent-replay:
1. Records a multi-step autonomous agent execution trajectory.
2. Performs time-travel debugging and frame inspection.
3. Forks a counterfactual branch with an intervention at step 2.
4. Generates an interactive HTML flight dossier.
"""

from agent_replay import (
    FlightRecorder,
    TimeTravelDebugger,
    CounterfactualBranchingEngine,
    Intervention,
    HTMLVisualizer,
    StepType,
    ExecutionStatus,
)


def main():
    print("=== Agent Replay Quickstart ===")

    # 1. Record flight trajectory
    recorder = FlightRecorder(
        agent_name="financial_analyst_agent",
        framework="langchain",
        session_id="session_demo_001",
    )

    recorder.record_thought("Fetching user account balance and portfolio positions")
    recorder.record_tool_call("fetch_balance", {"account_id": "acc_9921"})
    recorder.record_tool_result("fetch_balance", {"balance_usd": 12500.0, "risk_level": "moderate"})
    recorder.record_state_mutation({"balance": 12500.0, "risk_tier": "moderate"})

    recorder.record_thought("Balance is sufficient. Executing rebalancing order.")
    recorder.record_tool_call("execute_trade", {"symbol": "NVDA", "quantity": 10, "action": "BUY"})
    recorder.record_tool_result("execute_trade", {"order_id": "ord_881", "status": "FILLED"})
    recorder.record_state_mutation({"portfolio": {"NVDA": 10}, "balance": 11300.0})

    recorder.record_final_response(
        response="Portfolio successfully rebalanced with 10 shares of NVDA.",
        prompt_tokens=420,
        completion_tokens=85,
        status=ExecutionStatus.SUCCESS,
    )
    recording = recorder.complete()
    recorder.save("session_demo.flight")
    print(f"[1] Saved flight recording: {recording.recording_id} (Merkle root: {recording.merkle_root[:16]}...)")

    # 2. Time-Travel Debugging
    debugger = TimeTravelDebugger(recording)
    print("\n[2] Stepping through time:")
    for step in recording.steps:
        s = debugger.step_forward()
        print(f"    Step [{s.step_index}]: {s.step_type.value:16} | Content: {s.content or s.tool_name or ''}")

    # Inspect state at step 3
    state_at_3 = debugger.get_state_at(3)
    print(f"    Reconstructed state at step 3: {state_at_3}")

    # 3. Counterfactual Branching
    print("\n[3] Forking counterfactual branch:")
    forked_recorder = CounterfactualBranchingEngine.fork_branch(
        base_recording=recording,
        at_step=2,
        branch_name="low_balance_scenario",
        intervention=Intervention(
            description="Simulate low balance return from banking API",
            override_tool_result={"balance_usd": 250.0, "risk_level": "conservative"},
        ),
    )
    forked_recording = forked_recorder.complete()
    print(f"    Forked branch '{forked_recording.branch_name}' (ID: {forked_recording.recording_id}) from step 2")

    # 4. Export visual dossier
    HTMLVisualizer.render(recording, output_path="dossier_demo.html")
    print("    Exported HTML flight dossier to dossier_demo.html")
    print("\n[Done] Quickstart executed successfully.")


if __name__ == "__main__":
    main()
