"""
agent_replay.debugger

Interactive time-travel debugger for stepping forwards and backwards through
recorded agent executions, evaluating breakpoints, and inspecting state frames.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import Breakpoint, FlightRecording, FlightStep, StepType


class TimeTravelDebugger:
    """Time-travel debugger navigating recorded agent execution trajectories."""

    def __init__(self, recording: Union[FlightRecording, str, Path]) -> None:
        if isinstance(recording, (str, Path)):
            self.recording = FlightRecorder.load(recording)
        else:
            self.recording = recording

        self.cursor: int = -1  # -1 represents before the first step
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.watchpoints: Dict[str, Any] = {}  # key_path -> last_known_value
        self._state_cache: Dict[int, Dict[str, Any]] = {}

    @property
    def total_steps(self) -> int:
        return len(self.recording.steps)

    @property
    def current_step(self) -> Optional[FlightStep]:
        if 0 <= self.cursor < self.total_steps:
            return self.recording.steps[self.cursor]
        return None

    def reset(self) -> None:
        """Reset cursor to initial pre-execution state."""
        self.cursor = -1

    def step_forward(self) -> Optional[FlightStep]:
        """Advance cursor by one step forward."""
        if self.cursor + 1 < self.total_steps:
            self.cursor += 1
            return self.current_step
        return None

    def step_backward(self) -> Optional[FlightStep]:
        """Rewind cursor by one step backward."""
        if self.cursor > 0:
            self.cursor -= 1
            return self.current_step
        elif self.cursor == 0:
            self.cursor = -1
            return None
        return None

    def goto(self, step_index: int) -> Optional[FlightStep]:
        """Jump cursor directly to a specific step index."""
        if -1 <= step_index < self.total_steps:
            self.cursor = step_index
            return self.current_step
        raise IndexError(f"Step index {step_index} out of bounds [-1, {self.total_steps - 1}]")

    def jump_to_end(self) -> Optional[FlightStep]:
        """Jump directly to the final recorded step."""
        if self.total_steps > 0:
            self.cursor = self.total_steps - 1
            return self.current_step
        return None

    def add_breakpoint(
        self,
        step_index: Optional[int] = None,
        tool_name: Optional[str] = None,
        step_type: Optional[Union[StepType, str]] = None,
        condition_expr: Optional[str] = None,
    ) -> Breakpoint:
        """Register a new execution breakpoint."""
        bp_id = f"bp_{uuid.uuid4().hex[:6]}"
        st = None
        if step_type:
            st = StepType(step_type) if isinstance(step_type, str) else step_type

        bp = Breakpoint(
            breakpoint_id=bp_id,
            step_index=step_index,
            tool_name=tool_name,
            step_type=st,
            condition_expr=condition_expr,
        )
        self.breakpoints[bp_id] = bp
        return bp

    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove an existing breakpoint by ID."""
        return self.breakpoints.pop(breakpoint_id, None) is not None

    def clear_breakpoints(self) -> None:
        """Clear all active breakpoints."""
        self.breakpoints.clear()

    def continue_execution(self) -> Optional[FlightStep]:
        """Advance forward until a breakpoint matches or trajectory terminates."""
        while self.cursor + 1 < self.total_steps:
            next_step = self.recording.steps[self.cursor + 1]
            state = self.get_state_at(self.cursor + 1)
            
            # Check breakpoints
            for bp in self.breakpoints.values():
                if bp.matches(next_step, state):
                    bp.hit_count += 1
                    self.cursor += 1
                    return next_step

            self.cursor += 1

        return self.current_step

    def reverse_continue(self) -> Optional[FlightStep]:
        """Rewind backward until a breakpoint matches or beginning is reached."""
        while self.cursor > 0:
            prev_step = self.recording.steps[self.cursor - 1]
            state = self.get_state_at(self.cursor - 1)
            
            for bp in self.breakpoints.values():
                if bp.matches(prev_step, state):
                    bp.hit_count += 1
                    self.cursor -= 1
                    return prev_step

            self.cursor -= 1

        if self.cursor == 0:
            self.cursor = -1
        return None

    def get_state_at(self, step_index: int) -> Dict[str, Any]:
        """Reconstruct state snapshot at step_index by rolling forward deltas."""
        if step_index < 0:
            return {}
        if step_index in self._state_cache:
            return dict(self._state_cache[step_index])

        # Find closest cached step
        start_idx = 0
        state: Dict[str, Any] = {}
        cached_keys = [k for k in self._state_cache.keys() if k <= step_index]
        if cached_keys:
            start_idx = max(cached_keys)
            state = dict(self._state_cache[start_idx])
            start_idx += 1

        for i in range(start_idx, min(step_index + 1, self.total_steps)):
            s = self.recording.steps[i]
            if s.memory_snapshot:
                state.update(s.memory_snapshot)
            elif s.state_delta:
                state.update(s.state_delta)

        self._state_cache[step_index] = dict(state)
        return state

    def get_state_at_cursor(self) -> Dict[str, Any]:
        """Return state snapshot at current cursor location."""
        return self.get_state_at(self.cursor)

    def get_tool_history(self) -> List[FlightStep]:
        """List all tool calls and results up to the current cursor position."""
        if self.cursor < 0:
            return []
        return [
            s
            for s in self.recording.steps[: self.cursor + 1]
            if s.step_type in (StepType.TOOL_CALL, StepType.TOOL_RESULT)
        ]

    def inspect_frame(self) -> Dict[str, Any]:
        """Produce rich diagnostics frame of current execution state."""
        step = self.current_step
        state = self.get_state_at_cursor()
        return {
            "cursor": self.cursor,
            "total_steps": self.total_steps,
            "step": step.to_dict() if step else None,
            "state": state,
            "active_breakpoints": len(self.breakpoints),
            "at_start": self.cursor <= 0,
            "at_end": self.cursor >= self.total_steps - 1,
        }
