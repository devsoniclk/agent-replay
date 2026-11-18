"""
agent_replay.adapters

Zero-friction framework integration adapters and Python decorators for recording
arbitrary agent workflows, tool calls, and state transitions.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional

from agent_replay.flight_recorder import FlightRecorder
from agent_replay.types import StepType


def record_agent_step(
    recorder: FlightRecorder,
    step_type: StepType = StepType.THOUGHT,
    role: str = "assistant",
) -> Callable:
    """Decorator to record a function execution as an atomic agent step."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_t = time.perf_counter()
            content = f"Invoked {func.__name__}"
            try:
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - start_t) * 1000.0
                recorder.record_step(
                    step_type=step_type,
                    role=role,
                    content=f"{content} -> returned {type(result).__name__}",
                    latency_ms=latency,
                )
                return result
            except Exception as e:
                latency = (time.perf_counter() - start_t) * 1000.0
                recorder.record_error(
                    message=f"{func.__name__} raised {type(e).__name__}: {str(e)}",
                    exception_type=type(e).__name__,
                )
                raise
        return wrapper
    return decorator


def record_tool(
    recorder: FlightRecorder,
    tool_name: Optional[str] = None,
) -> Callable:
    """Decorator to automatically intercept tool calls and their results into flight recorder."""
    def decorator(func: Callable) -> Callable:
        name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t_call = recorder.record_tool_call(
                tool_name=name,
                tool_args={"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}},
            )
            start_t = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - start_t) * 1000.0
                recorder.record_tool_result(
                    tool_name=name,
                    tool_result=result,
                    latency_ms=latency,
                )
                return result
            except Exception as e:
                latency = (time.perf_counter() - start_t) * 1000.0
                recorder.record_tool_result(
                    tool_name=name,
                    tool_result=None,
                    error=str(e),
                    latency_ms=latency,
                )
                raise
        return wrapper
    return decorator
