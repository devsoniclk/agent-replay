"""
agent_replay.flight_recorder

High-fidelity black-box flight recorder capturing multi-step agent trajectories,
cryptographic hash-chaining, environment snapshots, and serialization.
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agent_replay.types import ExecutionStatus, FlightRecording, FlightStep, StepType


class FlightRecorder:
    """Black-box flight recorder for capturing autonomous agent executions."""

    def __init__(
        self,
        agent_name: str = "default_agent",
        session_id: Optional[str] = None,
        framework: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
        storage_dir: Optional[Union[str, Path]] = None,
        auto_seal: bool = True,
    ) -> None:
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())
        self.recording_id = f"flight_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.framework = framework
        self.metadata = metadata or {}
        self.storage_dir = Path(storage_dir) if storage_dir else Path.cwd() / ".flights"
        self.auto_seal = auto_seal

        # Capture environmental baseline
        self.environment_baseline = self._capture_environment_baseline()
        self.env_hash = self._compute_env_hash(self.environment_baseline)

        self.recording = FlightRecording(
            recording_id=self.recording_id,
            session_id=self.session_id,
            agent_name=self.agent_name,
            framework=self.framework,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status=ExecutionStatus.RECORDING,
            metadata=self.metadata,
            environment_baseline=self.environment_baseline,
            steps=[],
            branch_name="main",
        )
        self._current_step_index = 0
        self._last_digest = ""
        self._current_state: Dict[str, Any] = {}

    def _capture_environment_baseline(self) -> Dict[str, Any]:
        """Capture host runtime environment baseline for deterministic replay context."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "os_name": os.name,
            "processor": platform.processor(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def _compute_env_hash(self, env: Dict[str, Any]) -> str:
        """Compute compact SHA-256 hash of environmental baseline."""
        raw = json.dumps(env, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def record_step(
        self,
        step_type: Union[StepType, str],
        content: str = "",
        role: str = "assistant",
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Any] = None,
        tool_error: Optional[str] = None,
        state_delta: Optional[Dict[str, Any]] = None,
        memory_snapshot: Optional[Dict[str, Any]] = None,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: float = 0.0,
    ) -> FlightStep:
        """Record an arbitrary trajectory step and append it to the hash chain."""
        if isinstance(step_type, str):
            try:
                st = StepType(step_type)
            except ValueError:
                st = StepType.CUSTOM
        else:
            st = step_type

        # Update in-memory state tracking if state_delta provided
        if state_delta:
            self._current_state.update(state_delta)

        step = FlightStep(
            step_index=self._current_step_index,
            step_type=st,
            agent_id=self.agent_name,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_error=tool_error,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot or (dict(self._current_state) if self._current_state else None),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            timestamp_ns=time.time_ns(),
            environment_hash=self.env_hash,
            parent_digest=self._last_digest,
        )

        if self.auto_seal:
            step.seal()
            self._last_digest = step.digest

        self.recording.steps.append(step)
        self._current_step_index += 1
        return step

    def record_thought(self, thought: str, tokens: int = 0, latency_ms: float = 0.0) -> FlightStep:
        """Shortcut to record an internal cognitive deliberation thought."""
        return self.record_step(
            step_type=StepType.THOUGHT,
            content=thought,
            tokens_completion=tokens,
            latency_ms=latency_ms,
        )

    def record_observation(self, observation: str, source: str = "environment") -> FlightStep:
        """Shortcut to record an environmental observation or perceived event."""
        return self.record_step(
            step_type=StepType.OBSERVATION,
            content=observation,
            role=source,
        )

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tokens: int = 0,
        latency_ms: float = 0.0,
        tool_output: Optional[Any] = None,
    ) -> FlightStep:
        """Shortcut to record an outbound tool call request."""
        return self.record_step(
            step_type=StepType.TOOL_CALL,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_output,
            tokens_prompt=tokens,
            latency_ms=latency_ms,
        )

    def record_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> FlightStep:
        """Shortcut to record inbound tool response or error."""
        return self.record_step(
            step_type=StepType.TOOL_RESULT,
            tool_name=tool_name,
            tool_result=tool_result,
            tool_error=error,
            latency_ms=latency_ms,
        )

    def record_state_mutation(
        self,
        delta_or_key: Union[Dict[str, Any], str],
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        memory_snapshot: Optional[Dict[str, Any]] = None,
    ) -> FlightStep:
        """Record state mutation or memory state update."""
        if isinstance(delta_or_key, dict):
            delta = delta_or_key
        else:
            delta = {str(delta_or_key): new_value}
        return self.record_step(
            step_type=StepType.STATE_MUTATION,
            state_delta=delta,
            memory_snapshot=memory_snapshot,
        )

    def record_final_response(
        self,
        response: str = "",
        content: str = "",
        tokens_prompt: int = 0,
        prompt_tokens: int = 0,
        tokens_completion: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        status: Optional[ExecutionStatus] = None,
    ) -> FlightStep:
        """Record the agent's final delivered response to user or caller."""
        text = response or content
        ptok = tokens_prompt or prompt_tokens
        ctok = tokens_completion or completion_tokens
        step = self.record_step(
            step_type=StepType.FINAL_RESPONSE,
            content=text,
            tokens_prompt=ptok,
            tokens_completion=ctok,
            latency_ms=latency_ms,
        )
        if status:
            self.recording.status = status
        return step

    def record_error(self, message: str, exception_type: Optional[str] = None) -> FlightStep:
        """Record an unhandled exception or agent error event."""
        content = f"{exception_type}: {message}" if exception_type else message
        return self.record_step(
            step_type=StepType.ERROR,
            content=content,
            tool_error=message,
        )

    def complete(self, status: ExecutionStatus = ExecutionStatus.COMPLETED) -> FlightRecording:
        """Conclude the recording session and calculate the cryptographic Merkle root."""
        self.recording.status = status
        self.recording.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.recording.compute_merkle_root()
        return self.recording

    def save(self, filepath: Optional[Union[str, Path]] = None, compress: bool = True) -> str:
        """Serialize and persist recording to disk (.flight or .json)."""
        if self.recording.status == ExecutionStatus.RECORDING:
            self.complete()

        if filepath is None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            ext = ".flight" if compress else ".json"
            filepath = self.storage_dir / f"{self.recording_id}{ext}"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        payload_bytes = json.dumps(self.recording.to_dict(), indent=2, default=str).encode("utf-8")

        if compress or str(filepath).endswith(".flight") or str(filepath).endswith(".gz"):
            with gzip.open(filepath, "wb") as f:
                f.write(payload_bytes)
        else:
            with open(filepath, "wb") as f:
                f.write(payload_bytes)

        return str(filepath.resolve())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> FlightRecording:
        """Load and deserialize a flight recording from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Recording file not found: {path}")

        try:
            with gzip.open(path, "rb") as f:
                content = f.read().decode("utf-8")
        except (gzip.BadGzipFile, OSError):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        data = json.loads(content)
        return FlightRecording.from_dict(data)

    def __enter__(self) -> FlightRecorder:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.record_error(str(exc_val), exception_type=exc_type.__name__)
            self.complete(status=ExecutionStatus.FAILED)
        else:
            self.complete(status=ExecutionStatus.COMPLETED)
