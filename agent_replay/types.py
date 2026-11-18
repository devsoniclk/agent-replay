"""
agent_replay.types

Core data models, enums, and structures for deterministic time-travel
debugging, flight recording, and counterfactual trajectory analysis.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


class StepType(str, enum.Enum):
    """Categorical classification of agent trajectory lifecycle events."""
    SYSTEM_PROMPT = "system_prompt"
    USER_INPUT = "user_input"
    OBSERVATION = "observation"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_MUTATION = "state_mutation"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"
    CUSTOM = "custom"


class ExecutionStatus(str, enum.Enum):
    """Lifecycle operational status of an agent flight recording."""
    INITIALIZED = "initialized"
    RECORDING = "recording"
    COMPLETED = "completed"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    FORKED = "forked"


@dataclass
class FlightStep:
    """An immutable atomic event recorded during an autonomous agent's lifecycle."""
    step_index: int
    step_type: StepType
    agent_id: str
    role: str
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    tool_error: Optional[str] = None
    state_delta: Optional[Dict[str, Any]] = None
    memory_snapshot: Optional[Dict[str, Any]] = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    timestamp_ns: int = field(default_factory=time.time_ns)
    environment_hash: str = ""
    parent_digest: str = ""
    digest: str = ""

    def calculate_digest(self) -> str:
        """Compute deterministic SHA-256 digest binding this step to its parent."""
        payload = {
            "step_index": self.step_index,
            "step_type": self.step_type.value if isinstance(self.step_type, StepType) else str(self.step_type),
            "agent_id": self.agent_id,
            "role": self.role,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "tool_error": self.tool_error,
            "state_delta": self.state_delta,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "environment_hash": self.environment_hash,
            "parent_digest": self.parent_digest,
        }
        canonical_json = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def seal(self) -> FlightStep:
        """Calculate and commit the cryptographic digest."""
        self.digest = self.calculate_digest()
        return self

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
        """Append a new sealed step directly to the recording."""
        last_digest = self.steps[-1].digest if self.steps else ""
        st = StepType(step_type) if isinstance(step_type, str) else step_type
        step = FlightStep(
            step_index=len(self.steps),
            step_type=st,
            agent_id=self.agent_name,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_error=tool_error,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            parent_digest=last_digest,
        ).seal()
        self.steps.append(step)
        self.compute_merkle_root()
        return step

    def record_thought(self, thought: str, tokens: int = 0, latency_ms: float = 0.0) -> FlightStep:
        """Shortcut to record an internal thought."""
        return self.record_step(StepType.THOUGHT, content=thought, tokens_completion=tokens, latency_ms=latency_ms)

    def save(self, filepath: Union[str, Any], compress: bool = True) -> str:
        """Serialize and persist recording to disk (.flight or .json)."""
        import gzip
        from pathlib import Path
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=str).encode("utf-8")
        if compress or str(p).endswith(".flight") or str(p).endswith(".gz"):
            with gzip.open(p, "wb") as f:
                f.write(payload)
        else:
            with open(p, "wb") as f:
                f.write(payload)
        return str(p.resolve())

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to serializable dictionary."""
        d = asdict(self)
        d["step_type"] = self.step_type.value if isinstance(self.step_type, StepType) else str(self.step_type)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FlightStep:
        """Reconstruct step from dictionary."""
        d = dict(data)
        if "step_type" in d and isinstance(d["step_type"], str):
            try:
                d["step_type"] = StepType(d["step_type"])
            except ValueError:
                d["step_type"] = StepType.CUSTOM
        return cls(**d)


@dataclass
class FlightRecording:
    """Complete flight recording container representing an agent mission or session."""
    recording_id: str
    session_id: str
    agent_name: str
    framework: str = "custom"
    created_at: str = ""
    completed_at: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.INITIALIZED
    metadata: Dict[str, Any] = field(default_factory=dict)
    environment_baseline: Dict[str, Any] = field(default_factory=dict)
    steps: List[FlightStep] = field(default_factory=list)
    merkle_root: str = ""
    parent_recording_id: Optional[str] = None
    fork_step_index: Optional[int] = None
    branch_name: str = "main"

    def compute_merkle_root(self) -> str:
        """Calculate cryptographic Merkle root across all recorded step digests."""
        if not self.steps:
            return hashlib.sha256(b"empty_recording").hexdigest()

        digests = [s.digest or s.calculate_digest() for s in self.steps]
        
        # Build Merkle tree
        current_layer = [d.encode("utf-8") if isinstance(d, str) else d for d in digests]
        while len(current_layer) > 1:
            next_layer = []
            for i in range(0, len(current_layer), 2):
                left = current_layer[i]
                right = current_layer[i + 1] if i + 1 < len(current_layer) else left
                combined = hashlib.sha256(left + right).digest()
                next_layer.append(combined)
            current_layer = next_layer
        
        self.merkle_root = current_layer[0].hex() if isinstance(current_layer[0], bytes) else current_layer[0]
        return self.merkle_root

    def verify_integrity(self) -> tuple[bool, Optional[int], str]:
        """Verify the cryptographic hash chain and digests of all steps.
        
        Returns (is_valid, failing_step_index, message).
        """
        if not self.steps:
            return True, None, "Empty recording is valid"

        expected_parent = ""
        for i, step in enumerate(self.steps):
            if step.parent_digest != expected_parent:
                return (
                    False,
                    i,
                    f"Hash chain broken at step {i}: parent_digest '{step.parent_digest}' "
                    f"does not match previous digest '{expected_parent}'",
                )
            computed_digest = step.calculate_digest()
            if step.digest and step.digest != computed_digest:
                return (
                    False,
                    i,
                    f"Step {i} digest tampered: stored '{step.digest}', computed '{computed_digest}'",
                )
            expected_parent = step.digest or computed_digest

        stored_merkle = self.merkle_root
        computed_merkle = self.compute_merkle_root()
        if stored_merkle and stored_merkle != computed_merkle:
            self.merkle_root = stored_merkle
            return False, None, f"Merkle root mismatch: stored '{stored_merkle}', computed '{computed_merkle}'"

        return True, None, "Cryptographic integrity verified"

    def total_tokens(self) -> tuple[int, int, int]:
        """Return (prompt_tokens, completion_tokens, total_tokens)."""
        prompt = sum(s.tokens_prompt for s in self.steps)
        completion = sum(s.tokens_completion for s in self.steps)
        return prompt, completion, prompt + completion

    def total_latency_ms(self) -> float:
        """Calculate total recorded latency in milliseconds."""
        return sum(s.latency_ms for s in self.steps)

    def tool_usage_frequency(self) -> Dict[str, int]:
        """Count frequency of each tool invoked during the flight."""
        freq: Dict[str, int] = {}
        for s in self.steps:
            if s.tool_name:
                freq[s.tool_name] = freq.get(s.tool_name, 0) + 1
        return freq

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
        """Append a new sealed step directly to the recording."""
        last_digest = self.steps[-1].digest if self.steps else ""
        st = StepType(step_type) if isinstance(step_type, str) else step_type
        step = FlightStep(
            step_index=len(self.steps),
            step_type=st,
            agent_id=self.agent_name,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_error=tool_error,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            parent_digest=last_digest,
        ).seal()
        self.steps.append(step)
        self.compute_merkle_root()
        return step

    def record_thought(self, thought: str, tokens: int = 0, latency_ms: float = 0.0) -> FlightStep:
        """Shortcut to record an internal thought."""
        return self.record_step(StepType.THOUGHT, content=thought, tokens_completion=tokens, latency_ms=latency_ms)

    def save(self, filepath: Union[str, Any], compress: bool = True) -> str:
        """Serialize and persist recording to disk (.flight or .json)."""
        import gzip
        from pathlib import Path
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=str).encode("utf-8")
        if compress or str(p).endswith(".flight") or str(p).endswith(".gz"):
            with gzip.open(p, "wb") as f:
                f.write(payload)
        else:
            with open(p, "wb") as f:
                f.write(payload)
        return str(p.resolve())

    def to_dict(self) -> Dict[str, Any]:
        """Convert recording to serializable dictionary."""
        return {
            "recording_id": self.recording_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "framework": self.framework,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status.value if isinstance(self.status, ExecutionStatus) else str(self.status),
            "metadata": self.metadata,
            "environment_baseline": self.environment_baseline,
            "steps": [s.to_dict() for s in self.steps],
            "merkle_root": self.merkle_root,
            "parent_recording_id": self.parent_recording_id,
            "fork_step_index": self.fork_step_index,
            "branch_name": self.branch_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FlightRecording:
        """Reconstruct recording from dictionary."""
        d = dict(data)
        if "status" in d and isinstance(d["status"], str):
            try:
                d["status"] = ExecutionStatus(d["status"])
            except ValueError:
                d["status"] = ExecutionStatus.COMPLETED
        if "steps" in d and isinstance(d["steps"], list):
            d["steps"] = [FlightStep.from_dict(s) for s in d["steps"]]
        return cls(**d)


@dataclass
class Breakpoint:
    """Time-travel debugger breakpoint definition."""
    breakpoint_id: str
    step_index: Optional[int] = None
    tool_name: Optional[str] = None
    step_type: Optional[StepType] = None
    condition_expr: Optional[str] = None
    hit_count: int = 0
    enabled: bool = True

    def matches(self, step: FlightStep, current_state: Dict[str, Any]) -> bool:
        """Evaluate if the breakpoint triggers on the current step."""
        if not self.enabled:
            return False

        has_criteria = False

        if self.step_index is not None:
            has_criteria = True
            if step.step_index != self.step_index:
                return False

        if self.tool_name is not None:
            has_criteria = True
            if step.tool_name != self.tool_name:
                return False

        if self.step_type is not None:
            has_criteria = True
            if step.step_type != self.step_type:
                return False

        if self.condition_expr:
            has_criteria = True
            try:
                # Safe eval with restricted globals
                ctx = {
                    "step": step.to_dict(),
                    "state": current_state,
                    "tool": step.tool_name,
                    "error": step.tool_error,
                    "content": step.content,
                }
                if not bool(eval(self.condition_expr, {"__builtins__": {}}, ctx)):
                    return False
            except Exception:
                return False

        return has_criteria


@dataclass
class DivergenceReport:
    """Comparative trajectory divergence analysis between two flights or counterfactual branches."""
    base_recording_id: str
    fork_recording_id: str
    divergence_step_index: Optional[int]
    total_steps_base: int
    total_steps_fork: int
    tool_sequence_similarity: float
    state_divergence_keys: List[str]
    token_delta: int
    latency_delta_ms: float
    tool_deltas: Dict[str, Dict[str, int]]
    verdict: str

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
        """Append a new sealed step directly to the recording."""
        last_digest = self.steps[-1].digest if self.steps else ""
        st = StepType(step_type) if isinstance(step_type, str) else step_type
        step = FlightStep(
            step_index=len(self.steps),
            step_type=st,
            agent_id=self.agent_name,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_error=tool_error,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            parent_digest=last_digest,
        ).seal()
        self.steps.append(step)
        self.compute_merkle_root()
        return step

    def record_thought(self, thought: str, tokens: int = 0, latency_ms: float = 0.0) -> FlightStep:
        """Shortcut to record an internal thought."""
        return self.record_step(StepType.THOUGHT, content=thought, tokens_completion=tokens, latency_ms=latency_ms)

    def save(self, filepath: Union[str, Any], compress: bool = True) -> str:
        """Serialize and persist recording to disk (.flight or .json)."""
        import gzip
        from pathlib import Path
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=str).encode("utf-8")
        if compress or str(p).endswith(".flight") or str(p).endswith(".gz"):
            with gzip.open(p, "wb") as f:
                f.write(payload)
        else:
            with open(p, "wb") as f:
                f.write(payload)
        return str(p.resolve())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayVerificationResult:
    """Verification outcome of replaying a recording in the deterministic sandbox."""
    recording_id: str
    is_deterministic: bool
    tampered: bool
    matched_steps: int
    total_steps: int
    divergent_step_indices: List[int]
    invariant_violations: List[Dict[str, Any]]
    details: str

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
        """Append a new sealed step directly to the recording."""
        last_digest = self.steps[-1].digest if self.steps else ""
        st = StepType(step_type) if isinstance(step_type, str) else step_type
        step = FlightStep(
            step_index=len(self.steps),
            step_type=st,
            agent_id=self.agent_name,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_error=tool_error,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            parent_digest=last_digest,
        ).seal()
        self.steps.append(step)
        self.compute_merkle_root()
        return step

    def record_thought(self, thought: str, tokens: int = 0, latency_ms: float = 0.0) -> FlightStep:
        """Shortcut to record an internal thought."""
        return self.record_step(StepType.THOUGHT, content=thought, tokens_completion=tokens, latency_ms=latency_ms)

    def save(self, filepath: Union[str, Any], compress: bool = True) -> str:
        """Serialize and persist recording to disk (.flight or .json)."""
        import gzip
        from pathlib import Path
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=str).encode("utf-8")
        if compress or str(p).endswith(".flight") or str(p).endswith(".gz"):
            with gzip.open(p, "wb") as f:
                f.write(payload)
        else:
            with open(p, "wb") as f:
                f.write(payload)
        return str(p.resolve())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
