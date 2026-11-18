"""Tests for cryptographic Merkle trees, hash chains, and tamper resistance."""

import hashlib
import json
import pytest
from agent_replay.types import FlightRecording, FlightStep, StepType


def test_empty_recording_merkle_root():
    rec = FlightRecording("rec_empty", "sess", "agent")
    root = rec.compute_merkle_root()
    assert len(root) == 64
    assert root == hashlib.sha256(b"empty_recording").hexdigest()


def test_odd_number_of_steps_merkle_root():
    rec = FlightRecording("rec_odd", "sess", "agent")
    s0 = FlightStep(0, StepType.THOUGHT, "agent", "assistant", content="t0").seal()
    s1 = FlightStep(1, StepType.THOUGHT, "agent", "assistant", content="t1", parent_digest=s0.digest).seal()
    s2 = FlightStep(2, StepType.THOUGHT, "agent", "assistant", content="t2", parent_digest=s1.digest).seal()
    rec.steps = [s0, s1, s2]
    root = rec.compute_merkle_root()
    assert len(root) == 64

    # Verification passes
    is_valid, bad, msg = rec.verify_integrity()
    assert is_valid is True


def test_merkle_root_mismatch_detection():
    rec = FlightRecording("rec_mismatch", "sess", "agent")
    s0 = FlightStep(0, StepType.THOUGHT, "agent", "assistant", content="t0").seal()
    rec.steps = [s0]
    rec.compute_merkle_root()
    
    # Forge the stored merkle root
    rec.merkle_root = "0" * 64
    is_valid, bad, msg = rec.verify_integrity()
    assert is_valid is False
    assert "Merkle root mismatch" in msg


def test_step_unrecognized_type_fallback():
    data = {
        "step_index": 0,
        "step_type": "some_future_ai_signal",
        "agent_id": "future_bot",
        "role": "model",
    }
    step = FlightStep.from_dict(data)
    assert step.step_type == StepType.CUSTOM
