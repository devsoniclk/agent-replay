"""
agent-replay: Deterministic Time-Travel Debugger, Flight Recorder & Counterfactual Branching Engine for Autonomous AI Agents.
"""

from .types import (
    Breakpoint,
    DivergenceReport,
    ExecutionStatus,
    FlightRecording,
    FlightStep,
    ReplayVerificationResult,
    StepType,
)
from .flight_recorder import FlightRecorder
from .debugger import TimeTravelDebugger
from .diff import StepDiff, TrajectoryDiffResult, TrajectoryDifferencer
from .counterfactual import CounterfactualBranchingEngine, Intervention
from .sandbox import ReplaySandbox
from .visualizer import HTMLVisualizer

__version__ = "0.1.0"

__all__ = [
    "Breakpoint",
    "CounterfactualBranchingEngine",
    "DivergenceReport",
    "ExecutionStatus",
    "FlightRecorder",
    "FlightRecording",
    "FlightStep",
    "HTMLVisualizer",
    "Intervention",
    "ReplaySandbox",
    "ReplayVerificationResult",
    "StepDiff",
    "StepType",
    "TimeTravelDebugger",
    "TrajectoryDiffResult",
    "TrajectoryDifferencer",
]
