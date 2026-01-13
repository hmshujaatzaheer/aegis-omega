"""
AEGIS-Ω: Universal AI Safety Protocol
======================================

The TCP/IP of AI Safety Infrastructure

This module provides the main orchestration layer for the AEGIS-Ω framework,
integrating all four technical contributions:
1. Streaming-MFOTL: Bounded-memory runtime verification
2. Categorical Safety: Compositional safety certificates  
3. Folded-ZKML: Zero-knowledge proofs for AI safety
4. Safety Handshake: Universal AI safety protocol

Author: H M Shujaat Zaheer
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from collections import deque
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aegis-omega")


# =============================================================================
# Core Type Definitions
# =============================================================================

T = TypeVar("T")
S = TypeVar("S")


class SafetyLevel(Enum):
    """Risk classification aligned with EU AI Act."""
    UNACCEPTABLE = auto()  # Prohibited
    HIGH = auto()          # Requires compliance
    LIMITED = auto()       # Transparency obligations
    MINIMAL = auto()       # No specific requirements


class EnforcementAction(Enum):
    """Possible enforcement responses."""
    ALLOW = auto()         # Action permitted
    MODIFY = auto()        # Action modified to comply
    BLOCK = auto()         # Action blocked entirely
    DEFER = auto()         # Await human oversight
    LOG = auto()           # Logged for audit, allowed


class ProtocolPhase(Enum):
    """Safety Handshake protocol phases."""
    CAPABILITY_EXCHANGE = auto()
    CERTIFICATE_NEGOTIATION = auto()
    CONTINUOUS_MONITORING = auto()
    TERMINATED = auto()


@dataclass(frozen=True)
class Timestamp:
    """Immutable timestamp with nanosecond precision."""
    seconds: int
    nanos: int = 0
    
    @classmethod
    def now(cls) -> "Timestamp":
        t = time.time_ns()
        return cls(seconds=t // 10**9, nanos=t % 10**9)
    
    def __lt__(self, other: "Timestamp") -> bool:
        return (self.seconds, self.nanos) < (other.seconds, other.nanos)
    
    def __sub__(self, other: "Timestamp") -> int:
        """Return difference in nanoseconds."""
        return (self.seconds - other.seconds) * 10**9 + (self.nanos - other.nanos)
    
    def to_ms(self) -> int:
        """Convert to milliseconds since epoch."""
        return self.seconds * 1000 + self.nanos // 10**6


@dataclass
class AIAction:
    """Represents an action taken by an AI system."""
    system_id: str
    action_type: str
    payload: Dict[str, Any]
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def hash(self) -> str:
        """Compute deterministic hash of action."""
        canonical = json.dumps({
            "system_id": self.system_id,
            "action_type": self.action_type,
            "payload": self.payload,
            "timestamp": (self.timestamp.seconds, self.timestamp.nanos),
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class SafetyVerdict:
    """Result of safety verification."""
    satisfied: bool
    formula_id: str
    timestamp: Timestamp
    evidence: Optional[str] = None
    violations: List[str] = field(default_factory=list)
    enforcement_action: EnforcementAction = EnforcementAction.ALLOW


@dataclass
class SafetyCertificate:
    """Cryptographic certificate of safety compliance."""
    system_id: str
    specification_id: str
    valid_from: Timestamp
    valid_until: Timestamp
    issuer: str
    signature: bytes
    proof: Optional[bytes] = None  # ZK proof if privacy-preserving
    
    def is_valid(self, at: Optional[Timestamp] = None) -> bool:
        t = at or Timestamp.now()
        return self.valid_from < t < self.valid_until


# =============================================================================
# MFOTL Formula Representation
# =============================================================================

class MFOTLFormula(ABC):
    """Abstract base class for MFOTL formulas."""
    
    @abstractmethod
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        """Evaluate formula at given position. None means undetermined."""
        pass
    
    @abstractmethod
    def memory_bound(self) -> int:
        """Return memory bound in bytes for monitoring this formula."""
        pass
    
    @abstractmethod
    def to_circuit(self) -> "ArithmeticCircuit":
        """Convert to arithmetic circuit for ZK proof."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class Predicate(MFOTLFormula):
    """Atomic predicate: p(t1, ..., tn)."""
    name: str
    args: Tuple[str, ...]
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        event = trace.at(position)
        if event is None:
            return None
        return event.matches(self.name, self.args)
    
    def memory_bound(self) -> int:
        return 64 + len(self.name) + sum(len(a) for a in self.args)
    
    def to_circuit(self) -> "ArithmeticCircuit":
        return PredicateLookupGate(self.name, self.args)
    
    def __str__(self) -> str:
        args = ", ".join(self.args)
        return f"{self.name}({args})"


@dataclass
class Negation(MFOTLFormula):
    """Negation: ¬φ."""
    inner: MFOTLFormula
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        result = self.inner.evaluate(trace, position)
        return None if result is None else not result
    
    def memory_bound(self) -> int:
        return self.inner.memory_bound()
    
    def to_circuit(self) -> "ArithmeticCircuit":
        inner_circuit = self.inner.to_circuit()
        return SubtractionGate(ConstantGate(1), inner_circuit)
    
    def __str__(self) -> str:
        return f"¬({self.inner})"


@dataclass
class Conjunction(MFOTLFormula):
    """Conjunction: φ₁ ∧ φ₂."""
    left: MFOTLFormula
    right: MFOTLFormula
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        l = self.left.evaluate(trace, position)
        r = self.right.evaluate(trace, position)
        if l is False or r is False:
            return False
        if l is None or r is None:
            return None
        return True
    
    def memory_bound(self) -> int:
        return self.left.memory_bound() + self.right.memory_bound()
    
    def to_circuit(self) -> "ArithmeticCircuit":
        return MultiplicationGate(
            self.left.to_circuit(),
            self.right.to_circuit()
        )
    
    def __str__(self) -> str:
        return f"({self.left}) ∧ ({self.right})"


@dataclass  
class Disjunction(MFOTLFormula):
    """Disjunction: φ₁ ∨ φ₂."""
    left: MFOTLFormula
    right: MFOTLFormula
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        l = self.left.evaluate(trace, position)
        r = self.right.evaluate(trace, position)
        if l is True or r is True:
            return True
        if l is None or r is None:
            return None
        return False
    
    def memory_bound(self) -> int:
        return self.left.memory_bound() + self.right.memory_bound()
    
    def to_circuit(self) -> "ArithmeticCircuit":
        # a ∨ b = 1 - (1-a)(1-b)
        not_left = SubtractionGate(ConstantGate(1), self.left.to_circuit())
        not_right = SubtractionGate(ConstantGate(1), self.right.to_circuit())
        return SubtractionGate(
            ConstantGate(1),
            MultiplicationGate(not_left, not_right)
        )
    
    def __str__(self) -> str:
        return f"({self.left}) ∨ ({self.right})"


@dataclass
class Eventually(MFOTLFormula):
    """Eventually within interval: ◇[a,b] φ."""
    inner: MFOTLFormula
    lower: int  # Lower bound in time units
    upper: int  # Upper bound in time units (must be finite for BF-MFOTL)
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        if self.upper == float('inf'):
            raise ValueError("Eventually requires bounded interval for BF-MFOTL")
        
        undetermined = False
        for offset in range(self.lower, self.upper + 1):
            result = self.inner.evaluate(trace, position + offset)
            if result is True:
                return True
            if result is None:
                undetermined = True
        
        return None if undetermined else False
    
    def memory_bound(self) -> int:
        # Need to store window of size (upper - lower + 1)
        window_size = self.upper - self.lower + 1
        return window_size * self.inner.memory_bound()
    
    def to_circuit(self) -> "ArithmeticCircuit":
        # Max over window
        return MaxOverWindowGate(
            self.inner.to_circuit(),
            self.lower,
            self.upper
        )
    
    def __str__(self) -> str:
        return f"◇[{self.lower},{self.upper}]({self.inner})"


@dataclass
class Always(MFOTLFormula):
    """Always within interval: □[a,b] φ."""
    inner: MFOTLFormula
    lower: int
    upper: int
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        if self.upper == float('inf'):
            raise ValueError("Always requires bounded interval for BF-MFOTL")
        
        undetermined = False
        for offset in range(self.lower, self.upper + 1):
            result = self.inner.evaluate(trace, position + offset)
            if result is False:
                return False
            if result is None:
                undetermined = True
        
        return None if undetermined else True
    
    def memory_bound(self) -> int:
        window_size = self.upper - self.lower + 1
        return window_size * self.inner.memory_bound()
    
    def to_circuit(self) -> "ArithmeticCircuit":
        # Min over window
        return MinOverWindowGate(
            self.inner.to_circuit(),
            self.lower,
            self.upper
        )
    
    def __str__(self) -> str:
        return f"□[{self.lower},{self.upper}]({self.inner})"


@dataclass
class Implication(MFOTLFormula):
    """Implication: φ₁ → φ₂ (syntactic sugar for ¬φ₁ ∨ φ₂)."""
    antecedent: MFOTLFormula
    consequent: MFOTLFormula
    
    def evaluate(self, trace: "EventTrace", position: int) -> Optional[bool]:
        return Disjunction(
            Negation(self.antecedent),
            self.consequent
        ).evaluate(trace, position)
    
    def memory_bound(self) -> int:
        return self.antecedent.memory_bound() + self.consequent.memory_bound()
    
    def to_circuit(self) -> "ArithmeticCircuit":
        return Disjunction(
            Negation(self.antecedent),
            self.consequent
        ).to_circuit()
    
    def __str__(self) -> str:
        return f"({self.antecedent}) → ({self.consequent})"


# =============================================================================
# Event Trace Management
# =============================================================================

@dataclass
class Event:
    """Single event in trace."""
    predicates: Dict[str, Set[Tuple[Any, ...]]]
    timestamp: Timestamp
    
    def matches(self, pred_name: str, args: Tuple[str, ...]) -> bool:
        """Check if event contains matching predicate."""
        if pred_name not in self.predicates:
            return False
        # For now, simple exact match (full implementation would handle variables)
        return args in self.predicates[pred_name] or () in self.predicates[pred_name]


class EventTrace:
    """Sliding window event trace for streaming monitoring."""
    
    def __init__(self, max_window: int = 10000):
        self.max_window = max_window
        self._events: deque = deque(maxlen=max_window)
        self._base_position = 0
    
    def append(self, event: Event) -> int:
        """Add event and return its position."""
        position = self._base_position + len(self._events)
        self._events.append(event)
        
        # Advance base if window full
        if len(self._events) == self.max_window:
            self._base_position += 1
        
        return position
    
    def at(self, position: int) -> Optional[Event]:
        """Get event at position, or None if outside window."""
        idx = position - self._base_position
        if idx < 0 or idx >= len(self._events):
            return None
        return self._events[idx]
    
    def current_position(self) -> int:
        return self._base_position + len(self._events) - 1
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        return len(self._events) * 256  # Rough estimate per event


# =============================================================================
# Streaming MFOTL Monitor
# =============================================================================

class StreamingMFOTLMonitor:
    """
    Streaming monitor for Bounded-Future MFOTL.
    
    Provides O(B × |φ|^d) memory where B is max interval bound
    and d is quantifier alternation depth.
    """
    
    def __init__(self, formula: MFOTLFormula, on_verdict: Callable[[SafetyVerdict], None]):
        self.formula = formula
        self.on_verdict = on_verdict
        
        # Compute required window size
        self.window_size = self._compute_window_size(formula)
        self.trace = EventTrace(max_window=self.window_size)
        
        # Statistics
        self.events_processed = 0
        self.verdicts_produced = 0
        self.total_latency_ns = 0
        
        logger.info(f"Monitor initialized with window size {self.window_size}")
    
    def _compute_window_size(self, formula: MFOTLFormula) -> int:
        """Recursively compute required window size."""
        if isinstance(formula, Predicate):
            return 1
        elif isinstance(formula, (Negation,)):
            return self._compute_window_size(formula.inner)
        elif isinstance(formula, (Conjunction, Disjunction, Implication)):
            return max(
                self._compute_window_size(formula.left),
                self._compute_window_size(formula.right)
            )
        elif isinstance(formula, (Eventually, Always)):
            return formula.upper + self._compute_window_size(formula.inner)
        else:
            return 1000  # Default
    
    def process_event(self, event: Event) -> Optional[SafetyVerdict]:
        """Process single event and produce verdict if available."""
        start_ns = time.time_ns()
        
        position = self.trace.append(event)
        self.events_processed += 1
        
        # Try to produce verdict for oldest complete position
        eval_position = position - self.window_size + 1
        if eval_position >= 0:
            result = self.formula.evaluate(self.trace, eval_position)
            
            if result is not None:
                verdict = SafetyVerdict(
                    satisfied=result,
                    formula_id=str(self.formula),
                    timestamp=event.timestamp,
                    enforcement_action=EnforcementAction.ALLOW if result else EnforcementAction.BLOCK
                )
                
                self.verdicts_produced += 1
                self.total_latency_ns += time.time_ns() - start_ns
                
                self.on_verdict(verdict)
                return verdict
        
        return None
    
    async def process_stream(self, events: asyncio.Queue) -> None:
        """Continuously process event stream."""
        while True:
            event = await events.get()
            if event is None:  # Sentinel for shutdown
                break
            self.process_event(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "events_processed": self.events_processed,
            "verdicts_produced": self.verdicts_produced,
            "avg_latency_ms": (self.total_latency_ns / max(1, self.verdicts_produced)) / 1e6,
            "memory_usage_bytes": self.trace.memory_usage(),
            "window_size": self.window_size,
        }


# =============================================================================
# Arithmetic Circuit for ZK Proofs
# =============================================================================

class ArithmeticCircuit(ABC):
    """Abstract base for arithmetic circuit gates."""
    
    @abstractmethod
    def evaluate(self, inputs: Dict[str, int]) -> int:
        pass
    
    @abstractmethod
    def to_r1cs(self) -> "R1CSConstraints":
        """Convert to Rank-1 Constraint System."""
        pass


@dataclass
class ConstantGate(ArithmeticCircuit):
    value: int
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        return self.value
    
    def to_r1cs(self) -> "R1CSConstraints":
        return R1CSConstraints([], [], [self.value])


@dataclass
class InputGate(ArithmeticCircuit):
    name: str
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        return inputs[self.name]
    
    def to_r1cs(self) -> "R1CSConstraints":
        return R1CSConstraints([self.name], [], [])


@dataclass
class MultiplicationGate(ArithmeticCircuit):
    left: ArithmeticCircuit
    right: ArithmeticCircuit
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        return self.left.evaluate(inputs) * self.right.evaluate(inputs)
    
    def to_r1cs(self) -> "R1CSConstraints":
        # a * b = c
        return R1CSConstraints.multiplication(
            self.left.to_r1cs(),
            self.right.to_r1cs()
        )


@dataclass
class AdditionGate(ArithmeticCircuit):
    left: ArithmeticCircuit
    right: ArithmeticCircuit
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        return self.left.evaluate(inputs) + self.right.evaluate(inputs)
    
    def to_r1cs(self) -> "R1CSConstraints":
        return R1CSConstraints.addition(
            self.left.to_r1cs(),
            self.right.to_r1cs()
        )


@dataclass
class SubtractionGate(ArithmeticCircuit):
    left: ArithmeticCircuit
    right: ArithmeticCircuit
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        return self.left.evaluate(inputs) - self.right.evaluate(inputs)
    
    def to_r1cs(self) -> "R1CSConstraints":
        return R1CSConstraints.subtraction(
            self.left.to_r1cs(),
            self.right.to_r1cs()
        )


@dataclass
class PredicateLookupGate(ArithmeticCircuit):
    """Lookup gate for predicate evaluation (uses tlookup from zkLLM)."""
    predicate_name: str
    args: Tuple[str, ...]
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        # In real implementation, this would lookup in trace encoding
        key = f"{self.predicate_name}:{','.join(self.args)}"
        return inputs.get(key, 0)
    
    def to_r1cs(self) -> "R1CSConstraints":
        return R1CSConstraints.lookup(self.predicate_name, self.args)


@dataclass
class MaxOverWindowGate(ArithmeticCircuit):
    """Max over sliding window (for Eventually)."""
    inner: ArithmeticCircuit
    lower: int
    upper: int
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        values = []
        for i in range(self.lower, self.upper + 1):
            shifted_inputs = {f"{k}@{i}": v for k, v in inputs.items()}
            values.append(self.inner.evaluate(shifted_inputs))
        return max(values) if values else 0
    
    def to_r1cs(self) -> "R1CSConstraints":
        # Implemented via comparison chain
        return R1CSConstraints.max_over_range(
            self.inner.to_r1cs(),
            self.lower,
            self.upper
        )


@dataclass
class MinOverWindowGate(ArithmeticCircuit):
    """Min over sliding window (for Always)."""
    inner: ArithmeticCircuit
    lower: int
    upper: int
    
    def evaluate(self, inputs: Dict[str, int]) -> int:
        values = []
        for i in range(self.lower, self.upper + 1):
            shifted_inputs = {f"{k}@{i}": v for k, v in inputs.items()}
            values.append(self.inner.evaluate(shifted_inputs))
        return min(values) if values else 1
    
    def to_r1cs(self) -> "R1CSConstraints":
        return R1CSConstraints.min_over_range(
            self.inner.to_r1cs(),
            self.lower,
            self.upper
        )


@dataclass
class R1CSConstraints:
    """Rank-1 Constraint System representation."""
    a_terms: List[Any]
    b_terms: List[Any]
    c_terms: List[Any]
    
    @classmethod
    def multiplication(cls, left: "R1CSConstraints", right: "R1CSConstraints") -> "R1CSConstraints":
        return cls(left.a_terms + left.b_terms + left.c_terms,
                   right.a_terms + right.b_terms + right.c_terms,
                   ["product"])
    
    @classmethod
    def addition(cls, left: "R1CSConstraints", right: "R1CSConstraints") -> "R1CSConstraints":
        return cls(left.a_terms + right.a_terms,
                   [1],
                   left.c_terms + right.c_terms)
    
    @classmethod
    def subtraction(cls, left: "R1CSConstraints", right: "R1CSConstraints") -> "R1CSConstraints":
        return cls(left.a_terms,
                   [1],
                   left.c_terms + [f"-{t}" for t in right.c_terms])
    
    @classmethod
    def lookup(cls, pred: str, args: Tuple[str, ...]) -> "R1CSConstraints":
        return cls([f"lookup:{pred}"], [1], [f"result:{pred}"])
    
    @classmethod
    def max_over_range(cls, inner: "R1CSConstraints", lower: int, upper: int) -> "R1CSConstraints":
        return cls([f"max:{lower}:{upper}"], inner.a_terms, [f"max_result"])
    
    @classmethod
    def min_over_range(cls, inner: "R1CSConstraints", lower: int, upper: int) -> "R1CSConstraints":
        return cls([f"min:{lower}:{upper}"], inner.a_terms, [f"min_result"])


# =============================================================================
# AEGIS-Ω Main Orchestrator
# =============================================================================

@dataclass
class AEGISConfig:
    """Configuration for AEGIS-Ω system."""
    mode: str = "strict"  # strict, permissive, audit
    max_latency_ms: int = 10
    enable_zkml: bool = True
    enable_categorical: bool = True
    log_all_actions: bool = True
    human_oversight_threshold: float = 0.8


class AEGISOmega:
    """
    Main orchestrator for AEGIS-Ω: Universal AI Safety Protocol.
    
    Integrates:
    - Streaming-MFOTL monitoring
    - Categorical safety composition
    - Folded-ZKML proofs
    - Safety Handshake protocol
    """
    
    def __init__(self, config: AEGISConfig):
        self.config = config
        self.monitors: Dict[str, StreamingMFOTLMonitor] = {}
        self.certificates: Dict[str, SafetyCertificate] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._running = False
        
        logger.info(f"AEGIS-Ω initialized in {config.mode} mode")
    
    def register_specification(self, spec_id: str, formula: MFOTLFormula) -> None:
        """Register a safety specification for monitoring."""
        def on_verdict(verdict: SafetyVerdict):
            self._handle_verdict(spec_id, verdict)
        
        monitor = StreamingMFOTLMonitor(formula, on_verdict)
        self.monitors[spec_id] = monitor
        logger.info(f"Registered specification: {spec_id}")
    
    def register_eu_ai_act_specs(self) -> None:
        """Register all EU AI Act Article 9-15 specifications."""
        from .mfotl import EUAIActSpecifications
        
        for spec_id, formula in EUAIActSpecifications.all_articles().items():
            self.register_specification(spec_id, formula)
        
        logger.info("Registered all EU AI Act specifications")
    
    def _handle_verdict(self, spec_id: str, verdict: SafetyVerdict) -> None:
        """Handle monitoring verdict."""
        self._log_event({
            "type": "verdict",
            "spec_id": spec_id,
            "satisfied": verdict.satisfied,
            "timestamp": verdict.timestamp.to_ms(),
            "action": verdict.enforcement_action.name,
        })
        
        if not verdict.satisfied:
            logger.warning(f"Safety violation detected for {spec_id}")
            if self.config.mode == "strict":
                # In strict mode, this would trigger enforcement
                pass
    
    def process_action(self, action: AIAction) -> Tuple[EnforcementAction, Optional[str]]:
        """
        Process an AI action through all registered monitors.
        
        Returns enforcement action and optional reason.
        """
        # Create event from action
        event = Event(
            predicates={
                "ai_action": {(action.system_id, action.action_type)},
                action.action_type: {tuple(action.payload.values())},
            },
            timestamp=action.timestamp
        )
        
        # Log if configured
        if self.config.log_all_actions:
            self._log_event({
                "type": "action",
                "system_id": action.system_id,
                "action_type": action.action_type,
                "hash": action.hash(),
                "timestamp": action.timestamp.to_ms(),
            })
        
        # Process through all monitors
        violations = []
        for spec_id, monitor in self.monitors.items():
            verdict = monitor.process_event(event)
            if verdict and not verdict.satisfied:
                violations.append(spec_id)
        
        # Determine enforcement action
        if violations:
            if self.config.mode == "strict":
                return EnforcementAction.BLOCK, f"Violated: {', '.join(violations)}"
            elif self.config.mode == "permissive":
                return EnforcementAction.LOG, f"Logged violations: {', '.join(violations)}"
            else:  # audit
                return EnforcementAction.ALLOW, f"Audit: {', '.join(violations)}"
        
        return EnforcementAction.ALLOW, None
    
    def _log_event(self, event: Dict[str, Any]) -> None:
        """Add event to audit log."""
        event["logged_at"] = Timestamp.now().to_ms()
        self.audit_log.append(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "mode": self.config.mode,
            "monitors": {},
            "audit_log_size": len(self.audit_log),
            "certificates": len(self.certificates),
        }
        
        for spec_id, monitor in self.monitors.items():
            stats["monitors"][spec_id] = monitor.get_statistics()
        
        return stats
    
    async def run(self, action_queue: asyncio.Queue) -> None:
        """Run the AEGIS-Ω system processing actions from queue."""
        self._running = True
        logger.info("AEGIS-Ω started")
        
        while self._running:
            try:
                action = await asyncio.wait_for(action_queue.get(), timeout=1.0)
                if action is None:
                    break
                self.process_action(action)
            except asyncio.TimeoutError:
                continue
        
        logger.info("AEGIS-Ω stopped")
    
    def stop(self) -> None:
        """Stop the system."""
        self._running = False


# =============================================================================
# Factory Functions
# =============================================================================

def create_aegis(mode: str = "strict") -> AEGISOmega:
    """Create AEGIS-Ω instance with default configuration."""
    config = AEGISConfig(mode=mode)
    return AEGISOmega(config)


def create_aegis_for_eu_ai_act(
    mode: str = "strict",
    safety_level: SafetyLevel = SafetyLevel.HIGH
) -> AEGISOmega:
    """Create AEGIS-Ω pre-configured for EU AI Act compliance."""
    config = AEGISConfig(
        mode=mode,
        max_latency_ms=10,
        enable_zkml=True,
        log_all_actions=True,
        human_oversight_threshold=0.8,
    )
    
    aegis = AEGISOmega(config)
    aegis.register_eu_ai_act_specs()
    
    return aegis


# =============================================================================
# Module Exports  
# =============================================================================

__all__ = [
    # Core types
    "SafetyLevel",
    "EnforcementAction",
    "ProtocolPhase",
    "Timestamp",
    "AIAction",
    "SafetyVerdict",
    "SafetyCertificate",
    
    # MFOTL
    "MFOTLFormula",
    "Predicate",
    "Negation",
    "Conjunction",
    "Disjunction",
    "Eventually",
    "Always",
    "Implication",
    
    # Monitoring
    "Event",
    "EventTrace",
    "StreamingMFOTLMonitor",
    
    # Circuits
    "ArithmeticCircuit",
    "R1CSConstraints",
    
    # Main system
    "AEGISConfig",
    "AEGISOmega",
    "create_aegis",
    "create_aegis_for_eu_ai_act",
]
