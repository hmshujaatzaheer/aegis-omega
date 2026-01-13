"""
AEGIS-Ω Core Module
===================

The main orchestrator for the Universal AI Safety Protocol.

This module provides:
- AEGISOmega: Main orchestrator integrating all four contributions
- StreamingMFOTLMonitor: O(B × |φ|^d) bounded-memory runtime verification
- MFOTLToCircuitCompiler: Compile MFOTL formulas to arithmetic circuits
- SafetyHandshake: AI-to-AI safety negotiation protocol

Usage:
    from aegis_omega.core import AEGISOmega, create_aegis
    
    aegis = create_aegis()
    verdict = aegis.process_action(action)
"""

from aegis_omega.core.aegis import (
    # Main orchestrator
    AEGISOmega,
    create_aegis,
    create_aegis_for_eu_ai_act,
    
    # Types and enums
    SafetyLevel,
    EnforcementAction,
    ProtocolPhase,
    Timestamp,
    
    # Actions and verdicts
    AIAction,
    SafetyVerdict,
    SafetyCertificate,
    
    # MFOTL formulas
    MFOTLFormula,
    Predicate,
    Negation,
    Conjunction,
    Disjunction,
    Eventually,
    Always,
    Implication,
    
    # Monitoring
    Event,
    EventTrace,
    StreamingMFOTLMonitor,
    
    # Circuit compilation
    ArithmeticCircuit,
    ConstantGate,
    InputGate,
    MultiplicationGate,
    AdditionGate,
    SubtractionGate,
    PredicateLookupGate,
    MaxOverWindowGate,
    MinOverWindowGate,
    R1CSConstraints,
    MFOTLToCircuitCompiler,
)

__all__ = [
    "AEGISOmega",
    "create_aegis",
    "create_aegis_for_eu_ai_act",
    "SafetyLevel",
    "EnforcementAction",
    "ProtocolPhase",
    "Timestamp",
    "AIAction",
    "SafetyVerdict",
    "SafetyCertificate",
    "MFOTLFormula",
    "Predicate",
    "Negation",
    "Conjunction",
    "Disjunction",
    "Eventually",
    "Always",
    "Implication",
    "Event",
    "EventTrace",
    "StreamingMFOTLMonitor",
    "ArithmeticCircuit",
    "ConstantGate",
    "InputGate",
    "MultiplicationGate",
    "AdditionGate",
    "SubtractionGate",
    "PredicateLookupGate",
    "MaxOverWindowGate",
    "MinOverWindowGate",
    "R1CSConstraints",
    "MFOTLToCircuitCompiler",
]
