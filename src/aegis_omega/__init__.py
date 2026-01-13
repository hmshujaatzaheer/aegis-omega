"""
AEGIS-Ω: Universal AI Safety Protocol.

The TCP/IP of AI Safety Infrastructure.

This package provides the world's first formally verified, compositional AI safety
protocol stack, enabling mathematically guaranteed safety properties across
distributed AI systems.

Four Revolutionary Contributions:
1. Streaming-MFOTL: O(B × |φ|^d) bounded-memory runtime verification
2. Categorical Safety: Category-theoretic composition with proven guarantees
3. Folded-ZKML: Zero-knowledge proofs for AI safety compliance
4. Safety Handshake: Universal AI-to-AI safety negotiation protocol

Author: H M Shujaat Zaheer
Supervisor: Prof. David Basin, ETH Zürich
License: Apache 2.0
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujaat.zaheer@example.com"
__license__ = "Apache-2.0"

# Core AEGIS-Ω orchestrator
from aegis_omega.core.aegis import (
    AEGISOmega,
    AIAction,
    AdditionGate,
    Always,
    ArithmeticCircuit,
    Conjunction,
    ConstantGate,
    Disjunction,
    EnforcementAction,
    Event,
    EventTrace,
    Eventually,
    Implication,
    InputGate,
    MFOTLFormula,
    MultiplicationGate,
    Negation,
    Predicate,
    ProtocolPhase,
    R1CSConstraints,
    SafetyCertificate,
    SafetyLevel,
    SafetyVerdict,
    StreamingMFOTLMonitor,
    Timestamp,
    create_aegis,
    create_aegis_for_eu_ai_act,
)

# MFOTL specifications (EU AI Act)
from aegis_omega.mfotl import (
    EUAIActSpecifications,
    MFOTLBuilder,
    TimeConstants,
)

# Zero-knowledge proofs
from aegis_omega.zkml import (
    Commitment,
    FieldElement,
    FoldedZKMLProver,
    FoldedZKMLVerifier,
    MerkleTree,
    MFOTLCircuitCompiler,
    NovaProver,
    NovaVerifier,
    R1CSInstance,
    RelaxedR1CS,
    SafetyProof,
)

# Category theory
from aegis_omega.category_theory import (
    AbstractionMorphism,
    CertificateFunctor,
    CertificateObject,
    ComposedMorphism,
    IdentityMorphism,
    Morphism,
    RefinementMorphism,
    SafetyCategory,
    SafetyObject,
    SafePipeline,
    SafetyPreservationProof,
)

__all__ = [
    "__version__",
    "__author__",
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
    "EUAIActSpecifications",
    "MFOTLBuilder",
    "TimeConstants",
    "Event",
    "EventTrace",
    "StreamingMFOTLMonitor",
    "ArithmeticCircuit",
    "ConstantGate",
    "InputGate",
    "MultiplicationGate",
    "AdditionGate",
    "R1CSConstraints",
    "FieldElement",
    "Commitment",
    "MerkleTree",
    "R1CSInstance",
    "RelaxedR1CS",
    "MFOTLCircuitCompiler",
    "NovaProver",
    "NovaVerifier",
    "FoldedZKMLProver",
    "FoldedZKMLVerifier",
    "SafetyProof",
    "Morphism",
    "SafetyObject",
    "SafetyPreservationProof",
    "IdentityMorphism",
    "ComposedMorphism",
    "RefinementMorphism",
    "AbstractionMorphism",
    "SafetyCategory",
    "CertificateObject",
    "CertificateFunctor",
    "SafePipeline",
]


def get_version() -> str:
    """Return the current AEGIS-Ω version."""
    return __version__


def check_installation() -> dict:
    """
    Verify AEGIS-Ω installation and return status.

    Returns:
        dict: Installation status for each component
    """
    status: dict = {
        "version": __version__,
        "components": {},
        "ready": True,
    }

    components = [
        ("core", "AEGISOmega"),
        ("mfotl", "EUAIActSpecifications"),
        ("zkml", "FoldedZKMLProver"),
        ("category_theory", "SafetyCategory"),
    ]

    for module_name, class_name in components:
        try:
            module = __import__(f"aegis_omega.{module_name}", fromlist=[class_name])
            getattr(module, class_name)
            status["components"][module_name] = {
                "available": True,
                "class": class_name,
            }
        except Exception as e:
            status["components"][module_name] = {
                "available": False,
                "error": str(e),
            }
            status["ready"] = False

    return status


if __name__ == "__main__":
    print(f"AEGIS-Ω v{__version__}")
    print("=" * 50)
    print("The TCP/IP of AI Safety Infrastructure")
    print()

    status = check_installation()
    print("Installation Status:")
    for component, info in status["components"].items():
        symbol = "✓" if info["available"] else "✗"
        print(f"  {symbol} {component}: {info.get('class', info.get('error'))}")

    print()
    if status["ready"]:
        print("✓ AEGIS-Ω is ready for use!")
    else:
        print("✗ Some components failed to load. Check errors above.")
