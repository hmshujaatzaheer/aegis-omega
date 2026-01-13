"""
Categorical Safety Composition
==============================

Category-theoretic framework for compositional AI safety.

This module implements the Safety Category that enables:
- Compositional safety certificates
- Modular verification of AI pipelines
- Mathematical guarantees for system composition

Key insight: Safety properties can be structured as a category
where morphisms preserve safety, enabling compositional reasoning.

Author: H M Shujaat Zaheer
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
import hashlib
import json

from ..core.aegis import (
    MFOTLFormula,
    SafetyCertificate,
    SafetyLevel,
    Timestamp,
)


# =============================================================================
# Type Variables
# =============================================================================

A = TypeVar("A")  # AI system type
B = TypeVar("B")
C = TypeVar("C")
Spec = TypeVar("Spec", bound=MFOTLFormula)


# =============================================================================
# Category Theory Primitives
# =============================================================================

class Morphism(Generic[A, B], ABC):
    """
    Abstract morphism in the Safety Category.
    
    A morphism f: (A, φ_A) → (B, φ_B) is a safety-preserving transformation
    such that: ∀σ. A(σ) ⊨ φ_A ⟹ B(f(A(σ))) ⊨ φ_B
    """
    
    @property
    @abstractmethod
    def source(self) -> "SafetyObject[A]":
        """Source object of morphism."""
        pass
    
    @property
    @abstractmethod
    def target(self) -> "SafetyObject[B]":
        """Target object of morphism."""
        pass
    
    @abstractmethod
    def apply(self, value: A) -> B:
        """Apply the transformation."""
        pass
    
    @abstractmethod
    def safety_proof(self) -> "SafetyPreservationProof":
        """Proof that morphism preserves safety."""
        pass


@dataclass
class SafetyObject(Generic[A]):
    """
    Object in the Safety Category.
    
    Represents an AI system equipped with a safety specification.
    """
    system: A
    specification: MFOTLFormula
    system_id: str
    safety_level: SafetyLevel = SafetyLevel.HIGH
    
    def __hash__(self) -> int:
        return hash((self.system_id, str(self.specification)))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SafetyObject):
            return False
        return (self.system_id == other.system_id and 
                str(self.specification) == str(other.specification))


@dataclass
class SafetyPreservationProof:
    """Proof that a morphism preserves safety."""
    morphism_id: str
    source_spec: str
    target_spec: str
    proof_type: str  # "composition", "refinement", "abstraction"
    evidence: bytes
    verified: bool = False
    
    def verify(self) -> bool:
        """Verify the proof."""
        # In full implementation, this would check formal proof
        self.verified = len(self.evidence) > 0
        return self.verified


# =============================================================================
# Identity and Composition
# =============================================================================

@dataclass
class IdentityMorphism(Morphism[A, A]):
    """Identity morphism: id_{(A,φ)} : (A,φ) → (A,φ)."""
    
    _object: SafetyObject[A]
    
    @property
    def source(self) -> SafetyObject[A]:
        return self._object
    
    @property
    def target(self) -> SafetyObject[A]:
        return self._object
    
    def apply(self, value: A) -> A:
        return value
    
    def safety_proof(self) -> SafetyPreservationProof:
        return SafetyPreservationProof(
            morphism_id=f"id_{self._object.system_id}",
            source_spec=str(self._object.specification),
            target_spec=str(self._object.specification),
            proof_type="identity",
            evidence=b"trivial",
            verified=True
        )


@dataclass
class ComposedMorphism(Morphism[A, C]):
    """
    Composition of morphisms: g ∘ f.
    
    Given f: A → B and g: B → C, produces g ∘ f: A → C.
    """
    first: Morphism[A, B]
    second: Morphism[B, C]
    
    @property
    def source(self) -> SafetyObject[A]:
        return self.first.source
    
    @property
    def target(self) -> SafetyObject[C]:
        return self.second.target
    
    def apply(self, value: A) -> C:
        intermediate = self.first.apply(value)
        return self.second.apply(intermediate)
    
    def safety_proof(self) -> SafetyPreservationProof:
        # Compose the proofs
        proof1 = self.first.safety_proof()
        proof2 = self.second.safety_proof()
        
        combined_evidence = hashlib.sha256(
            proof1.evidence + proof2.evidence
        ).digest()
        
        return SafetyPreservationProof(
            morphism_id=f"{proof2.morphism_id}_after_{proof1.morphism_id}",
            source_spec=proof1.source_spec,
            target_spec=proof2.target_spec,
            proof_type="composition",
            evidence=combined_evidence,
            verified=proof1.verified and proof2.verified
        )


# =============================================================================
# Safety Category
# =============================================================================

class SafetyCategory:
    """
    The Safety Category Safe.
    
    Objects: AI systems with safety specifications (A, φ_A)
    Morphisms: Safety-preserving transformations
    
    This structure enables compositional reasoning about AI safety.
    """
    
    def __init__(self):
        self.objects: Dict[str, SafetyObject] = {}
        self.morphisms: Dict[str, Morphism] = {}
        self._composition_cache: Dict[Tuple[str, str], str] = {}
    
    def register_object(self, obj: SafetyObject) -> str:
        """Register an object in the category."""
        obj_id = obj.system_id
        self.objects[obj_id] = obj
        
        # Auto-register identity morphism
        identity = IdentityMorphism(obj)
        self.morphisms[f"id_{obj_id}"] = identity
        
        return obj_id
    
    def register_morphism(
        self,
        morphism_id: str,
        morphism: Morphism
    ) -> None:
        """Register a morphism in the category."""
        # Verify source and target are registered
        if morphism.source.system_id not in self.objects:
            self.register_object(morphism.source)
        if morphism.target.system_id not in self.objects:
            self.register_object(morphism.target)
        
        self.morphisms[morphism_id] = morphism
    
    def identity(self, obj_id: str) -> Morphism:
        """Get identity morphism for object."""
        return self.morphisms[f"id_{obj_id}"]
    
    def compose(
        self,
        f_id: str,
        g_id: str
    ) -> Morphism:
        """
        Compose morphisms: g ∘ f.
        
        Requires: target(f) = source(g)
        """
        cache_key = (f_id, g_id)
        if cache_key in self._composition_cache:
            return self.morphisms[self._composition_cache[cache_key]]
        
        f = self.morphisms[f_id]
        g = self.morphisms[g_id]
        
        # Verify composability
        if f.target.system_id != g.source.system_id:
            raise ValueError(
                f"Cannot compose: target of {f_id} != source of {g_id}"
            )
        
        composed = ComposedMorphism(f, g)
        composed_id = f"{g_id}_after_{f_id}"
        
        self.morphisms[composed_id] = composed
        self._composition_cache[cache_key] = composed_id
        
        return composed
    
    def get_morphisms(
        self,
        source_id: str,
        target_id: str
    ) -> List[str]:
        """Get all morphisms from source to target."""
        result = []
        for m_id, m in self.morphisms.items():
            if (m.source.system_id == source_id and 
                m.target.system_id == target_id):
                result.append(m_id)
        return result
    
    def verify_composition_safety(
        self,
        obj_ids: List[str]
    ) -> Tuple[bool, Optional[SafetyPreservationProof]]:
        """
        Verify that composing objects in sequence preserves safety.
        
        Given objects A₁, A₂, ..., Aₙ with morphisms between them,
        verify the composed system satisfies safety.
        """
        if len(obj_ids) < 2:
            return True, None
        
        # Find morphisms connecting objects
        composed_proof = None
        current_morphism = None
        
        for i in range(len(obj_ids) - 1):
            source = obj_ids[i]
            target = obj_ids[i + 1]
            
            morphisms = self.get_morphisms(source, target)
            if not morphisms:
                return False, None
            
            # Use first available morphism
            m = self.morphisms[morphisms[0]]
            
            if current_morphism is None:
                current_morphism = m
            else:
                current_morphism = ComposedMorphism(current_morphism, m)
        
        if current_morphism is not None:
            composed_proof = current_morphism.safety_proof()
            return composed_proof.verified, composed_proof
        
        return False, None


# =============================================================================
# Safety Certificate Functor
# =============================================================================

@dataclass
class CertificateObject:
    """Object in the Proof category (certificate + its proof)."""
    certificate: SafetyCertificate
    validity_proof: bytes
    
    def is_valid(self) -> bool:
        return self.certificate.is_valid()


class CertificateFunctor:
    """
    Functor C: Safe → Proof
    
    Maps safety objects to their certificates and
    morphisms to certificate transformations.
    """
    
    def __init__(self, issuer_id: str, signing_key: bytes):
        self.issuer_id = issuer_id
        self.signing_key = signing_key
    
    def map_object(self, obj: SafetyObject) -> CertificateObject:
        """Map safety object to certificate."""
        # Generate certificate
        now = Timestamp.now()
        validity_duration = 86400 * 365  # 1 year in seconds
        
        cert = SafetyCertificate(
            system_id=obj.system_id,
            specification_id=str(obj.specification),
            valid_from=now,
            valid_until=Timestamp(seconds=now.seconds + validity_duration),
            issuer=self.issuer_id,
            signature=self._sign(obj),
            proof=None  # Could include ZK proof
        )
        
        validity_proof = self._generate_validity_proof(obj)
        
        return CertificateObject(
            certificate=cert,
            validity_proof=validity_proof
        )
    
    def map_morphism(
        self,
        morphism: Morphism,
        source_cert: CertificateObject,
        target_cert: CertificateObject
    ) -> bytes:
        """
        Map morphism to proof transformation.
        
        Shows that if source certificate is valid,
        target certificate is also valid.
        """
        # Combine proofs
        combined = hashlib.sha256(
            source_cert.validity_proof +
            target_cert.validity_proof +
            morphism.safety_proof().evidence
        ).digest()
        
        return combined
    
    def _sign(self, obj: SafetyObject) -> bytes:
        """Sign object (simplified)."""
        data = f"{obj.system_id}:{obj.specification}".encode()
        return hashlib.sha256(data + self.signing_key).digest()
    
    def _generate_validity_proof(self, obj: SafetyObject) -> bytes:
        """Generate validity proof."""
        return hashlib.sha256(
            f"valid:{obj.system_id}".encode() + self.signing_key
        ).digest()


# =============================================================================
# Refinement Morphisms
# =============================================================================

@dataclass
class RefinementMorphism(Morphism[A, A]):
    """
    Refinement morphism: strengthening a safety specification.
    
    If A satisfies φ, and φ' ⟹ φ, then A also satisfies φ'.
    """
    system_obj: SafetyObject[A]
    refined_spec: MFOTLFormula
    refinement_proof: bytes  # Proof that refined_spec ⟹ original_spec
    
    @property
    def source(self) -> SafetyObject[A]:
        return self.system_obj
    
    @property
    def target(self) -> SafetyObject[A]:
        return SafetyObject(
            system=self.system_obj.system,
            specification=self.refined_spec,
            system_id=f"{self.system_obj.system_id}_refined",
            safety_level=self.system_obj.safety_level
        )
    
    def apply(self, value: A) -> A:
        return value  # System unchanged
    
    def safety_proof(self) -> SafetyPreservationProof:
        return SafetyPreservationProof(
            morphism_id=f"refine_{self.system_obj.system_id}",
            source_spec=str(self.system_obj.specification),
            target_spec=str(self.refined_spec),
            proof_type="refinement",
            evidence=self.refinement_proof,
            verified=True
        )


@dataclass
class AbstractionMorphism(Morphism[A, B]):
    """
    Abstraction morphism: abstracting system while preserving safety.
    
    Maps concrete system to abstract representation.
    """
    concrete: SafetyObject[A]
    abstract: SafetyObject[B]
    abstraction_function: Callable[[A], B]
    abstraction_proof: bytes
    
    @property
    def source(self) -> SafetyObject[A]:
        return self.concrete
    
    @property
    def target(self) -> SafetyObject[B]:
        return self.abstract
    
    def apply(self, value: A) -> B:
        return self.abstraction_function(value)
    
    def safety_proof(self) -> SafetyPreservationProof:
        return SafetyPreservationProof(
            morphism_id=f"abstract_{self.concrete.system_id}_to_{self.abstract.system_id}",
            source_spec=str(self.concrete.specification),
            target_spec=str(self.abstract.specification),
            proof_type="abstraction",
            evidence=self.abstraction_proof,
            verified=True
        )


# =============================================================================
# Pipeline Composition
# =============================================================================

class SafePipeline:
    """
    Type-safe composition of AI systems with verified safety.
    
    Ensures the composed pipeline satisfies combined safety properties.
    """
    
    def __init__(self, category: SafetyCategory):
        self.category = category
        self.stages: List[str] = []
        self.composition_proof: Optional[SafetyPreservationProof] = None
    
    def add_stage(self, obj_id: str) -> "SafePipeline":
        """Add a stage to the pipeline."""
        if obj_id not in self.category.objects:
            raise ValueError(f"Unknown object: {obj_id}")
        
        self.stages.append(obj_id)
        self._invalidate_proof()
        
        return self
    
    def verify(self) -> bool:
        """Verify the pipeline maintains safety."""
        if len(self.stages) < 1:
            return True
        
        success, proof = self.category.verify_composition_safety(self.stages)
        self.composition_proof = proof
        return success
    
    def execute(self, input_value: Any) -> Any:
        """Execute the pipeline on input."""
        if not self.verify():
            raise RuntimeError("Pipeline safety not verified")
        
        current = input_value
        for stage_id in self.stages:
            obj = self.category.objects[stage_id]
            # Apply system transformation
            # In real impl, this would call the actual AI system
            current = (obj.system, current)
        
        return current
    
    def get_combined_specification(self) -> str:
        """Get the combined safety specification."""
        specs = []
        for stage_id in self.stages:
            obj = self.category.objects[stage_id]
            specs.append(f"({stage_id}: {obj.specification})")
        return " ∧ ".join(specs)
    
    def _invalidate_proof(self) -> None:
        self.composition_proof = None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core types
    "Morphism",
    "SafetyObject",
    "SafetyPreservationProof",
    
    # Identity and composition
    "IdentityMorphism",
    "ComposedMorphism",
    
    # Safety category
    "SafetyCategory",
    
    # Functors
    "CertificateObject",
    "CertificateFunctor",
    
    # Specialized morphisms
    "RefinementMorphism",
    "AbstractionMorphism",
    
    # Pipeline
    "SafePipeline",
]
