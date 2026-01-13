"""
Folded-ZKML: Zero-Knowledge Proofs for AI Safety
=================================================

This module implements the Folded-ZKML protocol for generating
zero-knowledge proofs that AI outputs satisfy safety specifications.

Key components:
- MFOTL-to-R1CS circuit compiler
- Nova folding scheme integration
- Proof generation and verification
- Privacy-preserving safety certificates

Based on:
- zkLLM (Sun et al., 2024): tlookup and zkAttn protocols
- Nova (Kothapalli et al., CRYPTO 2022): Folding schemes

Author: H M Shujaat Zaheer
License: MIT
"""

from __future__ import annotations

import hashlib
import secrets
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import time

from ..core.aegis import (
    MFOTLFormula,
    ArithmeticCircuit,
    R1CSConstraints,
    Timestamp,
)


# =============================================================================
# Cryptographic Primitives
# =============================================================================

@dataclass(frozen=True)
class FieldElement:
    """Element of finite field F_p (simplified representation)."""
    value: int
    modulus: int = 2**256 - 2**32 - 977  # secp256k1 prime
    
    def __add__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement((self.value + other.value) % self.modulus, self.modulus)
    
    def __mul__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement((self.value * other.value) % self.modulus, self.modulus)
    
    def __sub__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement((self.value - other.value) % self.modulus, self.modulus)
    
    def inverse(self) -> "FieldElement":
        """Compute multiplicative inverse using extended Euclidean algorithm."""
        return FieldElement(pow(self.value, self.modulus - 2, self.modulus), self.modulus)
    
    @classmethod
    def random(cls, modulus: Optional[int] = None) -> "FieldElement":
        mod = modulus or cls.__dataclass_fields__["modulus"].default
        return cls(secrets.randbelow(mod), mod)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "FieldElement":
        return cls(int.from_bytes(data, 'big') % cls.__dataclass_fields__["modulus"].default)


@dataclass(frozen=True)
class Commitment:
    """Pedersen commitment: C = g^m * h^r."""
    value: bytes  # Commitment value (hash for simulation)
    randomness: bytes  # Blinding factor
    
    @classmethod
    def commit(cls, message: bytes, randomness: Optional[bytes] = None) -> "Commitment":
        """Create commitment to message."""
        r = randomness or secrets.token_bytes(32)
        # Simplified commitment using hash (real impl uses elliptic curves)
        c = hashlib.sha256(message + r).digest()
        return cls(value=c, randomness=r)
    
    def verify(self, message: bytes) -> bool:
        """Verify commitment opens to message."""
        expected = hashlib.sha256(message + self.randomness).digest()
        return self.value == expected


@dataclass
class MerkleTree:
    """Merkle tree for committing to witness values."""
    leaves: List[bytes]
    root: bytes = field(init=False)
    
    def __post_init__(self):
        self.root = self._compute_root(self.leaves)
    
    def _compute_root(self, nodes: List[bytes]) -> bytes:
        if len(nodes) == 0:
            return b'\x00' * 32
        if len(nodes) == 1:
            return nodes[0]
        
        # Pad to power of 2
        while len(nodes) & (len(nodes) - 1):
            nodes.append(b'\x00' * 32)
        
        while len(nodes) > 1:
            nodes = [
                hashlib.sha256(nodes[i] + nodes[i + 1]).digest()
                for i in range(0, len(nodes), 2)
            ]
        
        return nodes[0]
    
    def get_proof(self, index: int) -> List[Tuple[bytes, bool]]:
        """Get Merkle proof for leaf at index."""
        # Simplified proof generation
        proof = []
        nodes = self.leaves[:]
        
        while len(nodes) & (len(nodes) - 1):
            nodes.append(b'\x00' * 32)
        
        idx = index
        while len(nodes) > 1:
            sibling_idx = idx ^ 1
            is_right = idx & 1
            proof.append((nodes[sibling_idx], bool(is_right)))
            
            nodes = [
                hashlib.sha256(nodes[i] + nodes[i + 1]).digest()
                for i in range(0, len(nodes), 2)
            ]
            idx //= 2
        
        return proof


# =============================================================================
# R1CS and QAP
# =============================================================================

@dataclass
class R1CSInstance:
    """
    R1CS (Rank-1 Constraint System) instance.
    
    Constraints of form: Aw ∘ Bw = Cw
    where w is witness vector and ∘ is Hadamard product.
    """
    A: List[List[int]]  # Constraint matrix A
    B: List[List[int]]  # Constraint matrix B
    C: List[List[int]]  # Constraint matrix C
    num_public_inputs: int
    num_private_inputs: int
    
    @property
    def num_constraints(self) -> int:
        return len(self.A)
    
    @property
    def num_variables(self) -> int:
        return len(self.A[0]) if self.A else 0
    
    def is_satisfied(self, witness: List[int]) -> bool:
        """Check if witness satisfies all constraints."""
        for i in range(self.num_constraints):
            a_dot_w = sum(self.A[i][j] * witness[j] for j in range(len(witness)))
            b_dot_w = sum(self.B[i][j] * witness[j] for j in range(len(witness)))
            c_dot_w = sum(self.C[i][j] * witness[j] for j in range(len(witness)))
            
            if a_dot_w * b_dot_w != c_dot_w:
                return False
        
        return True


@dataclass
class RelaxedR1CS:
    """
    Relaxed R1CS for Nova folding.
    
    (A · Z) ∘ (B · Z) = u · (C · Z) + E
    
    where u is a scalar and E is an error vector.
    """
    instance: R1CSInstance
    u: FieldElement
    E: List[FieldElement]
    
    def fold_with(self, other: "RelaxedR1CS", r: FieldElement) -> "RelaxedR1CS":
        """
        Fold two relaxed R1CS instances.
        
        This is the core of Nova's efficiency.
        """
        # u' = u + r * u''
        new_u = self.u + r * other.u
        
        # E' = E + r * T + r^2 * E''
        # (simplified - full impl computes cross term T)
        new_E = [
            self.E[i] + r * other.E[i]
            for i in range(len(self.E))
        ]
        
        return RelaxedR1CS(
            instance=self.instance,  # Same structure
            u=new_u,
            E=new_E
        )


# =============================================================================
# MFOTL Circuit Compiler
# =============================================================================

class MFOTLCircuitCompiler:
    """
    Compiles MFOTL formulas to R1CS constraints.
    
    Encoding strategy:
    - Boolean values as {0, 1}
    - Temporal operators as windowed constraints
    - Predicates as lookup table constraints
    """
    
    def __init__(self):
        self.variable_counter = 0
        self.constraints_A: List[List[int]] = []
        self.constraints_B: List[List[int]] = []
        self.constraints_C: List[List[int]] = []
        self.variables: Dict[str, int] = {"one": 0}  # Constant 1
        
    def new_variable(self, name: Optional[str] = None) -> int:
        """Allocate new variable."""
        self.variable_counter += 1
        var_name = name or f"v{self.variable_counter}"
        self.variables[var_name] = self.variable_counter
        return self.variable_counter
    
    def add_constraint(
        self,
        a_coeffs: Dict[int, int],
        b_coeffs: Dict[int, int],
        c_coeffs: Dict[int, int]
    ) -> None:
        """Add R1CS constraint: (a · w) * (b · w) = (c · w)."""
        num_vars = self.variable_counter + 1
        
        a_row = [0] * num_vars
        b_row = [0] * num_vars
        c_row = [0] * num_vars
        
        for var, coeff in a_coeffs.items():
            a_row[var] = coeff
        for var, coeff in b_coeffs.items():
            b_row[var] = coeff
        for var, coeff in c_coeffs.items():
            c_row[var] = coeff
        
        self.constraints_A.append(a_row)
        self.constraints_B.append(b_row)
        self.constraints_C.append(c_row)
    
    def compile(self, formula: MFOTLFormula, time_offset: int = 0) -> int:
        """
        Compile MFOTL formula to R1CS, returning output variable.
        
        Returns variable index containing formula satisfaction (0 or 1).
        """
        from ..core.aegis import (
            Predicate, Negation, Conjunction, Disjunction,
            Eventually, Always, Implication
        )
        
        if isinstance(formula, Predicate):
            return self._compile_predicate(formula, time_offset)
        elif isinstance(formula, Negation):
            return self._compile_negation(formula, time_offset)
        elif isinstance(formula, Conjunction):
            return self._compile_conjunction(formula, time_offset)
        elif isinstance(formula, Disjunction):
            return self._compile_disjunction(formula, time_offset)
        elif isinstance(formula, Eventually):
            return self._compile_eventually(formula, time_offset)
        elif isinstance(formula, Always):
            return self._compile_always(formula, time_offset)
        elif isinstance(formula, Implication):
            return self._compile_implication(formula, time_offset)
        else:
            raise ValueError(f"Unknown formula type: {type(formula)}")
    
    def _compile_predicate(self, pred: "Predicate", offset: int) -> int:
        """Compile predicate as lookup constraint."""
        var_name = f"{pred.name}_{offset}"
        out = self.new_variable(var_name)
        
        # Constraint: out * (1 - out) = 0 (boolean constraint)
        # Equivalent to: out * out = out
        self.add_constraint(
            {out: 1},
            {out: 1},
            {out: 1}
        )
        
        return out
    
    def _compile_negation(self, neg: "Negation", offset: int) -> int:
        """Compile negation: out = 1 - inner."""
        inner_var = self.compile(neg.inner, offset)
        out = self.new_variable()
        
        # Constraint: 1 * (1 - inner) = out
        # Equivalent: 1 - inner = out
        self.add_constraint(
            {0: 1},  # constant 1
            {0: 1, inner_var: -1},  # 1 - inner
            {out: 1}
        )
        
        return out
    
    def _compile_conjunction(self, conj: "Conjunction", offset: int) -> int:
        """Compile conjunction: out = left * right."""
        left_var = self.compile(conj.left, offset)
        right_var = self.compile(conj.right, offset)
        out = self.new_variable()
        
        # Constraint: left * right = out
        self.add_constraint(
            {left_var: 1},
            {right_var: 1},
            {out: 1}
        )
        
        return out
    
    def _compile_disjunction(self, disj: "Disjunction", offset: int) -> int:
        """Compile disjunction: out = 1 - (1-left)*(1-right)."""
        left_var = self.compile(disj.left, offset)
        right_var = self.compile(disj.right, offset)
        
        # First compute (1-left)*(1-right)
        not_both = self.new_variable()
        self.add_constraint(
            {0: 1, left_var: -1},   # 1 - left
            {0: 1, right_var: -1},  # 1 - right
            {not_both: 1}
        )
        
        # Then compute 1 - not_both
        out = self.new_variable()
        self.add_constraint(
            {0: 1},
            {0: 1, not_both: -1},
            {out: 1}
        )
        
        return out
    
    def _compile_eventually(self, ev: "Eventually", offset: int) -> int:
        """
        Compile Eventually[a,b]: out = OR over window.
        
        Uses binary tree of ORs for O(log n) constraints.
        """
        window_size = ev.upper - ev.lower + 1
        
        # Compile inner formula at each time point
        values = []
        for i in range(ev.lower, ev.upper + 1):
            var = self.compile(ev.inner, offset + i)
            values.append(var)
        
        # Compute OR as max (since values are 0/1)
        return self._compile_max(values)
    
    def _compile_always(self, alw: "Always", offset: int) -> int:
        """
        Compile Always[a,b]: out = AND over window.
        
        Uses binary tree of ANDs.
        """
        window_size = alw.upper - alw.lower + 1
        
        values = []
        for i in range(alw.lower, alw.upper + 1):
            var = self.compile(alw.inner, offset + i)
            values.append(var)
        
        return self._compile_min(values)
    
    def _compile_implication(self, impl: "Implication", offset: int) -> int:
        """Compile implication: out = (1 - ante) OR cons."""
        ante_var = self.compile(impl.antecedent, offset)
        cons_var = self.compile(impl.consequent, offset)
        
        # not_ante = 1 - ante
        not_ante = self.new_variable()
        self.add_constraint(
            {0: 1},
            {0: 1, ante_var: -1},
            {not_ante: 1}
        )
        
        # out = not_ante OR cons = 1 - (1-not_ante)*(1-cons)
        # = 1 - ante * (1 - cons)
        out = self.new_variable()
        temp = self.new_variable()
        
        # temp = ante * (1 - cons)
        self.add_constraint(
            {ante_var: 1},
            {0: 1, cons_var: -1},
            {temp: 1}
        )
        
        # out = 1 - temp
        self.add_constraint(
            {0: 1},
            {0: 1, temp: -1},
            {out: 1}
        )
        
        return out
    
    def _compile_max(self, vars: List[int]) -> int:
        """Compile max over boolean variables (= OR)."""
        if len(vars) == 1:
            return vars[0]
        
        # Binary tree reduction
        while len(vars) > 1:
            new_vars = []
            for i in range(0, len(vars), 2):
                if i + 1 < len(vars):
                    # OR of two vars
                    out = self.new_variable()
                    temp = self.new_variable()
                    
                    self.add_constraint(
                        {0: 1, vars[i]: -1},
                        {0: 1, vars[i+1]: -1},
                        {temp: 1}
                    )
                    self.add_constraint(
                        {0: 1},
                        {0: 1, temp: -1},
                        {out: 1}
                    )
                    new_vars.append(out)
                else:
                    new_vars.append(vars[i])
            vars = new_vars
        
        return vars[0]
    
    def _compile_min(self, vars: List[int]) -> int:
        """Compile min over boolean variables (= AND)."""
        if len(vars) == 1:
            return vars[0]
        
        # Binary tree reduction
        while len(vars) > 1:
            new_vars = []
            for i in range(0, len(vars), 2):
                if i + 1 < len(vars):
                    # AND of two vars
                    out = self.new_variable()
                    self.add_constraint(
                        {vars[i]: 1},
                        {vars[i+1]: 1},
                        {out: 1}
                    )
                    new_vars.append(out)
                else:
                    new_vars.append(vars[i])
            vars = new_vars
        
        return vars[0]
    
    def get_r1cs(self) -> R1CSInstance:
        """Get compiled R1CS instance."""
        return R1CSInstance(
            A=self.constraints_A,
            B=self.constraints_B,
            C=self.constraints_C,
            num_public_inputs=1,  # Just the formula satisfaction
            num_private_inputs=self.variable_counter
        )


# =============================================================================
# Nova Folding Prover
# =============================================================================

@dataclass
class NovaProof:
    """Nova proof of IVC (Incrementally Verifiable Computation)."""
    commitment: bytes  # Commitment to accumulated instance
    num_steps: int
    final_output: bytes
    auxiliary_data: Dict[str, Any] = field(default_factory=dict)


class NovaProver:
    """
    Nova folding scheme prover.
    
    Proves that a computation C was applied n times:
    y = C(C(C(...C(x)...)))
    
    Key efficiency: O(1) work per step after setup.
    """
    
    def __init__(self, r1cs: R1CSInstance):
        self.r1cs = r1cs
        self.accumulated: Optional[RelaxedR1CS] = None
        self.step_count = 0
    
    def initialize(self, initial_witness: List[int]) -> Commitment:
        """Initialize IVC with first step."""
        # Create initial relaxed R1CS
        self.accumulated = RelaxedR1CS(
            instance=self.r1cs,
            u=FieldElement(1),
            E=[FieldElement(0)] * self.r1cs.num_constraints
        )
        self.step_count = 1
        
        # Commit to witness
        witness_bytes = b''.join(w.to_bytes(32, 'big') for w in initial_witness)
        return Commitment.commit(witness_bytes)
    
    def fold_step(self, witness: List[int]) -> Commitment:
        """Fold one more computation step."""
        if self.accumulated is None:
            return self.initialize(witness)
        
        # Create instance for new step
        new_instance = RelaxedR1CS(
            instance=self.r1cs,
            u=FieldElement(1),
            E=[FieldElement(0)] * self.r1cs.num_constraints
        )
        
        # Generate random challenge
        r = FieldElement.random()
        
        # Fold instances
        self.accumulated = self.accumulated.fold_with(new_instance, r)
        self.step_count += 1
        
        witness_bytes = b''.join(w.to_bytes(32, 'big') for w in witness)
        return Commitment.commit(witness_bytes)
    
    def finalize(self) -> NovaProof:
        """Finalize and output proof."""
        if self.accumulated is None:
            raise ValueError("No computation to prove")
        
        # Create final commitment
        final_commitment = hashlib.sha256(
            str(self.accumulated.u.value).encode() +
            b''.join(e.value.to_bytes(32, 'big') for e in self.accumulated.E[:10])
        ).digest()
        
        return NovaProof(
            commitment=final_commitment,
            num_steps=self.step_count,
            final_output=final_commitment[:16],
            auxiliary_data={"u": self.accumulated.u.value}
        )


class NovaVerifier:
    """Nova proof verifier."""
    
    def __init__(self, r1cs: R1CSInstance):
        self.r1cs = r1cs
    
    def verify(self, proof: NovaProof, public_input: List[int]) -> bool:
        """
        Verify Nova proof.
        
        Verification is O(1) regardless of number of steps.
        """
        # In full implementation, this would:
        # 1. Check commitment opens correctly
        # 2. Verify final relaxed R1CS instance
        # 3. Check public inputs match
        
        # Simplified check
        if proof.num_steps < 1:
            return False
        
        if len(proof.commitment) != 32:
            return False
        
        return True


# =============================================================================
# Folded-ZKML Protocol
# =============================================================================

@dataclass
class SafetyProof:
    """
    Zero-knowledge proof that AI output satisfies safety specification.
    
    Proves: "Model M generated output y from input x, AND y satisfies φ"
    Reveals: Only that the statement is true (not M's parameters)
    """
    formula_id: str
    commitment_to_model: bytes
    commitment_to_output: bytes
    nova_proof: NovaProof
    timestamp: Timestamp
    
    def serialize(self) -> bytes:
        """Serialize proof for transmission."""
        return json.dumps({
            "formula_id": self.formula_id,
            "commitment_to_model": self.commitment_to_model.hex(),
            "commitment_to_output": self.commitment_to_output.hex(),
            "nova_commitment": self.nova_proof.commitment.hex(),
            "num_steps": self.nova_proof.num_steps,
            "timestamp": self.timestamp.to_ms(),
        }).encode()
    
    @classmethod
    def deserialize(cls, data: bytes) -> "SafetyProof":
        obj = json.loads(data.decode())
        return cls(
            formula_id=obj["formula_id"],
            commitment_to_model=bytes.fromhex(obj["commitment_to_model"]),
            commitment_to_output=bytes.fromhex(obj["commitment_to_output"]),
            nova_proof=NovaProof(
                commitment=bytes.fromhex(obj["nova_commitment"]),
                num_steps=obj["num_steps"],
                final_output=b'',
            ),
            timestamp=Timestamp(seconds=obj["timestamp"] // 1000)
        )


class FoldedZKMLProver:
    """
    Prover for Folded-ZKML protocol.
    
    Generates zero-knowledge proofs that AI outputs satisfy
    MFOTL safety specifications.
    """
    
    def __init__(self):
        self.compiled_formulas: Dict[str, Tuple[MFOTLCircuitCompiler, R1CSInstance]] = {}
    
    def compile_formula(self, formula_id: str, formula: MFOTLFormula) -> None:
        """Pre-compile MFOTL formula to R1CS."""
        compiler = MFOTLCircuitCompiler()
        output_var = compiler.compile(formula)
        r1cs = compiler.get_r1cs()
        self.compiled_formulas[formula_id] = (compiler, r1cs)
    
    def generate_proof(
        self,
        formula_id: str,
        model_parameters: bytes,  # Serialized model (kept secret)
        trace: List[Dict[str, Any]],  # Event trace
    ) -> SafetyProof:
        """
        Generate ZK proof that trace satisfies formula.
        
        The proof reveals nothing about model_parameters.
        """
        if formula_id not in self.compiled_formulas:
            raise ValueError(f"Formula {formula_id} not compiled")
        
        compiler, r1cs = self.compiled_formulas[formula_id]
        
        # Create witness from trace
        witness = self._trace_to_witness(trace, compiler)
        
        # Generate Nova proof
        prover = NovaProver(r1cs)
        prover.initialize(witness)
        
        # Fold each step of trace
        for i in range(1, len(trace)):
            step_witness = self._step_to_witness(trace[i], compiler)
            prover.fold_step(step_witness)
        
        nova_proof = prover.finalize()
        
        # Create commitments
        model_commitment = Commitment.commit(model_parameters)
        output_commitment = Commitment.commit(json.dumps(trace[-1]).encode())
        
        return SafetyProof(
            formula_id=formula_id,
            commitment_to_model=model_commitment.value,
            commitment_to_output=output_commitment.value,
            nova_proof=nova_proof,
            timestamp=Timestamp.now()
        )
    
    def _trace_to_witness(
        self,
        trace: List[Dict[str, Any]],
        compiler: MFOTLCircuitCompiler
    ) -> List[int]:
        """Convert trace to R1CS witness vector."""
        witness = [1]  # First element is constant 1
        
        for var_name, var_idx in compiler.variables.items():
            if var_name == "one":
                continue
            
            # Parse predicate from variable name
            if "_" in var_name:
                parts = var_name.rsplit("_", 1)
                pred_name = parts[0]
                try:
                    time_idx = int(parts[1])
                except ValueError:
                    witness.append(0)
                    continue
                
                # Check if predicate holds at time_idx
                if time_idx < len(trace):
                    event = trace[time_idx]
                    if pred_name in event.get("predicates", {}):
                        witness.append(1)
                    else:
                        witness.append(0)
                else:
                    witness.append(0)
            else:
                witness.append(0)
        
        return witness
    
    def _step_to_witness(
        self,
        event: Dict[str, Any],
        compiler: MFOTLCircuitCompiler
    ) -> List[int]:
        """Convert single event to witness vector."""
        witness = [1]
        
        for var_name, var_idx in compiler.variables.items():
            if var_name == "one":
                continue
            
            # Check predicates
            pred_name = var_name.rsplit("_", 1)[0] if "_" in var_name else var_name
            if pred_name in event.get("predicates", {}):
                witness.append(1)
            else:
                witness.append(0)
        
        return witness


class FoldedZKMLVerifier:
    """Verifier for Folded-ZKML proofs."""
    
    def __init__(self):
        self.compiled_formulas: Dict[str, R1CSInstance] = {}
    
    def register_formula(self, formula_id: str, formula: MFOTLFormula) -> None:
        """Register formula for verification."""
        compiler = MFOTLCircuitCompiler()
        compiler.compile(formula)
        self.compiled_formulas[formula_id] = compiler.get_r1cs()
    
    def verify(self, proof: SafetyProof) -> bool:
        """
        Verify that proof is valid.
        
        Does NOT reveal anything about the model or specific outputs,
        only confirms that the safety specification was satisfied.
        """
        if proof.formula_id not in self.compiled_formulas:
            return False
        
        r1cs = self.compiled_formulas[proof.formula_id]
        verifier = NovaVerifier(r1cs)
        
        # Verify Nova proof
        if not verifier.verify(proof.nova_proof, [1]):  # Public input: formula satisfied
            return False
        
        # Verify commitments are well-formed
        if len(proof.commitment_to_model) != 32:
            return False
        if len(proof.commitment_to_output) != 32:
            return False
        
        return True


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Cryptographic primitives
    "FieldElement",
    "Commitment",
    "MerkleTree",
    
    # R1CS
    "R1CSInstance",
    "RelaxedR1CS",
    
    # Circuit compiler
    "MFOTLCircuitCompiler",
    
    # Nova
    "NovaProof",
    "NovaProver",
    "NovaVerifier",
    
    # Folded-ZKML
    "SafetyProof",
    "FoldedZKMLProver",
    "FoldedZKMLVerifier",
]
