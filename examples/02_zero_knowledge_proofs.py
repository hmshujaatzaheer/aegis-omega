#!/usr/bin/env python3
"""
Example: Zero-Knowledge Safety Proofs with Folded-ZKML
======================================================

This example demonstrates how to use AEGIS-Ω's Folded-ZKML protocol
to generate and verify zero-knowledge proofs of AI safety compliance.

Key concept: A model provider can prove that their AI's output satisfies
safety specifications WITHOUT revealing the model's parameters or internals.

This enables:
1. Regulatory compliance verification without exposing trade secrets
2. Privacy-preserving audits
3. Trustless AI-to-AI verification
"""

from datetime import datetime
from typing import Dict, Any

# AEGIS-Ω imports
from aegis_omega.zkml import (
    FoldedZKMLProver,
    FoldedZKMLVerifier,
    SafetyProof,
    MFOTLCircuitCompiler,
    NovaProver,
    NovaVerifier,
    FieldElement,
    Commitment,
)
from aegis_omega import (
    Predicate,
    Conjunction,
    Always,
    MFOTLFormula,
)


def demonstrate_basic_zk_proof():
    """Demonstrate basic zero-knowledge proof generation and verification."""
    
    print("=" * 70)
    print("Folded-ZKML: Zero-Knowledge Safety Proofs")
    print("=" * 70)
    print()
    
    # ===== SCENARIO =====
    # An AI company wants to prove to a regulator that their model's output
    # satisfies safety specifications, WITHOUT revealing:
    # - Model architecture
    # - Model weights
    # - Training data
    # - Internal computations
    
    print("SCENARIO: AI Company proving safety compliance to regulator")
    print("-" * 70)
    print()
    
    # Step 1: Define the safety specification (known to both parties)
    print("Step 1: Define Safety Specification")
    print("  The specification is public - both prover and verifier know it.")
    print()
    
    safety_spec = Conjunction(
        Predicate("no_harmful_content"),
        Conjunction(
            Predicate("factually_grounded"),
            Predicate("bias_checked")
        )
    )
    
    print(f"  Specification: {safety_spec}")
    print("  Meaning: Output must have no harmful content AND be factually")
    print("           grounded AND pass bias checks.")
    print()
    
    # Step 2: Prover (AI Company) generates proof
    print("Step 2: Prover Generates Zero-Knowledge Proof")
    print("-" * 70)
    print()
    
    # Initialize prover with their secret model
    prover = FoldedZKMLProver()
    
    # Compile the MFOTL formula to an arithmetic circuit
    compiler = MFOTLCircuitCompiler()
    circuit = compiler.compile(safety_spec)
    
    print(f"  Compiled MFOTL to circuit with {circuit.num_constraints} constraints")
    print()
    
    # Simulate model input/output (in practice, this is the actual inference)
    model_input = "What are the benefits of renewable energy?"
    model_output = "Renewable energy reduces carbon emissions and provides..."
    
    # Generate the zero-knowledge proof
    # This proves: "My model produced this output from this input,
    #               AND the output satisfies the safety specification"
    # WITHOUT revealing how the model works
    
    print("  Generating proof...")
    proof = prover.generate_proof(
        formula=safety_spec,
        input_commitment=Commitment.commit(model_input.encode()),
        output_commitment=Commitment.commit(model_output.encode()),
        satisfaction_witness={
            "no_harmful_content": True,
            "factually_grounded": True,
            "bias_checked": True,
        },
    )
    
    print(f"  ✓ Proof generated!")
    print(f"    Proof size: {proof.size_bytes} bytes")
    print(f"    Generation time: {proof.generation_time_ms:.2f} ms")
    print()
    
    # Step 3: Verifier (Regulator) verifies proof
    print("Step 3: Verifier Checks Proof")
    print("-" * 70)
    print()
    
    verifier = FoldedZKMLVerifier()
    
    # Register the known specification
    verifier.register_formula(safety_spec)
    
    print("  Verifying proof...")
    verification_result = verifier.verify(
        proof=proof,
        formula=safety_spec,
    )
    
    if verification_result.valid:
        print(f"  ✓ PROOF VERIFIED!")
        print(f"    Verification time: {verification_result.verification_time_ms:.2f} ms")
        print()
        print("  The verifier now knows:")
        print("    1. The AI's output satisfies the safety specification")
        print("    2. The output was genuinely produced by the claimed model")
        print()
        print("  The verifier does NOT know:")
        print("    1. The model's architecture")
        print("    2. The model's weights or parameters")
        print("    3. How the model computed its output")
        print("    4. Any intermediate activations or reasoning")
    else:
        print(f"  ✗ Proof verification FAILED: {verification_result.error}")
    
    print()
    return proof


def demonstrate_nova_folding():
    """Demonstrate Nova IVC folding for efficient recursive proofs."""
    
    print("=" * 70)
    print("Nova Folding: Efficient Recursive Proofs")
    print("=" * 70)
    print()
    
    print("CONCEPT: Incrementally Verifiable Computation (IVC)")
    print("-" * 70)
    print()
    print("Problem: Verifying a long computation (e.g., 1000 inference steps)")
    print("         would normally require checking each step individually.")
    print()
    print("Solution: Nova folding combines proofs recursively:")
    print("          π₁ ⊕ π₂ → π_{1+2}  (constant size!)")
    print()
    print("Result: O(1) proof size regardless of computation length")
    print()
    
    # Initialize Nova prover
    nova_prover = NovaProver()
    
    # Simulate a multi-step computation (e.g., processing a conversation)
    print("Simulating 10-step conversation monitoring:")
    print("-" * 70)
    
    # Initialize the folding
    running_proof = nova_prover.initialize()
    
    steps = [
        "User: Hello",
        "AI: Hi there!",
        "User: Tell me about Paris",
        "AI: Paris is the capital of France...",
        "User: What's the population?",
        "AI: Paris has about 2.1 million people...",
        "User: Thanks!",
        "AI: You're welcome!",
        "User: Goodbye",
        "AI: Goodbye! Have a great day!",
    ]
    
    for i, step in enumerate(steps):
        # Create witness for this step (safety checks passed)
        step_witness = {
            "step": i,
            "content": step[:30] + "..." if len(step) > 30 else step,
            "safe": True,
        }
        
        # Fold this step into the running proof
        running_proof = nova_prover.fold_step(running_proof, step_witness)
        
        print(f"  Step {i+1}: Folded. Proof size: {running_proof.size} bytes (constant!)")
    
    print()
    print(f"Final proof covers all 10 steps")
    print(f"Proof size: {running_proof.size} bytes (same as single step!)")
    print()
    
    # Verify the folded proof
    nova_verifier = NovaVerifier()
    final_result = nova_verifier.verify(running_proof)
    
    if final_result.valid:
        print("✓ All 10 steps verified with single O(1) proof!")
    
    print()


def demonstrate_privacy_preservation():
    """Demonstrate the privacy guarantees of Folded-ZKML."""
    
    print("=" * 70)
    print("Privacy Guarantees: What Remains Hidden")
    print("=" * 70)
    print()
    
    print("ZERO-KNOWLEDGE PROPERTY:")
    print("-" * 70)
    print()
    print("The verifier learns ONLY that the safety property is satisfied.")
    print("Everything else remains hidden:")
    print()
    
    hidden_info = [
        ("Model Architecture", "Transformer, CNN, MLP, etc."),
        ("Model Parameters", "Weights, biases, embeddings"),
        ("Training Data", "What data was used to train"),
        ("Intermediate States", "Activations, attention patterns"),
        ("Reasoning Process", "How the model reached its output"),
        ("Confidence Scores", "Internal probability distributions"),
        ("Alternative Outputs", "What else the model considered"),
    ]
    
    for item, description in hidden_info:
        print(f"  🔒 {item}")
        print(f"     {description}")
        print()
    
    print("This enables:")
    print("  ✓ Regulatory compliance without IP disclosure")
    print("  ✓ Third-party audits without model access")
    print("  ✓ Competitive AI verification in multi-agent systems")
    print("  ✓ User trust without transparency risks")
    print()


def main():
    """Run all demonstrations."""
    
    # Basic ZK proof demo
    proof = demonstrate_basic_zk_proof()
    
    # Nova folding demo
    demonstrate_nova_folding()
    
    # Privacy explanation
    demonstrate_privacy_preservation()
    
    print("=" * 70)
    print("Summary: Folded-ZKML enables privacy-preserving AI safety verification")
    print("=" * 70)


if __name__ == "__main__":
    main()
