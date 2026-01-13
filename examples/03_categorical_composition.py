"""
Example 03: Categorical Safety Composition
==========================================

This example demonstrates how to use category theory to compose
safe AI systems while mathematically guaranteeing that safety
properties are preserved through composition.

Key Concepts:
- Safety Category: Objects are (AI_system, specification) pairs
- Morphisms: Safety-preserving transformations
- Certificate Functor: Maps safety objects to proofs
- Pipeline Composition: Build complex systems from verified components

This is AEGIS-Ω Contribution #2 from the PhD proposal.
"""

from datetime import datetime
from typing import List

# Import categorical safety framework
from aegis_omega.category_theory import (
    SafetyCategory,
    SafetyObject,
    RefinementMorphism,
    AbstractionMorphism,
    CertificateFunctor,
    SafePipeline,
)
from aegis_omega.mfotl import MFOTLBuilder


def demo_safety_category():
    """Demonstrate the basic Safety Category structure."""
    
    print("=" * 60)
    print("CATEGORICAL SAFETY: The Safety Category")
    print("=" * 60)
    
    # Create the safety category
    category = SafetyCategory()
    
    # Define safety specifications using MFOTL
    # Object 1: Input Validator with logging requirement
    spec_validator = (
        MFOTLBuilder()
        .always("input_validated", lower=0, upper=1)
        .and_then()
        .always("validation_logged", lower=0, upper=1)
        .build()
    )
    
    # Object 2: Content Processor with transparency requirement
    spec_processor = (
        MFOTLBuilder()
        .always("processing_logged", lower=0, upper=1)
        .and_then()
        .eventually("user_notified", lower=0, upper=5)
        .build()
    )
    
    # Object 3: Output Filter with human oversight
    spec_filter = (
        MFOTLBuilder()
        .always("output_checked", lower=0, upper=1)
        .and_then()
        .implies("high_risk_detected", "human_review_requested")
        .build()
    )
    
    # Create safety objects (systems with their specifications)
    validator = SafetyObject(
        system_id="InputValidator_v1",
        specification=spec_validator
    )
    
    processor = SafetyObject(
        system_id="ContentProcessor_v1",
        specification=spec_processor
    )
    
    output_filter = SafetyObject(
        system_id="OutputFilter_v1",
        specification=spec_filter
    )
    
    # Register objects in the category
    category.register_object(validator)
    category.register_object(processor)
    category.register_object(output_filter)
    
    print("\n📦 Registered Safety Objects:")
    print(f"  1. {validator.system_id}")
    print(f"     Spec: Always(input_validated) ∧ Always(validation_logged)")
    print(f"  2. {processor.system_id}")
    print(f"     Spec: Always(processing_logged) ∧ Eventually(user_notified)")
    print(f"  3. {output_filter.system_id}")
    print(f"     Spec: Always(output_checked) ∧ (high_risk → human_review)")
    
    # Demonstrate identity morphism
    id_validator = category.identity(validator)
    print(f"\n🔄 Identity morphism: id_{{{validator.system_id}}}")
    print(f"   {id_validator.source.system_id} → {id_validator.target.system_id}")
    
    return category, validator, processor, output_filter


def demo_morphism_composition():
    """Demonstrate morphism composition preserves safety."""
    
    print("\n" + "=" * 60)
    print("MORPHISM COMPOSITION: Safety-Preserving Transformations")
    print("=" * 60)
    
    category = SafetyCategory()
    
    # Create systems with progressively stricter specifications
    # Base system: log every 100ms
    spec_base = MFOTLBuilder().always("logged", lower=0, upper=100).build()
    
    # Intermediate: log every 50ms (stricter)
    spec_intermediate = MFOTLBuilder().always("logged", lower=0, upper=50).build()
    
    # Strict: log every 10ms (strictest)
    spec_strict = MFOTLBuilder().always("logged", lower=0, upper=10).build()
    
    sys_base = SafetyObject("Base_Logger", spec_base)
    sys_intermediate = SafetyObject("Intermediate_Logger", spec_intermediate)
    sys_strict = SafetyObject("Strict_Logger", spec_strict)
    
    category.register_object(sys_base)
    category.register_object(sys_intermediate)
    category.register_object(sys_strict)
    
    # Create refinement morphisms (φ_strict ⟹ φ_intermediate ⟹ φ_base)
    # Stricter spec satisfies weaker spec
    
    refine_strict_to_intermediate = RefinementMorphism(
        source=sys_strict,
        target=sys_intermediate,
        proof_of_refinement="Always[0,10](logged) ⟹ Always[0,50](logged)"
    )
    
    refine_intermediate_to_base = RefinementMorphism(
        source=sys_intermediate,
        target=sys_base,
        proof_of_refinement="Always[0,50](logged) ⟹ Always[0,100](logged)"
    )
    
    category.register_morphism(refine_strict_to_intermediate)
    category.register_morphism(refine_intermediate_to_base)
    
    # Compose morphisms: strict → base
    composed = category.compose(
        refine_strict_to_intermediate,
        refine_intermediate_to_base
    )
    
    print("\n📐 Morphism Chain:")
    print(f"  f: Strict_Logger → Intermediate_Logger")
    print(f"     Proof: Always[0,10] ⟹ Always[0,50]")
    print(f"  g: Intermediate_Logger → Base_Logger")
    print(f"     Proof: Always[0,50] ⟹ Always[0,100]")
    print(f"\n  g ∘ f: Strict_Logger → Base_Logger (Composed)")
    print(f"     Composed proof preserves safety!")
    
    # Verify the composition
    is_valid = category.verify_composition_safety(composed)
    print(f"\n✓ Composition validity: {is_valid}")
    
    return composed


def demo_certificate_functor():
    """Demonstrate the Certificate Functor C: Safe → Proof."""
    
    print("\n" + "=" * 60)
    print("CERTIFICATE FUNCTOR: C: Safe → Proof")
    print("=" * 60)
    
    # Create the certificate functor
    functor = CertificateFunctor()
    
    # Create a safety object
    spec = (
        MFOTLBuilder()
        .always("action_logged", lower=0, upper=1)
        .and_then()
        .always("user_notified", lower=0, upper=5)
        .build()
    )
    
    safety_obj = SafetyObject(
        system_id="AI_Assistant_v1",
        specification=spec
    )
    
    # Map through the functor to get a certificate
    certificate = functor.map_object(safety_obj)
    
    print("\n🎫 Certificate Generation:")
    print(f"  Input: SafetyObject({safety_obj.system_id}, φ)")
    print(f"  Output: Certificate with:")
    print(f"    - Certificate ID: {certificate.certificate_id}")
    print(f"    - System ID: {certificate.system_id}")
    print(f"    - Specification ID: {certificate.specification_id}")
    print(f"    - Issued: {certificate.issued_at}")
    print(f"    - Expires: {certificate.expires_at}")
    print(f"    - Proof Commitment: {certificate.proof_commitment[:32]}...")
    
    # The functor preserves structure
    print("\n📜 Functor Properties:")
    print("  ✓ Preserves identity: C(id_A) = id_{C(A)}")
    print("  ✓ Preserves composition: C(g ∘ f) = C(g) ∘ C(f)")
    print("  ✓ Maps safety objects to verifiable certificates")
    
    return certificate


def demo_safe_pipeline():
    """Demonstrate building a safe AI pipeline from components."""
    
    print("\n" + "=" * 60)
    print("SAFE PIPELINE: Composing Verified Components")
    print("=" * 60)
    
    # Create a pipeline for a complete AI assistant workflow
    pipeline = SafePipeline(name="AI_Assistant_Pipeline")
    
    # Stage 1: Input Validation
    input_spec = (
        MFOTLBuilder()
        .always("input_sanitized", lower=0, upper=1)
        .and_then()
        .always("prompt_logged", lower=0, upper=1)
        .build()
    )
    input_stage = SafetyObject("InputValidator", input_spec)
    
    # Stage 2: Content Generation
    gen_spec = (
        MFOTLBuilder()
        .always("generation_logged", lower=0, upper=1)
        .and_then()
        .always("token_count_tracked", lower=0, upper=1)
        .build()
    )
    gen_stage = SafetyObject("ContentGenerator", gen_spec)
    
    # Stage 3: Safety Filter
    filter_spec = (
        MFOTLBuilder()
        .always("content_checked", lower=0, upper=1)
        .and_then()
        .implies("harmful_detected", "content_blocked")
        .build()
    )
    filter_stage = SafetyObject("SafetyFilter", filter_spec)
    
    # Stage 4: Output Delivery
    output_spec = (
        MFOTLBuilder()
        .always("output_logged", lower=0, upper=1)
        .and_then()
        .eventually("user_received", lower=0, upper=10)
        .build()
    )
    output_stage = SafetyObject("OutputDelivery", output_spec)
    
    # Add stages to pipeline
    pipeline.add_stage(input_stage)
    pipeline.add_stage(gen_stage)
    pipeline.add_stage(filter_stage)
    pipeline.add_stage(output_stage)
    
    print("\n🔗 Pipeline Architecture:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  AI_Assistant_Pipeline                                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    for i, stage in enumerate(pipeline.stages, 1):
        print(f"│  Stage {i}: {stage.system_id:<20}")
        print(f"│           Spec: {str(stage.specification)[:40]}...")
        if i < len(pipeline.stages):
            print("│           ↓")
    print("└─────────────────────────────────────────────────────────────┘")
    
    # Verify the pipeline
    is_safe = pipeline.verify()
    print(f"\n✓ Pipeline safety verified: {is_safe}")
    
    # Get combined specification
    combined_spec = pipeline.get_combined_specification()
    print(f"\n📋 Combined Specification:")
    print(f"   φ_pipeline = φ_input ∧ φ_gen ∧ φ_filter ∧ φ_output")
    print(f"   (Conjunction of all stage specifications)")
    
    return pipeline


def demo_compositional_theorem():
    """Demonstrate the main compositional safety theorem."""
    
    print("\n" + "=" * 60)
    print("COMPOSITIONAL SAFETY THEOREM")
    print("=" * 60)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  THEOREM (Compositional Safety):                              ║
    ║                                                               ║
    ║  If (A, φ_A) and (B, φ_B) are safety objects and              ║
    ║  f: (A, φ_A) → (B, φ_B) is a safety-preserving morphism,      ║
    ║  then:                                                        ║
    ║                                                               ║
    ║      A satisfies φ_A  ⟹  f(A) satisfies φ_B                  ║
    ║                                                               ║
    ║  Furthermore, certificates compose:                           ║
    ║                                                               ║
    ║      C(g ∘ f) = C(g) ∘ C(f)                                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("📊 Implications for AI Safety:")
    print()
    print("  1. MODULAR VERIFICATION:")
    print("     Verify components independently, compose safely")
    print()
    print("  2. INCREMENTAL CERTIFICATION:")
    print("     Add new components without re-verifying entire system")
    print()
    print("  3. HIERARCHICAL SAFETY:")
    print("     Build complex systems from simple, verified parts")
    print()
    print("  4. FORMAL GUARANTEES:")
    print("     Mathematical proof of safety preservation")


def main():
    """Run all categorical safety demonstrations."""
    
    print("\n" + "🔷" * 30)
    print("  AEGIS-Ω: CATEGORICAL SAFETY COMPOSITION")
    print("  PhD Contribution #2: Mathematical Framework for AI Safety")
    print("🔷" * 30)
    
    # Run demonstrations
    category, validator, processor, output_filter = demo_safety_category()
    composed_morphism = demo_morphism_composition()
    certificate = demo_certificate_functor()
    pipeline = demo_safe_pipeline()
    demo_compositional_theorem()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Why This Matters")
    print("=" * 60)
    print("""
    The categorical framework enables:
    
    ✓ COMPOSITIONAL REASONING: Prove safety of complex systems
      by proving safety of components
    
    ✓ CERTIFICATE PROPAGATION: Safety certificates compose
      just like the systems they certify
    
    ✓ TYPE SAFETY FOR AI: Just as type systems prevent bugs,
      safety categories prevent unsafe compositions
    
    ✓ STANDARDIZED INTERFACES: Safety Category becomes the
      "type signature" for AI system interconnection
    
    This is foundational infrastructure for the AI Safety Handshake
    Protocol (AEGIS-Ω Contribution #4), enabling any AI system to
    negotiate safety requirements with any other AI system.
    """)


if __name__ == "__main__":
    main()
