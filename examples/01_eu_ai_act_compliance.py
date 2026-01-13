#!/usr/bin/env python3
"""
Example: EU AI Act Compliance with AEGIS-Ω
==========================================

This example demonstrates how to use AEGIS-Ω for EU AI Act compliance
monitoring in a high-risk AI system (e.g., hiring/HR AI).

The EU AI Act (Regulation 2024/1689) requires high-risk AI systems to:
- Article 9: Implement risk management
- Article 12: Maintain logging and traceability
- Article 13: Provide transparency to users
- Article 14: Enable human oversight
- Article 15: Ensure accuracy and robustness
"""

import time
from datetime import datetime
from typing import Any

# AEGIS-Ω imports
from aegis_omega import (
    create_aegis_for_eu_ai_act,
    AIAction,
    SafetyLevel,
    EnforcementAction,
)
from aegis_omega.mfotl import (
    EUAIActSpecifications,
    TimeConstants,
)


def simulate_hr_ai_system():
    """Simulate a high-risk HR/hiring AI system with AEGIS-Ω monitoring."""
    
    print("=" * 70)
    print("AEGIS-Ω EU AI Act Compliance Demo")
    print("High-Risk AI System: HR/Hiring Decision Support")
    print("=" * 70)
    print()
    
    # Initialize AEGIS-Ω with EU AI Act specifications
    # enforcement_mode can be: "strict", "permissive", or "audit"
    aegis = create_aegis_for_eu_ai_act(enforcement_mode="strict")
    
    print("✓ AEGIS-Ω initialized with EU AI Act specifications")
    print(f"  Enforcement mode: strict")
    print(f"  Safety level: {SafetyLevel.HIGH_RISK}")
    print()
    
    # Display loaded specifications
    print("Loaded EU AI Act Specifications:")
    specs = EUAIActSpecifications.all_articles()
    for spec in specs[:5]:  # Show first 5
        print(f"  • {spec.name}")
    print(f"  ... and {len(specs) - 5} more specifications")
    print()
    
    # Simulate AI actions in the hiring process
    actions = [
        {
            "action_id": "hr_001",
            "action_type": "resume_screening",
            "content": {
                "candidate_id": "C-12345",
                "decision": "proceed_to_interview",
                "confidence": 0.87,
                "factors": ["experience_match", "skills_match", "education"],
            },
            "metadata": {
                "model": "hr-screening-v2",
                "timestamp": datetime.now().isoformat(),
                "user_notified": True,
                "explanation_provided": True,
            },
        },
        {
            "action_id": "hr_002",
            "action_type": "interview_analysis",
            "content": {
                "candidate_id": "C-12345",
                "sentiment_scores": {"positive": 0.72, "neutral": 0.23, "negative": 0.05},
                "recommendation": "strong_candidate",
            },
            "metadata": {
                "model": "interview-nlp-v1",
                "timestamp": datetime.now().isoformat(),
                "user_notified": True,
                "human_oversight_requested": True,
            },
        },
        {
            "action_id": "hr_003",
            "action_type": "final_recommendation",
            "content": {
                "candidate_id": "C-12345",
                "recommendation": "hire",
                "confidence": 0.91,
                "risk_assessment": "low",
            },
            "metadata": {
                "model": "hr-decision-v3",
                "timestamp": datetime.now().isoformat(),
                "user_notified": True,
                "human_approval_required": True,
                "explanation_provided": True,
            },
        },
    ]
    
    print("Processing AI Actions:")
    print("-" * 70)
    
    for action_data in actions:
        # Create AIAction object
        action = AIAction(
            action_id=action_data["action_id"],
            action_type=action_data["action_type"],
            content=action_data["content"],
            metadata=action_data["metadata"],
        )
        
        # Process through AEGIS-Ω
        verdict = aegis.process_action(action)
        
        # Display results
        print(f"\nAction: {action.action_id} ({action.action_type})")
        print(f"  Decision: {action.content.get('decision', action.content.get('recommendation'))}")
        
        if verdict.compliant:
            print(f"  Status: ✓ COMPLIANT")
            print(f"  Certificate ID: {verdict.certificate.certificate_id if verdict.certificate else 'N/A'}")
        else:
            print(f"  Status: ✗ NON-COMPLIANT")
            print(f"  Violations:")
            for violation in verdict.violations:
                print(f"    - {violation}")
            print(f"  Required Actions:")
            for required_action in verdict.required_actions:
                print(f"    → {required_action}")
        
        # Small delay to simulate real-time processing
        time.sleep(0.1)
    
    print()
    print("-" * 70)
    print("Compliance Summary:")
    print("-" * 70)
    
    # Get compliance summary
    summary = aegis.get_compliance_summary()
    print(f"  Total actions processed: {summary['total_actions']}")
    print(f"  Compliant: {summary['compliant_count']}")
    print(f"  Non-compliant: {summary['violation_count']}")
    print(f"  Compliance rate: {summary['compliance_rate']:.1%}")
    print()
    
    # Display certificate chain
    print("Safety Certificate Chain:")
    certificates = aegis.get_certificate_chain()
    for cert in certificates[-3:]:  # Last 3 certificates
        print(f"  [{cert.timestamp}] {cert.certificate_id}")
        print(f"    Covers: {cert.covered_specifications}")
    
    print()
    print("=" * 70)
    print("Demo complete. AEGIS-Ω provides continuous EU AI Act compliance.")
    print("=" * 70)


def demonstrate_specification_formulas():
    """Show the actual MFOTL formulas for EU AI Act requirements."""
    
    print("\n" + "=" * 70)
    print("EU AI Act MFOTL Specifications")
    print("=" * 70 + "\n")
    
    # Article 12: Logging
    print("Article 12 - Logging and Traceability:")
    print("-" * 40)
    logging_specs = EUAIActSpecifications.get_article_12_specifications()
    for spec in logging_specs:
        print(f"  {spec.name}:")
        print(f"    Formula: {spec.formula}")
        print(f"    Window: {spec.window}ms")
        print()
    
    # Article 14: Human Oversight
    print("Article 14 - Human Oversight:")
    print("-" * 40)
    oversight_specs = EUAIActSpecifications.get_article_14_specifications()
    for spec in oversight_specs:
        print(f"  {spec.name}:")
        print(f"    Formula: {spec.formula}")
        print()


if __name__ == "__main__":
    simulate_hr_ai_system()
    demonstrate_specification_formulas()
