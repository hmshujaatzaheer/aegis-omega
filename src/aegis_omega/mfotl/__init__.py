"""
EU AI Act MFOTL Specifications
==============================

Formal MFOTL encodings of EU AI Act Articles 9-15 requirements
for high-risk AI systems.

This module provides machine-verifiable specifications derived from
Regulation (EU) 2024/1689 (Artificial Intelligence Act).

Reference: Official Journal of the European Union, L 2024/1689
Effective: August 2026 for high-risk systems

Author: H M Shujaat Zaheer
License: MIT
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .aegis import (
    MFOTLFormula,
    Predicate,
    Negation,
    Conjunction,
    Disjunction,
    Eventually,
    Always,
    Implication,
)


# =============================================================================
# Time Constants (in milliseconds)
# =============================================================================

class TimeConstants:
    """Standard time bounds for EU AI Act compliance."""
    
    # Logging deadlines
    LOG_DEADLINE_MS = 1000        # 1 second for action logging
    AUDIT_DEADLINE_MS = 5000      # 5 seconds for audit trail
    
    # Transparency deadlines
    TRANSPARENCY_NOTICE_MS = 100   # 100ms for transparency notice
    DISCLOSURE_DEADLINE_MS = 30000 # 30 seconds for detailed disclosure
    
    # Human oversight deadlines  
    NOTIFICATION_DEADLINE_MS = 60000    # 1 minute for human notification
    APPROVAL_DEADLINE_MS = 300000       # 5 minutes for human approval
    OVERRIDE_DEADLINE_MS = 10000        # 10 seconds for human override
    
    # Accuracy and robustness
    ACCURACY_CHECK_INTERVAL_MS = 3600000  # 1 hour
    ROBUSTNESS_TEST_INTERVAL_MS = 86400000  # 24 hours
    
    # Risk thresholds
    HIGH_RISK_CONFIDENCE = 0.8
    CRITICAL_RISK_CONFIDENCE = 0.95


# =============================================================================
# Article 9: Risk Management System
# =============================================================================

class Article9_RiskManagement:
    """
    Article 9 - Risk Management System
    
    High-risk AI systems shall be subject to a risk management system
    that shall be established, implemented, documented and maintained.
    """
    
    @staticmethod
    def risk_assessment_documented() -> MFOTLFormula:
        """
        Risk assessment must be documented before deployment.
        
        MFOTL: □[0,∞) (deploy(S) → ∃D. risk_assessment(S, D) ∧ D < deploy_time)
        """
        return Always(
            Implication(
                Predicate("deploy", ("system",)),
                Predicate("risk_assessment_exists", ("system",))
            ),
            lower=0,
            upper=1  # Check at deployment
        )
    
    @staticmethod
    def continuous_monitoring() -> MFOTLFormula:
        """
        Risk management shall be continuous throughout lifecycle.
        
        MFOTL: □[0,3600000] (active(S) → ◇[0,3600000] risk_review(S))
        """
        return Always(
            Implication(
                Predicate("system_active", ("system",)),
                Eventually(
                    Predicate("risk_review_conducted", ("system",)),
                    lower=0,
                    upper=TimeConstants.ACCURACY_CHECK_INTERVAL_MS
                )
            ),
            lower=0,
            upper=TimeConstants.ACCURACY_CHECK_INTERVAL_MS
        )
    
    @staticmethod
    def residual_risk_acceptable() -> MFOTLFormula:
        """
        Residual risks must be within acceptable bounds.
        """
        return Always(
            Implication(
                Predicate("decision_made", ("system", "decision")),
                Negation(Predicate("unacceptable_risk", ("system", "decision")))
            ),
            lower=0,
            upper=1000
        )


# =============================================================================
# Article 12: Record-keeping (Logging)
# =============================================================================

class Article12_Logging:
    """
    Article 12 - Record-keeping
    
    High-risk AI systems shall be designed to automatically record events
    ('logs') while operating. The logging capabilities shall ensure
    traceability of functioning throughout the AI system's lifecycle.
    """
    
    @staticmethod
    def action_logging() -> MFOTLFormula:
        """
        Every AI action must be logged within 1 second.
        
        MFOTL: □[0,∞) (ai_action(S, A, C) → ◇[0,1000] log_entry(S, A, T))
        """
        return Always(
            Implication(
                Predicate("ai_action", ("system", "action", "context")),
                Eventually(
                    Predicate("log_entry", ("system", "action", "timestamp")),
                    lower=0,
                    upper=TimeConstants.LOG_DEADLINE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.LOG_DEADLINE_MS + 100
        )
    
    @staticmethod
    def audit_trail() -> MFOTLFormula:
        """
        Complete audit trail must be maintained.
        
        MFOTL: □[0,∞) (log_entry(S, A, T) → ◇[0,5000] audit_trail(S, A, C, T))
        """
        return Always(
            Implication(
                Predicate("log_entry", ("system", "action", "timestamp")),
                Eventually(
                    Predicate("audit_trail", ("system", "action", "context", "timestamp")),
                    lower=0,
                    upper=TimeConstants.AUDIT_DEADLINE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.AUDIT_DEADLINE_MS + 100
        )
    
    @staticmethod
    def log_integrity() -> MFOTLFormula:
        """
        Logs must not be tampered with.
        
        MFOTL: □[0,∞) (log_entry(S, A, T) → □[0,∞) ¬tampered(S, A, T))
        """
        return Always(
            Implication(
                Predicate("log_entry", ("system", "action", "timestamp")),
                Always(
                    Negation(Predicate("log_tampered", ("system", "action", "timestamp"))),
                    lower=0,
                    upper=86400000  # 24 hours check window
                )
            ),
            lower=0,
            upper=1000
        )
    
    @staticmethod
    def traceability() -> MFOTLFormula:
        """
        All decisions must be traceable to their inputs.
        """
        return Always(
            Implication(
                Predicate("decision", ("system", "output")),
                Predicate("traceable_inputs", ("system", "output", "inputs"))
            ),
            lower=0,
            upper=1000
        )


# =============================================================================
# Article 13: Transparency and Information
# =============================================================================

class Article13_Transparency:
    """
    Article 13 - Transparency and provision of information to deployers
    
    High-risk AI systems shall be designed in such a way as to ensure
    that their operation is sufficiently transparent to enable deployers
    to interpret the system's output and use it appropriately.
    """
    
    @staticmethod
    def user_notification() -> MFOTLFormula:
        """
        Users must be notified they are interacting with AI.
        
        MFOTL: □[0,∞) (user_query(U, Q) → ◇[0,100] transparency_notice(U))
        """
        return Always(
            Implication(
                Predicate("user_interaction_start", ("user",)),
                Eventually(
                    Predicate("ai_disclosure_provided", ("user",)),
                    lower=0,
                    upper=TimeConstants.TRANSPARENCY_NOTICE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.TRANSPARENCY_NOTICE_MS + 50
        )
    
    @staticmethod
    def output_explanation() -> MFOTLFormula:
        """
        Outputs must be accompanied by explanations when requested.
        
        MFOTL: □[0,∞) (explanation_requested(U, O) → ◇[0,30000] explanation_provided(U, O))
        """
        return Always(
            Implication(
                Predicate("explanation_requested", ("user", "output")),
                Eventually(
                    Predicate("explanation_provided", ("user", "output")),
                    lower=0,
                    upper=TimeConstants.DISCLOSURE_DEADLINE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.DISCLOSURE_DEADLINE_MS + 1000
        )
    
    @staticmethod
    def capability_disclosure() -> MFOTLFormula:
        """
        System capabilities and limitations must be disclosed.
        """
        return Always(
            Implication(
                Predicate("system_deployed", ("system",)),
                Predicate("capabilities_documented", ("system",))
            ),
            lower=0,
            upper=1000
        )
    
    @staticmethod
    def ai_generated_content_labeled() -> MFOTLFormula:
        """
        AI-generated content must be labeled as such.
        """
        return Always(
            Implication(
                Predicate("content_generated", ("system", "content")),
                Eventually(
                    Predicate("content_labeled_ai", ("content",)),
                    lower=0,
                    upper=100
                )
            ),
            lower=0,
            upper=200
        )


# =============================================================================
# Article 14: Human Oversight
# =============================================================================

class Article14_HumanOversight:
    """
    Article 14 - Human oversight
    
    High-risk AI systems shall be designed and developed in such a way
    that they can be effectively overseen by natural persons during
    their period of use.
    """
    
    @staticmethod
    def high_risk_notification() -> MFOTLFormula:
        """
        High-risk decisions must notify human overseer.
        
        MFOTL: □[0,∞) (high_risk_decision(D, R) ∧ R > 0.8 → 
                       ◇[0,60000] human_notified(H, D))
        """
        return Always(
            Implication(
                Conjunction(
                    Predicate("high_risk_decision", ("decision", "risk_score")),
                    Predicate("risk_above_threshold", ("risk_score",))  # > 0.8
                ),
                Eventually(
                    Predicate("human_notified", ("human", "decision")),
                    lower=0,
                    upper=TimeConstants.NOTIFICATION_DEADLINE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.NOTIFICATION_DEADLINE_MS + 1000
        )
    
    @staticmethod
    def human_approval_required() -> MFOTLFormula:
        """
        Critical decisions require human approval.
        
        MFOTL: □[0,∞) (critical_decision(D) → 
                       ◇[0,300000] (human_approval(H, D) ∨ human_rejection(H, D)))
        """
        return Always(
            Implication(
                Predicate("critical_decision", ("decision",)),
                Eventually(
                    Disjunction(
                        Predicate("human_approval", ("human", "decision")),
                        Predicate("human_rejection", ("human", "decision"))
                    ),
                    lower=0,
                    upper=TimeConstants.APPROVAL_DEADLINE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.APPROVAL_DEADLINE_MS + 1000
        )
    
    @staticmethod
    def override_capability() -> MFOTLFormula:
        """
        Humans must be able to override AI decisions.
        
        MFOTL: □[0,∞) (override_requested(H, D) → 
                       ◇[0,10000] override_executed(H, D))
        """
        return Always(
            Implication(
                Predicate("override_requested", ("human", "decision")),
                Eventually(
                    Predicate("override_executed", ("human", "decision")),
                    lower=0,
                    upper=TimeConstants.OVERRIDE_DEADLINE_MS
                )
            ),
            lower=0,
            upper=TimeConstants.OVERRIDE_DEADLINE_MS + 500
        )
    
    @staticmethod
    def stop_capability() -> MFOTLFormula:
        """
        Humans must be able to stop the AI system.
        """
        return Always(
            Implication(
                Predicate("stop_requested", ("human", "system")),
                Eventually(
                    Predicate("system_stopped", ("system",)),
                    lower=0,
                    upper=1000  # Within 1 second
                )
            ),
            lower=0,
            upper=2000
        )


# =============================================================================
# Article 15: Accuracy, Robustness, Cybersecurity
# =============================================================================

class Article15_Accuracy:
    """
    Article 15 - Accuracy, robustness and cybersecurity
    
    High-risk AI systems shall be designed and developed in such a way
    that they achieve an appropriate level of accuracy, robustness
    and cybersecurity.
    """
    
    @staticmethod
    def accuracy_maintenance() -> MFOTLFormula:
        """
        System must maintain declared accuracy levels.
        
        MFOTL: □[0,∞) (accuracy_check(S) → accuracy_within_bounds(S))
        """
        return Always(
            Implication(
                Predicate("accuracy_evaluation", ("system",)),
                Predicate("accuracy_within_declared_bounds", ("system",))
            ),
            lower=0,
            upper=TimeConstants.ACCURACY_CHECK_INTERVAL_MS
        )
    
    @staticmethod
    def adversarial_robustness() -> MFOTLFormula:
        """
        System must resist adversarial inputs.
        
        MFOTL: □[0,∞) (adversarial_input(I) → blocked(I))
        """
        return Always(
            Implication(
                Predicate("adversarial_input_detected", ("input",)),
                Predicate("input_blocked", ("input",))
            ),
            lower=0,
            upper=100  # Block within 100ms
        )
    
    @staticmethod
    def error_handling() -> MFOTLFormula:
        """
        Errors must be handled gracefully.
        
        MFOTL: □[0,∞) (error_detected(S, E) → 
                       ◇[0,1000] (error_logged(S, E) ∧ graceful_degradation(S)))
        """
        return Always(
            Implication(
                Predicate("error_detected", ("system", "error")),
                Eventually(
                    Conjunction(
                        Predicate("error_logged", ("system", "error")),
                        Predicate("graceful_degradation", ("system",))
                    ),
                    lower=0,
                    upper=1000
                )
            ),
            lower=0,
            upper=2000
        )
    
    @staticmethod
    def cybersecurity_monitoring() -> MFOTLFormula:
        """
        System must have cybersecurity monitoring.
        """
        return Always(
            Implication(
                Predicate("system_active", ("system",)),
                Predicate("security_monitoring_active", ("system",))
            ),
            lower=0,
            upper=1000
        )
    
    @staticmethod
    def bias_mitigation() -> MFOTLFormula:
        """
        System must actively mitigate bias.
        """
        return Always(
            Implication(
                Predicate("bias_detected", ("system", "category")),
                Eventually(
                    Predicate("bias_mitigation_applied", ("system", "category")),
                    lower=0,
                    upper=3600000  # Within 1 hour
                )
            ),
            lower=0,
            upper=3600000 + 1000
        )


# =============================================================================
# Combined EU AI Act Specification
# =============================================================================

class EUAIActSpecifications:
    """Combined EU AI Act specifications for high-risk AI systems."""
    
    @staticmethod
    def all_articles() -> Dict[str, MFOTLFormula]:
        """Return all EU AI Act specifications."""
        return {
            # Article 9 - Risk Management
            "art9_risk_assessment": Article9_RiskManagement.risk_assessment_documented(),
            "art9_continuous_monitoring": Article9_RiskManagement.continuous_monitoring(),
            "art9_residual_risk": Article9_RiskManagement.residual_risk_acceptable(),
            
            # Article 12 - Logging
            "art12_action_logging": Article12_Logging.action_logging(),
            "art12_audit_trail": Article12_Logging.audit_trail(),
            "art12_log_integrity": Article12_Logging.log_integrity(),
            "art12_traceability": Article12_Logging.traceability(),
            
            # Article 13 - Transparency
            "art13_user_notification": Article13_Transparency.user_notification(),
            "art13_output_explanation": Article13_Transparency.output_explanation(),
            "art13_capability_disclosure": Article13_Transparency.capability_disclosure(),
            "art13_ai_content_labeled": Article13_Transparency.ai_generated_content_labeled(),
            
            # Article 14 - Human Oversight
            "art14_high_risk_notification": Article14_HumanOversight.high_risk_notification(),
            "art14_human_approval": Article14_HumanOversight.human_approval_required(),
            "art14_override_capability": Article14_HumanOversight.override_capability(),
            "art14_stop_capability": Article14_HumanOversight.stop_capability(),
            
            # Article 15 - Accuracy/Robustness
            "art15_accuracy": Article15_Accuracy.accuracy_maintenance(),
            "art15_adversarial": Article15_Accuracy.adversarial_robustness(),
            "art15_error_handling": Article15_Accuracy.error_handling(),
            "art15_cybersecurity": Article15_Accuracy.cybersecurity_monitoring(),
            "art15_bias": Article15_Accuracy.bias_mitigation(),
        }
    
    @staticmethod
    def critical_only() -> Dict[str, MFOTLFormula]:
        """Return only critical specifications for minimum compliance."""
        return {
            "art12_action_logging": Article12_Logging.action_logging(),
            "art13_user_notification": Article13_Transparency.user_notification(),
            "art14_human_approval": Article14_HumanOversight.human_approval_required(),
            "art15_adversarial": Article15_Accuracy.adversarial_robustness(),
        }
    
    @staticmethod
    def to_monpoly_format(specs: Dict[str, MFOTLFormula]) -> str:
        """
        Export specifications to MonPoly-compatible format.
        
        Note: This is a simplified conversion. Full MonPoly syntax
        requires additional signature definitions.
        """
        lines = ["(* EU AI Act Specifications for MonPoly *)", ""]
        
        for spec_id, formula in specs.items():
            lines.append(f"(* {spec_id} *)")
            lines.append(f"(* {formula} *)")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# Specification Builder
# =============================================================================

class MFOTLBuilder:
    """Fluent builder for MFOTL formulas."""
    
    def __init__(self):
        self._formula: Optional[MFOTLFormula] = None
    
    def predicate(self, name: str, *args: str) -> "MFOTLBuilder":
        self._formula = Predicate(name, tuple(args))
        return self
    
    def always(self, lower: int = 0, upper: int = 1000) -> "MFOTLBuilder":
        if self._formula is None:
            raise ValueError("No formula to wrap")
        self._formula = Always(self._formula, lower, upper)
        return self
    
    def eventually(self, lower: int = 0, upper: int = 1000) -> "MFOTLBuilder":
        if self._formula is None:
            raise ValueError("No formula to wrap")
        self._formula = Eventually(self._formula, lower, upper)
        return self
    
    def implies(self, consequent: "MFOTLBuilder") -> "MFOTLBuilder":
        if self._formula is None or consequent._formula is None:
            raise ValueError("Missing formula")
        self._formula = Implication(self._formula, consequent._formula)
        return self
    
    def and_(self, other: "MFOTLBuilder") -> "MFOTLBuilder":
        if self._formula is None or other._formula is None:
            raise ValueError("Missing formula")
        self._formula = Conjunction(self._formula, other._formula)
        return self
    
    def or_(self, other: "MFOTLBuilder") -> "MFOTLBuilder":
        if self._formula is None or other._formula is None:
            raise ValueError("Missing formula")
        self._formula = Disjunction(self._formula, other._formula)
        return self
    
    def not_(self) -> "MFOTLBuilder":
        if self._formula is None:
            raise ValueError("No formula to negate")
        self._formula = Negation(self._formula)
        return self
    
    def build(self) -> MFOTLFormula:
        if self._formula is None:
            raise ValueError("No formula built")
        return self._formula


def mfotl() -> MFOTLBuilder:
    """Create a new MFOTL formula builder."""
    return MFOTLBuilder()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TimeConstants",
    "Article9_RiskManagement",
    "Article12_Logging",
    "Article13_Transparency",
    "Article14_HumanOversight",
    "Article15_Accuracy",
    "EUAIActSpecifications",
    "MFOTLBuilder",
    "mfotl",
]
