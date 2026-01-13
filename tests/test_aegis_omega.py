"""
AEGIS-Ω Test Suite.

Basic tests for the Universal AI Safety Protocol.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest


class TestImports:
    """Test that all modules import correctly."""

    def test_import_core(self):
        """Test core module imports."""
        from aegis_omega.core.aegis import (
            AEGISOmega,
            AIAction,
            SafetyLevel,
            EnforcementAction,
            create_aegis,
        )

        assert AEGISOmega is not None
        assert AIAction is not None
        assert SafetyLevel is not None

    def test_import_mfotl(self):
        """Test MFOTL module imports."""
        from aegis_omega.mfotl import (
            EUAIActSpecifications,
            MFOTLBuilder,
            TimeConstants,
        )

        assert EUAIActSpecifications is not None
        assert MFOTLBuilder is not None
        assert TimeConstants is not None

    def test_import_zkml(self):
        """Test ZKML module imports."""
        from aegis_omega.zkml import (
            FieldElement,
            Commitment,
            FoldedZKMLProver,
        )

        assert FieldElement is not None
        assert Commitment is not None
        assert FoldedZKMLProver is not None

    def test_import_category_theory(self):
        """Test category theory module imports."""
        from aegis_omega.category_theory import (
            SafetyCategory,
            SafetyObject,
            SafePipeline,
        )

        assert SafetyCategory is not None
        assert SafetyObject is not None
        assert SafePipeline is not None


class TestEventTrace:
    """Tests for event trace functionality."""

    def test_event_trace_creation(self):
        """Test that event traces can be created."""
        from aegis_omega.core.aegis import EventTrace

        trace = EventTrace(window_size=10)
        assert trace is not None
        assert len(trace.events) == 0

    def test_event_addition(self):
        """Test adding events to trace."""
        from aegis_omega.core.aegis import EventTrace, Event

        trace = EventTrace(window_size=10)
        trace.add_event(
            Event(timestamp=1.0, predicate="logged", arguments={"action": "test"})
        )
        assert len(trace.events) == 1

    def test_bounded_memory(self):
        """Test that memory stays bounded."""
        from aegis_omega.core.aegis import EventTrace, Event

        window_size = 50
        trace = EventTrace(window_size=window_size)

        for i in range(500):
            trace.add_event(
                Event(timestamp=float(i), predicate="test", arguments={"i": str(i)})
            )

        assert len(trace.events) <= window_size


class TestMFOTLFormulas:
    """Tests for MFOTL formula construction."""

    def test_predicate_creation(self):
        """Test predicate creation."""
        from aegis_omega.core.aegis import Predicate

        pred = Predicate("logged", ["action"])
        assert pred.name == "logged"
        assert pred.arguments == ["action"]

    def test_always_operator(self):
        """Test temporal Always operator."""
        from aegis_omega.core.aegis import Predicate, Always

        pred = Predicate("safe", [])
        always = Always(pred, lower=0, upper=10)

        assert always.formula == pred
        assert always.lower == 0
        assert always.upper == 10


class TestEUAIActSpecs:
    """Tests for EU AI Act specifications."""

    def test_specifications_exist(self):
        """Test that EU AI Act specifications are available."""
        from aegis_omega.mfotl import EUAIActSpecifications

        specs = EUAIActSpecifications.all_articles()
        assert len(specs) > 0

    def test_time_constants(self):
        """Test time constants are defined."""
        from aegis_omega.mfotl import TimeConstants

        assert TimeConstants.LOGGING_INTERVAL_MS > 0


class TestFieldElement:
    """Tests for finite field arithmetic."""

    def test_field_element_creation(self):
        """Test field element creation."""
        from aegis_omega.zkml import FieldElement

        a = FieldElement(5)
        assert a.value == 5

    def test_field_addition(self):
        """Test field element addition."""
        from aegis_omega.zkml import FieldElement

        a = FieldElement(5)
        b = FieldElement(7)
        c = a + b
        assert c.value == 12


class TestCommitment:
    """Tests for Pedersen commitments."""

    def test_commitment_creation(self):
        """Test commitment generation."""
        from aegis_omega.zkml import Commitment

        c = Commitment.commit(42)
        assert c.commitment is not None
        assert c.blinding != 0


class TestSafetyCategory:
    """Tests for categorical safety framework."""

    def test_category_creation(self):
        """Test safety category creation."""
        from aegis_omega.category_theory import SafetyCategory

        category = SafetyCategory()
        assert category is not None


class TestAEGISFactory:
    """Tests for AEGIS factory functions."""

    def test_create_aegis(self):
        """Test factory function creates valid instance."""
        from aegis_omega.core.aegis import create_aegis

        aegis = create_aegis()
        assert aegis is not None

    def test_create_aegis_for_eu_ai_act(self):
        """Test EU AI Act pre-configured AEGIS instance."""
        from aegis_omega.core.aegis import create_aegis_for_eu_ai_act

        aegis = create_aegis_for_eu_ai_act()
        assert aegis is not None
        assert len(aegis.specifications) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
