"""AEGIS-Ω Test Suite."""
import pytest

class TestImports:
    def test_import_core(self):
        from aegis_omega.core.aegis import AEGISOmega, create_aegis
        assert AEGISOmega is not None

    def test_import_mfotl(self):
        from aegis_omega.mfotl import EUAIActSpecifications
        assert EUAIActSpecifications is not None

    def test_import_zkml(self):
        from aegis_omega.zkml import FieldElement
        assert FieldElement is not None

    def test_import_category(self):
        from aegis_omega.category_theory import SafetyCategory
        assert SafetyCategory is not None

class TestCore:
    def test_create_aegis(self):
        from aegis_omega.core.aegis import create_aegis
        aegis = create_aegis()
        assert aegis is not None

class TestFieldElement:
    def test_arithmetic(self):
        from aegis_omega.zkml import FieldElement
        a = FieldElement(5)
        b = FieldElement(7)
        assert (a + b).value == 12

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
