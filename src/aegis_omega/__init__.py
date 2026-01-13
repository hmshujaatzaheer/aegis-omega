"""AEGIS-Ω: Universal AI Safety Protocol."""
__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"

# Lazy imports to avoid circular dependencies
def create_aegis(mode="strict"):
    from aegis_omega.core.aegis import create_aegis as _create
    return _create(mode)

def create_aegis_for_eu_ai_act(mode="strict"):
    from aegis_omega.core.aegis import create_aegis_for_eu_ai_act as _create
    return _create(mode)

__all__ = ["create_aegis", "create_aegis_for_eu_ai_act", "__version__"]
