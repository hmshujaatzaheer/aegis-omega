"""
Pytest configuration for AEGIS-Ω tests and benchmarks.
"""

import pytest
from datetime import datetime


@pytest.fixture
def sample_timestamp():
    """Provide a consistent timestamp for tests."""
    return datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_trace():
    """Create a sample event trace for testing."""
    from aegis_omega.core.aegis import EventTrace, Event
    
    trace = EventTrace(window_size=100)
    
    events = [
        ("action_logged", {"action": "generate"}),
        ("user_notified", {"user": "alice"}),
        ("risk_assessed", {"level": "low"}),
        ("human_approved", {"approver": "bob"}),
    ]
    
    for i, (predicate, args) in enumerate(events):
        trace.add_event(Event(
            timestamp=float(i),
            predicate=predicate,
            arguments=args
        ))
    
    return trace


@pytest.fixture
def sample_aegis():
    """Create a sample AEGIS instance for testing."""
    from aegis_omega import create_aegis_for_eu_ai_act
    return create_aegis_for_eu_ai_act()


@pytest.fixture
def sample_specification():
    """Create a sample MFOTL specification."""
    from aegis_omega.mfotl import MFOTLBuilder
    
    return (
        MFOTLBuilder()
        .always("action_logged", lower=0, upper=10)
        .and_then()
        .eventually("user_notified", lower=0, upper=60)
        .build()
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
