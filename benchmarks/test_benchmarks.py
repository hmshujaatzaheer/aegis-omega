"""
AEGIS-Ω Benchmarks
==================

Performance benchmarks for the Universal AI Safety Protocol.

Run with: pytest benchmarks/ --benchmark-json=results.json
"""

import pytest
import time
from typing import List


class TestStreamingBenchmarks:
    """Benchmarks for Streaming-MFOTL performance."""
    
    def test_throughput_10k_events(self, benchmark):
        """Benchmark: Process 10,000 events."""
        from aegis_omega.core.aegis import EventTrace, Event
        
        def process_events():
            trace = EventTrace(window_size=1000)
            for i in range(10000):
                trace.add_event(Event(
                    timestamp=float(i) / 1000,
                    predicate="action",
                    arguments={"id": str(i)}
                ))
            return len(trace.events)
        
        result = benchmark(process_events)
        assert result <= 1000  # Window bounded
    
    def test_memory_bounded(self, benchmark):
        """Benchmark: Memory stays bounded with 100k events."""
        from aegis_omega.core.aegis import EventTrace, Event
        
        def process_large_trace():
            trace = EventTrace(window_size=100)
            for i in range(100000):
                trace.add_event(Event(
                    timestamp=float(i),
                    predicate="test",
                    arguments={"data": "x" * 100}
                ))
            return len(trace.events)
        
        result = benchmark(process_large_trace)
        assert result <= 100


class TestZKMLBenchmarks:
    """Benchmarks for Folded-ZKML proof generation."""
    
    def test_commitment_generation(self, benchmark):
        """Benchmark: Pedersen commitment generation."""
        from aegis_omega.zkml import Commitment
        
        def generate_commitments():
            return [Commitment.commit(i) for i in range(100)]
        
        result = benchmark(generate_commitments)
        assert len(result) == 100
    
    def test_merkle_tree_construction(self, benchmark):
        """Benchmark: Merkle tree construction for 1024 leaves."""
        from aegis_omega.zkml import MerkleTree, Commitment
        
        def build_tree():
            leaves = [Commitment.commit(i) for i in range(1024)]
            return MerkleTree(leaves)
        
        tree = benchmark(build_tree)
        assert tree.root is not None


class TestCategoryBenchmarks:
    """Benchmarks for categorical safety operations."""
    
    def test_pipeline_composition(self, benchmark):
        """Benchmark: Compose 10-stage pipeline."""
        from aegis_omega.category_theory import SafePipeline, SafetyObject
        from aegis_omega.mfotl import MFOTLBuilder
        
        def compose_pipeline():
            pipeline = SafePipeline(name="Benchmark")
            for i in range(10):
                spec = MFOTLBuilder().always(f"stage_{i}", 0, 1).build()
                pipeline.add_stage(SafetyObject(f"Stage_{i}", spec))
            return pipeline.verify()
        
        result = benchmark(compose_pipeline)
        assert result == True


# Standalone benchmark functions for detailed analysis

def benchmark_streaming_throughput():
    """Detailed streaming throughput benchmark."""
    from aegis_omega.core.aegis import EventTrace, Event
    
    results = []
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        trace = EventTrace(window_size=1000)
        
        start = time.perf_counter()
        for i in range(size):
            trace.add_event(Event(
                timestamp=float(i) / 1000,
                predicate="action",
                arguments={"id": str(i)}
            ))
        elapsed = time.perf_counter() - start
        
        throughput = size / elapsed
        results.append({
            "events": size,
            "time_sec": elapsed,
            "throughput_eps": throughput
        })
        print(f"  {size:>10} events: {throughput:>12,.0f} events/sec")
    
    return results


def benchmark_memory_usage():
    """Memory usage benchmark."""
    import sys
    from aegis_omega.core.aegis import EventTrace, Event
    
    results = []
    window_sizes = [100, 1000, 10000]
    
    for window_size in window_sizes:
        trace = EventTrace(window_size=window_size)
        
        # Add many events
        for i in range(1000000):
            trace.add_event(Event(
                timestamp=float(i),
                predicate="test",
                arguments={"data": "x" * 100}
            ))
        
        # Estimate memory
        event_count = len(trace.events)
        estimated_memory = event_count * 200  # Rough estimate per event
        
        results.append({
            "window_size": window_size,
            "events_stored": event_count,
            "estimated_memory_kb": estimated_memory / 1024
        })
        print(f"  Window {window_size:>6}: {event_count:>6} events, ~{estimated_memory/1024:.0f} KB")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("AEGIS-Ω Performance Benchmarks")
    print("=" * 60)
    
    print("\n📊 Streaming Throughput:")
    benchmark_streaming_throughput()
    
    print("\n💾 Memory Usage:")
    benchmark_memory_usage()
    
    print("\n✅ Benchmarks complete")
