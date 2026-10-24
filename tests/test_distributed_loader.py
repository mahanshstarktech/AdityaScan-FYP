"""
Unit tests for ChunkedSolarDataset memory bounds.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

def test_chunked_loader_streaming():
    torch = pytest.importorskip("torch")
    from pipeline.ml.distributed_loader import ChunkedSolarDataset, create_distributed_loader

    chunk_files = ["chunk_2024_05.pt", "chunk_2024_08.pt"]
    loader = create_distributed_loader(chunk_files, batch_size=16, num_workers=0)
    batches = 0
    for x, y in loader:
        batches += 1
        assert x.shape == (16, 2, 120)
        assert y.shape == (16, 5)
        if batches >= 5:
            break
    assert batches == 5
