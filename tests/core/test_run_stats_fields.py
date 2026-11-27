"""Run-stats fields recorded by the unified cache and the run config."""

import unittest
from types import SimpleNamespace

import torch

from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
from pyhealth.utils import _git_revision


class FrozenTextCacheBytesTest(unittest.TestCase):
    def test_bytes_is_the_storage_of_cached_vectors(self):
        # The property reads only these attributes, so a stub stands in for
        # the model and the test builds no encoder.
        cache = SimpleNamespace(
            _frozen_text_cache={
                "notes": {
                    (0, 0): torch.zeros(4),  # float32: 4 * 4 bytes
                    (0, 1): torch.zeros(2, dtype=torch.float16),  # 2 * 2 bytes
                }
            },
            _frozen_text_cache_hits=3,
            _frozen_text_cache_misses=2,
            frozen_text_cache_size=2,
        )
        stats = UnifiedMultimodalEmbeddingModel.frozen_text_cache_stats.fget(cache)
        self.assertEqual(stats["bytes"], 20)
        self.assertEqual(stats["size"], 2)

    def test_empty_cache_has_zero_bytes(self):
        cache = SimpleNamespace(
            _frozen_text_cache={},
            _frozen_text_cache_hits=0,
            _frozen_text_cache_misses=0,
            frozen_text_cache_size=0,
        )
        stats = UnifiedMultimodalEmbeddingModel.frozen_text_cache_stats.fget(cache)
        self.assertEqual(stats["bytes"], 0)


class GitRevisionFieldsTest(unittest.TestCase):
    def test_commit_and_dirty_are_recorded_as_keys(self):
        revision = _git_revision()
        self.assertEqual(set(revision), {"commit", "dirty"})
        self.assertTrue(revision["commit"] is None or len(revision["commit"]) == 40)


if __name__ == "__main__":
    unittest.main()
