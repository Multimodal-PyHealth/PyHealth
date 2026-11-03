"""get_dataloader exposes DataLoader worker options.

``num_workers``, ``pin_memory``, ``persistent_workers`` and
``prefetch_factor`` pass through upstream's seeded loader (#1280), so
reproducible shuffling is kept while batch loading can use workers.
Invalid combinations raise instead of reaching DataLoader.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/core/test_dataloader_worker_options.py -q
"""

from __future__ import annotations

import unittest


def _dataset(n=8):
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {"patient_id": f"p{i}", "x": [float(i)], "y": i % 2} for i in range(n)
    ]
    return create_sample_dataset(samples, {"x": "tensor"}, {"y": "binary"})


class TestDataloaderWorkerOptions(unittest.TestCase):
    def test_defaults_match_plain_loader(self):
        from pyhealth.datasets import get_dataloader

        loader = get_dataloader(_dataset(), batch_size=4)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.pin_memory)
        batches = list(loader)
        self.assertEqual(len(batches), 2)

    def test_options_pass_through_seeded_loader(self):
        from pyhealth.datasets import get_dataloader

        loader = get_dataloader(
            _dataset(),
            batch_size=4,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
        )
        self.assertEqual(loader.num_workers, 1)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.prefetch_factor, 2)
        batches = list(loader)
        self.assertEqual(len(batches), 2)

    def test_shuffle_still_reproducible_with_workers(self):
        from pyhealth.datasets import get_dataloader

        ds = _dataset()
        order = lambda loader: [  # noqa: E731
            pid for b in loader for pid in b["patient_id"]
        ]
        first = order(
            get_dataloader(ds, 4, shuffle=True, seed=1, num_workers=1)
        )
        second = order(
            get_dataloader(ds, 4, shuffle=True, seed=1, num_workers=1)
        )
        self.assertEqual(first, second)

    def test_two_workers_cover_every_sample_once(self):
        from pyhealth.datasets import get_dataloader

        expected = {f"p{i}" for i in range(8)}
        for seed in (1, 2):
            loader = get_dataloader(
                _dataset(), 4, shuffle=True, seed=seed, num_workers=2
            )
            first_pass = [pid for b in loader for pid in b["patient_id"]]
            # Seeded: a fresh loader replays the same shuffle.
            replay = [
                pid
                for b in get_dataloader(
                    _dataset(), 4, shuffle=True, seed=seed, num_workers=2
                )
                for pid in b["patient_id"]
            ]
            self.assertEqual(first_pass, replay)
            # No duplication or loss across workers: every sample exactly once.
            self.assertEqual(sorted(first_pass), sorted(expected))
            # The next epoch reshuffles but still covers every sample once.
            second_pass = [pid for b in loader for pid in b["patient_id"]]
            self.assertEqual(sorted(second_pass), sorted(expected))

    def test_invalid_combinations_raise(self):
        from pyhealth.datasets import get_dataloader

        ds = _dataset()
        with self.assertRaises(ValueError):
            get_dataloader(ds, 4, num_workers=-1)
        with self.assertRaises(ValueError):
            get_dataloader(ds, 4, persistent_workers=True)
        with self.assertRaises(ValueError):
            get_dataloader(ds, 4, prefetch_factor=2)
        with self.assertRaises(ValueError):
            get_dataloader(ds, 4, num_workers=1, prefetch_factor=0)
