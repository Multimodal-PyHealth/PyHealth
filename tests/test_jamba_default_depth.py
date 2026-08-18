"""Proof that JambaEHR defaults to 2 transformer + 2 mamba layers.

The published runner used library defaults of 2+6, which over-parameterised
the e2e comparison against Transformer and RNN. Both the class and the CLI
now default to 2+2.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mortality_prediction"
    / "unified_embedding_e2e_mimic4.py"
)


class TestJambaDefaultDepth(unittest.TestCase):
    def test_library_defaults_to_two_plus_two(self):
        from pyhealth.models.jamba_ehr import JambaEHR, JambaLayer

        self.assertEqual(JambaLayer.__init__.__defaults__[0], 2)
        self.assertEqual(JambaLayer.__init__.__defaults__[1], 2)
        self.assertEqual(JambaEHR.__init__.__defaults__[1], 2)
        self.assertEqual(JambaEHR.__init__.__defaults__[2], 2)
        layer = JambaLayer(16)
        self.assertEqual(layer.schedule.count("transformer"), 2)
        self.assertEqual(layer.schedule.count("mamba"), 2)

    def test_cli_defaults_to_two_plus_two(self):
        src = RUNNER.read_text(encoding="utf-8")
        self.assertRegex(
            src,
            r'--jamba-transformer-layers", type=int, default=2',
        )
        self.assertRegex(
            src,
            r'--jamba-mamba-layers", type=int, default=2',
        )
