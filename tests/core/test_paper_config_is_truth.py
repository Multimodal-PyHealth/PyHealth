"""The runner's config must describe the run, or refuse to start.

Three ways a run used to be able to disagree with its own config:

  * bottleneck_transformer got max_grad_norm=0.5 and Adam eps=1e-6 from a
    branch on args.model, while run_config recorded the unset flags;
  * --amp-dtype without --use-amp ran fp32 with amp_dtype: "bf16" written down;
  * an empty patient split fell back to a by-sample split that puts one patient
    in train and test, behind a warning nobody reads in a nohup log.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "mortality_prediction"
    / "unified_embedding_e2e_mimic4.py"
)

_spec = importlib.util.spec_from_file_location("_e2e_runner", _RUNNER)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

_BASE = ["--ehr-root", "/tmp/ehr", "--task", "labs"]


def _parse(extra):
    argv = sys.argv
    sys.argv = ["unified_embedding_e2e_mimic4.py"] + _BASE + extra
    try:
        return runner.parse_args()
    finally:
        sys.argv = argv


class TestOptimizerFlagsAreUniform(unittest.TestCase):
    def test_every_model_gets_the_same_optimizer_defaults(self):
        for model in ("mlp", "rnn", "transformer", "bottleneck_transformer",
                      "ehrmamba", "jambaehr"):
            args = _parse(["--model", model])
            self.assertEqual(args.max_grad_norm, 1.0, model)
            self.assertEqual(args.adam_eps, 1e-8, model)
            self.assertEqual(args.lr, 1e-4, model)

    def test_runner_has_no_per_model_optimizer_branch(self):
        source = _RUNNER.read_text()
        self.assertNotIn("effective_max_grad_norm = 0.5", source)
        self.assertNotIn('optimizer_params["eps"] = args.adam_eps if', source)


class TestAmpFlagsCannotLie(unittest.TestCase):
    def test_amp_dtype_without_use_amp_is_refused(self):
        with self.assertRaises(SystemExit):
            _parse(["--model", "mlp", "--amp_dtype", "bf16"])

    def test_use_amp_alone_means_bf16(self):
        args = _parse(["--model", "mlp", "--use-amp"])
        self.assertTrue(args.use_amp)
        self.assertEqual(args.amp_dtype, "bf16")

    def test_use_amp_with_dtype_is_accepted(self):
        args = _parse(["--model", "mlp", "--use-amp", "--amp-dtype", "fp16"])
        self.assertTrue(args.use_amp)
        self.assertEqual(args.amp_dtype, "fp16")


class TestInertFlagsAreReported(unittest.TestCase):
    def test_mlp_reports_dropout_as_inert(self):
        # pyhealth.models.mlp.MLP has no dropout parameter at all.
        self.assertIn("--dropout", runner._inert_arch_flags("mlp"))

    def test_jambaehr_reports_num_layers_as_inert(self):
        # Depth comes from --jamba-transformer-layers / --jamba-mamba-layers.
        self.assertIn("--num-layers", runner._inert_arch_flags("jambaehr"))

    def test_flags_a_model_consumes_are_not_listed(self):
        expected_used = {
            "bottleneck_transformer": ["--bottlenecks-n", "--fusion-startidx",
                                       "--heads", "--num-layers", "--dropout"],
            "ehrmamba": ["--mamba-state-size", "--mamba-conv-kernel",
                         "--num-layers", "--dropout"],
            "rnn": ["--rnn-type", "--rnn-layers", "--hidden-dim", "--dropout"],
        }
        for model, used in expected_used.items():
            inert = runner._inert_arch_flags(model)
            for flag in used:
                self.assertNotIn(flag, inert, f"{model} {flag}")

    def test_every_model_has_an_entry(self):
        for model in ("mlp", "rnn", "transformer", "bottleneck_transformer",
                      "ehrmamba", "jambaehr"):
            self.assertIn(model, runner._ARCH_FLAGS_USED)


class _FakeSubset(list):
    pass


class TestLeakySplitIsRefused(unittest.TestCase):
    """_split_dataset must raise rather than silently leak patients."""

    def _patch(self, by_patient, by_sample):
        self._orig = (runner.split_by_patient, runner.split_by_sample)
        runner.split_by_patient = lambda ds, ratios, seed: by_patient
        runner.split_by_sample = lambda ds, ratios, seed: by_sample
        self.addCleanup(self._restore)

    def _restore(self):
        runner.split_by_patient, runner.split_by_sample = self._orig

    def test_empty_patient_split_raises_by_default(self):
        self._patch((_FakeSubset(), _FakeSubset(), _FakeSubset()),
                    (_FakeSubset([1]), _FakeSubset([2]), _FakeSubset([3])))
        with self.assertRaises(RuntimeError) as ctx:
            runner._split_dataset(_FakeSubset([1, 2, 3]), seed=1)
        self.assertIn("--allow-leaky-split", str(ctx.exception))

    def test_opt_in_still_available_for_smoke_tests(self):
        self._patch((_FakeSubset(), _FakeSubset(), _FakeSubset()),
                    (_FakeSubset([1]), _FakeSubset([2]), _FakeSubset([3])))
        with self.assertWarns(RuntimeWarning):
            *_, mode = runner._split_dataset(
                _FakeSubset([1, 2, 3]), seed=1, allow_leaky_split=True
            )
        self.assertEqual(mode, "by_sample_fallback_leaky")

    def test_healthy_patient_split_is_used_unchanged(self):
        self._patch((_FakeSubset([1]), _FakeSubset([2]), _FakeSubset([3])),
                    (_FakeSubset(), _FakeSubset(), _FakeSubset()))
        *_, mode = runner._split_dataset(_FakeSubset([1, 2, 3]), seed=1)
        self.assertEqual(mode, "by_patient")


if __name__ == "__main__":
    unittest.main()
