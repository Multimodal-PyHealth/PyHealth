"""Synthetic run-directory tests for tools/collect_run_stats.py (CPU, seconds)."""

import csv
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import collect_run_stats as crs


def _rich_run(root, seed=1):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "run_config.json"), "w") as f:
        json.dump({
            "config": {
                "task": "notes_labs", "backbone": "transformer",
                "embedding_dim": 128, "hidden_dim": 128, "num_layers": 2,
                "heads": 4, "batch_size": 32, "use_amp": True,
                "amp_dtype": "bf16", "freeze_encoder": True,
                "seed": seed, "window_hours": 0,
                "thread_pinning": "OMP_NUM_THREADS=8",
                "drop_modality": [], "lr": 1e-4, "adam_eps": 1e-8,
                "max_grad_norm": 1.0, "dev": 0,
                "split_mode": "by_patient", "eval_split": "test",
                "cache_frozen_text": True,
            },
            "git": {"commit": "0" * 40, "dirty": False},
            "source_sha256": "abc",
            "torch": "2.7.1+cu126",
        }, f)
    with open(os.path.join(root, "metrics.json"), "w") as f:
        json.dump({
            "split": {"train": 144586, "val": 18073, "test": 18074},
            "epoch_train_seconds": [1712.2, 438.6, 428.6],
            "eval_seconds": 484.7,
            "inference_seconds": 45.1,
            "frozen_text_cache": {"hits": 2110152, "misses": 922908,
                                  "size": 921814},
            "gpu_model": "NVIDIA RTX 6000 Ada Generation",
            "gpu_count": 1,
        }, f)
    with open(os.path.join(root, "test_metrics.json"), "w") as f:
        json.dump({"pr_auc": 0.58939, "roc_auc": 0.90715}, f)
    with open(os.path.join(root, "metrics_history.json"), "w") as f:
        json.dump([
            {"train_cpu_seconds": 4000.0, "train_cpu_util_pct": 233.0,
             "train_vram_allocated_mb": 6500.0, "train_vram_peak_mb": 6900.0},
            {"train_cpu_seconds": 1000.0, "train_cpu_util_pct": 228.0,
             "train_vram_allocated_mb": 6600.0, "train_vram_peak_mb": 6950.0},
        ], f)
    return root


def _thin_run(root):
    # Older-style aggregates only: no run_config, no test persistence.
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "metrics.json"), "w") as f:
        json.dump({
            "split": {"train": 144586, "val": 18073, "test": 18074},
            "epoch_train_seconds": [1681.6, 1717.7, 1707.7],
            "inference_seconds": 530.0,
            "test_scores": {"pr_auc": 0.58968, "roc_auc": 0.90714},
        }, f)
    return root


class TestCollectRunStats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rich = _rich_run(os.path.join(self.tmp.name, "on"))
        self.thin = _thin_run(os.path.join(self.tmp.name, "off"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_rich_row_filled_and_throughput(self):
        row = crs.collect_run(self.rich)
        self.assertEqual(row["tree_sha"], "0" * 40)
        self.assertEqual(row["source_sha256"], "abc")
        self.assertEqual(row["dirty"], False)
        self.assertEqual(row["cache_hits"], 2110152)
        self.assertEqual(row["test_pr_auc"], 0.58939)
        self.assertEqual(row["vram_allocated_peak_mb"], 6600.0)
        self.assertEqual(row["train_cpu_seconds_total"], 5000.0)
        # Train throughput uses the train split over summed epoch walls.
        self.assertAlmostEqual(row["throughput_train_pts_per_s"],
                               144586 / (1712.2 + 438.6 + 428.6), places=6)
        self.assertAlmostEqual(row["throughput_infer_pts_per_s"],
                               18074 / 45.1, places=6)

    def test_thin_row_missing_listed_never_inferred(self):
        row = crs.collect_run(self.thin)
        missing = row["missing_fields"].split(";")
        for field in ("tree_sha", "source_sha256", "seed", "task",
                      "backbone", "gpu_model",
                      "cache_hits", "eval_s", "peak_pss_mb",
                      "train_cpu_seconds_total"):
            self.assertIn(field, missing)
            self.assertIsNone(row[field])
        # Present aggregates still collected.
        self.assertEqual(row["test_pr_auc"], 0.58968)
        self.assertEqual(row["split_test"], 18074)

    def test_nested_config_read(self):
        # Older metrics.json nests the protocol under "config".
        nested = os.path.join(self.tmp.name, "nested")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, "metrics.json"), "w") as f:
            json.dump({"config": {"task": "labs", "model": "rnn",
                                  "seed": 2, "use_amp": False},
                       "split": {"train": 10, "val": 2, "test": 2},
                       "epoch_train_seconds": [5.0],
                       "inference_seconds": 1.0}, f)
        row = crs.collect_run(nested)
        self.assertEqual(row["task"], "labs")
        self.assertEqual(row["backbone"], "rnn")
        self.assertEqual(row["seed"], 2)
        self.assertEqual(row["use_amp"], False)

    def test_pairs_flagged(self):
        rich = crs.collect_run(self.rich)
        same = crs.collect_run(self.rich)
        self.assertEqual(crs.check_pair(rich, same), "comparable")
        other_seed = crs.collect_run(
            _rich_run(os.path.join(self.tmp.name, "on2"), seed=2))
        verdict = crs.check_pair(rich, other_seed)
        self.assertTrue(verdict.startswith("NON-COMPARABLE"))
        self.assertIn("seed", verdict)
        thin = crs.collect_run(self.thin)
        self.assertTrue(crs.check_pair(rich, thin).startswith("UNVERIFIABLE"))

    def test_write_outputs_roundtrip(self):
        rows = [crs.collect_run(self.rich), crs.collect_run(self.thin)]
        prefix = os.path.join(self.tmp.name, "rows")
        csv_path, md_path = crs.write_outputs(rows, prefix)
        with open(csv_path) as f:
            read = list(csv.DictReader(f))
        self.assertEqual(len(read), 2)
        self.assertEqual(read[0]["test_pr_auc"], "0.58939")
        with open(md_path) as f:
            md = f.read()
        self.assertIn("## Pairwise comparability", md)
        # Rich vs thin share no compare-field coverage: unverifiable.
        self.assertIn("UNVERIFIABLE", md)


def _runner():
    import importlib.util
    from pathlib import Path

    path = (Path(REPO_ROOT) / "examples" / "mortality_prediction"
            / "unified_embedding_e2e_mimic4.py")
    spec = importlib.util.spec_from_file_location("unified_e2e_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCollectRealRun(unittest.TestCase):
    """Collector against a real demo-fixture run directory (labs, mlp, CPU).

    The synthetic tests above pin the schema; this one proves the field
    names the runner actually writes land in the right columns, and
    records which standard fields a real run still leaves missing.
    """

    def test_real_run_fields_and_missing(self):
        from pathlib import Path as _Path

        runner = _runner()
        demo = _Path(REPO_ROOT) / "test-resources" / "core" / "mimic4demo"
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(
                ["--ehr-root", str(demo), "--task", "labs", "--model", "mlp",
                 "--epochs", "1", "--batch-size", "8", "--embedding-dim", "8",
                 "--hidden-dim", "8", "--device", "cpu", "--seed", "2",
                 "--no-cache-frozen-text",
                 "--cache-dir", str(_Path(tmp) / "cache"),
                 "--num-workers", "1",
                 "--output-dir", str(_Path(tmp) / "out")]
            )
            run_dir = _Path(runner.run(args)).parent
            row = crs.collect_run(str(run_dir))
            # Runner-written fields land in their columns.
            self.assertEqual(row["task"], "labs")
            self.assertEqual(row["backbone"], "mlp")
            self.assertEqual(row["seed"], 2)
            if (_Path(REPO_ROOT) / ".git").exists():
                self.assertIsNotNone(row["tree_sha"])
                self.assertIn(row["dirty"], (True, False))
            else:
                # Archived trees carry no git metadata: the collector
                # reports the fields missing instead of inventing them.
                self.assertIsNone(row["tree_sha"])
                self.assertIsNone(row["dirty"])
            self.assertGreater(row["split_train"], 0)
            self.assertIsNotNone(row["train_cpu_seconds_total"])
            self.assertIsNotNone(row["train_cpu_util_pct_mean"])
            self.assertEqual(
                (row["cache_hits"], row["cache_misses"], row["cache_entries"]),
                (0, 0, 0))
            self.assertIsNotNone(row["test_pr_auc"])
            # Hardware now comes from the runner's hardware object.
            self.assertEqual(row["gpu_model"], "cpu")
            self.assertEqual(row["gpu_count"], 0)
            self.assertIsNotNone(row["thread_pinning"])
            # New protocol fields land from a real run.
            self.assertEqual(row["drop_modality"], [])
            self.assertEqual(row["lr"], 1e-4)
            self.assertEqual(row["split_mode"], "by_patient")
            self.assertEqual(row["eval_split"], "test")
            # Standard fields a labs run on CPU still leaves missing: the
            # note-text fields (no note processor) and the GPU VRAM peaks.
            missing = row["missing_fields"].split(";")
            expected = ["note_length", "text_encoder",
                        "vram_allocated_peak_mb", "vram_reserved_peak_mb"]
            present = ["positive_rate", "val_pr_auc_best", "peak_rss_mb",
                       "cache_bytes"]
            # PSS needs psutil's memory_full_info().pss, which macOS lacks.
            (expected if sys.platform == "darwin" else present).append(
                "peak_pss_mb")
            if not (_Path(REPO_ROOT) / ".git").exists():
                expected += ["tree_sha", "dirty"]
            for field in expected:
                self.assertIn(field, missing, field)
            for field in present:
                self.assertNotIn(field, missing, field)
            for field in ("gpu_model", "gpu_count", "thread_pinning",
                          "drop_modality", "lr", "adam_eps",
                          "max_grad_norm", "dev", "split_mode",
                          "eval_split", "cache_frozen_text",
                          "source_sha256"):
                self.assertNotIn(field, missing, field)


class TestCollectRealPair(unittest.TestCase):
    """Comparability on two real demo-fixture run directories.

    Same config compares; an --lr change is NON-COMPARABLE naming lr.
    Code identity travels with the runs (run_config.json's
    source_sha256, written on git and archive trees alike), so the
    pair verdicts run on every tree.
    """

    def test_real_pair_same_config_comparable_lr_names_field(self):
        from pathlib import Path as _Path

        runner = _runner()
        demo = _Path(REPO_ROOT) / "test-resources" / "core" / "mimic4demo"

        def _run(tmp, name, extra=()):
            args = runner.parse_args(
                ["--ehr-root", str(demo), "--task", "labs", "--model", "mlp",
                 "--epochs", "1", "--batch-size", "8", "--embedding-dim", "8",
                 "--hidden-dim", "8", "--device", "cpu", "--seed", "2",
                 "--no-cache-frozen-text",
                 "--cache-dir", str(_Path(tmp) / "cache"),
                 "--num-workers", "1",
                 "--output-dir", str(_Path(tmp) / name), *extra]
            )
            return str(_Path(runner.run(args)).parent)

        with tempfile.TemporaryDirectory() as tmp:
            run_a = _run(tmp, "a")
            run_b = _run(tmp, "b")
            run_c = _run(tmp, "c", ("--lr", "0.001"))
            row_a = crs.collect_run(run_a)
            row_b = crs.collect_run(run_b)
            row_c = crs.collect_run(run_c)
            fields = ("drop_modality", "lr", "adam_eps", "max_grad_norm",
                      "dev", "split_mode", "eval_split",
                      "cache_frozen_text", "gpu_model", "gpu_count",
                      "thread_pinning", "source_sha256")
            for field in fields:
                self.assertEqual(row_a[field], row_b[field], field)
                self.assertIsNotNone(row_a[field], field)
            self.assertNotEqual(row_a["lr"], row_c["lr"])
            self.assertEqual(crs.check_pair(row_a, row_b), "comparable")
            verdict = crs.check_pair(row_a, row_c)
            self.assertTrue(verdict.startswith("NON-COMPARABLE"), verdict)
            differed = verdict.split("differ ", 1)[1].split(",")
            self.assertIn("lr", differed)


if __name__ == "__main__":
    unittest.main()
