"""Tests for the one-knob experiment driver.

Pure arm-construction checks run in milliseconds. The demo-data cases
drive the real runner on the public MIMIC-IV demo fixture (CPU, 1 epoch,
seed 2) -- slower, but no downloads except the notes_labs ablation case,
which is skipped unless the text model is already in the local cache.
"""

import csv
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _module(name, relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _driver():
    return _module(
        "unified_experiments",
        Path("examples") / "mortality_prediction" / "unified_embedding_experiments.py",
    )


def _runner():
    return _module(
        "unified_e2e_runner",
        Path("examples") / "mortality_prediction" / "unified_embedding_e2e_mimic4.py",
    )


def _base(driver, runner, **overrides):
    base = runner.parse_args(["--ehr-root", "/tmp/does-not-exist"])
    base.task = "labs"
    base.model = "rnn"
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _fake_run_factory(calls, arm_specs=None):
    """Fake arm runs cycling through per-arm specs.

    Each spec holds ``split``, ``seed``, and ``ids`` (inference-split
    patient ids written to a predictions file, as the real runner does).
    """
    specs = arm_specs or [{}]

    def _fake_run(args):
        spec = specs[len(calls) % len(specs)]
        calls.append(vars(args))
        outdir = Path(args.output_dir) / "fake"
        outdir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "config": {"seed": spec.get("seed", 2)},
            "split": spec.get("split", {"train": 10, "val": 1, "test": 2}),
            "epoch_train_seconds": [1.0],
            "eval_seconds": 0.5,
            "inference_seconds": 0.5,
            "frozen_text_cache": {"hits": 0, "misses": 0, "size": 0},
            "test_scores": {"pr_auc": 0.5},
        }
        (outdir / "metrics.json").write_text(json.dumps(metrics))
        with (outdir / "predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "y_true", "y_prob",
                             "y_pred_threshold_0_5"])
            for pid in spec.get("ids", ["p1", "p2"]):
                writer.writerow([pid, 0, 0.5, 1])
        return str(outdir / "predictions.csv")

    return _fake_run


class TestFlagTable(unittest.TestCase):
    def test_every_dest_exists_in_runner(self):
        driver = _driver()
        runner = _runner()
        probe = runner.parse_args(["--ehr-root", "/tmp/does-not-exist"])
        for dest in driver._CONVERTERS:
            self.assertTrue(
                hasattr(probe, dest), f"converter entry {dest!r} missing from runner"
            )

    def test_alias(self):
        self.assertEqual(_driver().resolve_flag("window-hours"),
                         "observation_window_hours")

    def test_unknown_flag_refused(self):
        driver = _driver()
        with self.assertRaises(ValueError):
            driver.resolve_flag("learning-rate-warmup")

    def test_model_scope_comes_from_runner(self):
        # The driver reuses the runner's own wiring map. JambaEHR receives
        # --heads from the runner, so it is a live knob (the old driver
        # table refused it); --num-layers never reaches JambaEHR, so it is
        # refused (the old table allowed the no-op arm).
        driver = _driver()
        runner = _runner()
        base = _base(driver, runner, task="notes_labs", model="jambaehr")
        arms = driver.build_arms("ab", base, flag="heads", values=["2", "4"])
        self.assertEqual([a[2]["heads"] for a in arms], [2, 4])
        with self.assertRaises(ValueError):
            driver.build_arms("ab", base, flag="num-layers",
                              values=["1", "2"])


class TestBuildArms(unittest.TestCase):
    def test_ab_boolean_defaults_to_on_off(self):
        driver = _driver()
        arms = driver.build_arms("ab", _base(driver, _runner()), flag="bidirectional")
        self.assertEqual([a[0] for a in arms],
                         ["ab_bidirectional_off", "ab_bidirectional_on"])
        self.assertEqual([a[2] for a in arms],
                         [{"bidirectional": False}, {"bidirectional": True}])

    def test_ab_values(self):
        driver = _driver()
        arms = driver.build_arms("ab", _base(driver, _runner()),
                                 flag="batch-size", values=["8", "4"])
        self.assertEqual([a[2]["batch_size"] for a in arms], [8, 4])

    def test_ab_refusals(self):
        driver = _driver()
        runner = _runner()
        with self.assertRaises(ValueError):  # identical arms
            driver.build_arms("ab", _base(driver, runner),
                              flag="batch-size", values=["8", "8"])
        with self.assertRaises(ValueError):  # non-boolean needs values
            driver.build_arms("ab", _base(driver, runner), flag="batch-size")
        with self.assertRaises(ValueError):  # unknown flag
            driver.build_arms("ab", _base(driver, runner), flag="frobnicate")
        with self.assertRaises(ValueError):  # inert for task
            driver.build_arms("ab", _base(driver, runner),
                              flag="cache-frozen-text")
        with self.assertRaises(ValueError):  # inert for backbone
            driver.build_arms(
                "ab", _base(driver, runner, model="transformer"),
                flag="hidden-dim", values=["8", "16"])

    def test_sweep(self):
        driver = _driver()
        arms = driver.build_arms("ab", _base(driver, _runner()),
                                 flag="bidirectional", values=["0", "1"])
        self.assertEqual([a[2]["bidirectional"] for a in arms], [False, True])
        arms = driver.build_arms("sweep", _base(driver, _runner()),
                                 flag="window-hours", values=["0", "24", "48"])
        self.assertEqual([a[2]["observation_window_hours"] for a in arms],
                         [0.0, 24.0, 48.0])
        with self.assertRaises(ValueError):  # duplicates
            driver.build_arms("sweep", _base(driver, _runner()),
                              flag="window-hours", values=["0", "0"])
        with self.assertRaises(ValueError):  # single value
            driver.build_arms("sweep", _base(driver, _runner()),
                              flag="window-hours", values=["0"])

    def test_ablate_modalities(self):
        driver = _driver()
        runner = _runner()
        arms = driver.build_arms(
            "ablate", _base(driver, runner, task="notes_labs"),
            modalities=["notes"])
        self.assertEqual([a[0] for a in arms], ["ablate_full", "ablate_no_notes"])
        self.assertEqual(arms[0][2], {"drop_modality": []})
        self.assertEqual(arms[1][2], {"drop_modality": ["notes"]})
        arms = driver.build_arms(
            "ablate", _base(driver, runner, task="notes_labs"),
            modalities=["labs"])
        self.assertEqual([a[0] for a in arms], ["ablate_full", "ablate_no_labs"])
        self.assertEqual(arms[1][2], {"drop_modality": ["labs"]})
        arms = driver.build_arms(
            "ablate", _base(driver, runner, task="notes_labs"),
            modalities=["notes", "labs"])
        self.assertEqual([a[0] for a in arms],
                         ["ablate_full", "ablate_no_notes", "ablate_no_labs"])
        with self.assertRaises(ValueError):  # unknown modality
            driver.build_arms("ablate", _base(driver, runner, task="notes_labs"),
                              modalities=["image"])
        with self.assertRaises(ValueError):  # duplicates
            driver.build_arms("ablate", _base(driver, runner, task="notes_labs"),
                              modalities=["notes", "notes"])
        with self.assertRaises(ValueError):  # labs task has no notes
            driver.build_arms("ablate", _base(driver, runner, task="labs"),
                              modalities=["notes"])
        with self.assertRaises(ValueError):  # nothing left on labs either
            driver.build_arms("ablate", _base(driver, runner, task="labs"),
                              modalities=["labs"])
        with self.assertRaises(ValueError):  # nothing to drop
            driver.build_arms("ablate", _base(driver, runner))

    def test_ablate_off(self):
        driver = _driver()
        runner = _runner()
        arms = driver.build_arms(
            "ablate", _base(driver, runner, model="rnn", bidirectional=True),
            offs=["bidirectional"])
        self.assertEqual([a[0] for a in arms],
                         ["ablate_full", "ablate_no_bidirectional"])
        self.assertEqual(arms[0][2], {"drop_modality": []})
        self.assertEqual(arms[1][2], {"bidirectional": False})
        with self.assertRaises(ValueError):  # already off: no-op arm
            driver.build_arms("ablate", _base(driver, runner),
                              offs=["bidirectional"])
        with self.assertRaises(ValueError):  # not a boolean component
            driver.build_arms(
                "ablate", _base(driver, runner, model="rnn", bidirectional=True),
                offs=["hidden-dim"])

    def test_single_knob_guard(self):
        driver = _driver()
        base = {"batch_size": 8, "seed": 2, "output_dir": "root"}
        arm = {"batch_size": 4, "seed": 2, "output_dir": "root/arm"}
        driver.assert_single_knob(base, arm, "batch_size")
        bad = dict(arm, seed=3)
        with self.assertRaises(ValueError):
            driver.assert_single_knob(base, bad, "batch_size")

    def test_main_fake_run_writes_summary(self):
        driver = _driver()
        calls = []
        with TemporaryDirectory() as tmp:
            root = driver.main(
                ["--output-root", str(Path(tmp) / "exp"),
                 "ab", "--flag", "batch-size", "--values", "8", "4", "--",
                 "--ehr-root", "/tmp/does-not-exist", "--seed", "2",
                 "--batch-size", "8"],
                run_fn=_fake_run_factory(calls),
            )
            self.assertEqual(len(calls), 2)
            # Extra args pass through to every arm unchanged.
            self.assertTrue(all(c["seed"] == 2 for c in calls))
            self.assertEqual([c["batch_size"] for c in calls], [8, 4])
            # Each arm gets its own output directory.
            self.assertEqual(len({c["output_dir"] for c in calls}), 2)
            summary = Path(root) / "summary.csv"
            self.assertTrue(summary.is_file())
            md = (Path(root) / "summary.md").read_text()
            with summary.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["seed"], "2")
            self.assertIn("identical across arms", md)

    def test_main_summary_reports_differences(self):
        # The summary computes identity; it never asserts it. Arms with
        # different seeds, splits, and samples say so, naming the arms.
        driver = _driver()
        calls = []
        specs = [
            {"seed": 2, "split": {"train": 10, "val": 1, "test": 2},
             "ids": ["p1", "p2"]},
            {"seed": 3, "split": {"train": 9, "val": 1, "test": 2},
             "ids": ["p1", "p3"]},
        ]
        with TemporaryDirectory() as tmp:
            root = driver.main(
                ["--output-root", str(Path(tmp) / "exp"),
                 "ab", "--flag", "batch-size", "--values", "8", "4", "--",
                 "--ehr-root", "/tmp/does-not-exist", "--seed", "2",
                 "--batch-size", "8"],
                run_fn=_fake_run_factory(calls, specs),
            )
            md = (Path(root) / "summary.md").read_text()
            self.assertIn("Seeds DIFFER across arms", md)
            self.assertIn("Split sizes DIFFER across arms", md)
            self.assertIn("Sample identity DIFFERS across arms", md)

    def test_main_refuses_before_running(self):
        driver = _driver()
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                driver.main(
                    ["ab", "--flag", "cache-frozen-text", "--",
                     "--ehr-root", "/tmp/does-not-exist"],
                    run_fn=lambda args: (_ for _ in ()).throw(
                        AssertionError("must refuse before running")),
                )
            with self.assertRaises(SystemExit):
                driver.main(
                    ["sweep", "--flag", "batch-size", "--values", "8", "--",
                     "--ehr-root", "/tmp/does-not-exist",
                     "--output-dir", str(Path(tmp) / "x")],
                    run_fn=lambda args: (_ for _ in ()).throw(
                        AssertionError("must refuse before running")),
                )
            with self.assertRaises(SystemExit):  # runner args need '--'
                driver.main(
                    ["ab", "--flag", "batch-size", "--values", "8", "4",
                     "--ehr-root", "/tmp/does-not-exist"],
                    run_fn=lambda args: (_ for _ in ()).throw(
                        AssertionError("must refuse before running")),
                )
            with self.assertRaises(SystemExit):  # nothing left on labs
                driver.main(
                    ["ablate", "--modalities", "labs", "--",
                     "--ehr-root", "/tmp/does-not-exist",
                     "--task", "labs"],
                    run_fn=lambda args: (_ for _ in ()).throw(
                        AssertionError("must refuse before running")),
                )
            with self.assertRaises(SystemExit):  # ablate owns --drop-modality
                driver.main(
                    ["ablate", "--modalities", "notes", "--",
                     "--ehr-root", "/tmp/does-not-exist",
                     "--task", "notes_labs",
                     "--drop-modality", "notes"],
                    run_fn=lambda args: (_ for _ in ()).throw(
                        AssertionError("must refuse before running")),
                )

    def test_single_cached_runner_loader(self):
        # One cached _load_runner(); no _cached_runner or _CACHED_RUNNER.
        driver = _driver()
        self.assertFalse(hasattr(driver, "_cached_runner"))
        self.assertFalse(hasattr(driver, "_CACHED_RUNNER"))
        self.assertIs(driver._load_runner(), driver._load_runner())

    def test_main_summary_empty_predictions_not_recorded(self):
        # An empty predictions file (header only) reports identity as
        # not recorded, not "0 ids match".
        driver = _driver()
        calls = []
        specs = [{"seed": 2, "ids": []}]
        with TemporaryDirectory() as tmp:
            root = driver.main(
                ["--output-root", str(Path(tmp) / "exp"),
                 "ab", "--flag", "batch-size", "--values", "8", "4", "--",
                 "--ehr-root", "/tmp/does-not-exist", "--seed", "2",
                 "--batch-size", "8"],
                run_fn=_fake_run_factory(calls, specs),
            )
            md = (Path(root) / "summary.md").read_text()
            self.assertIn("Sample identity not recorded", md)


DEMO_ROOT = (
    Path(__file__).resolve().parents[2] / "test-resources" / "core" / "mimic4demo"
)


def _demo_argv(mode_args, tmp, root, cache, model="rnn", extra=()):
    return ["--output-root", str(Path(tmp) / root)] + mode_args + [
        "--",
        "--ehr-root", str(DEMO_ROOT),
        "--task", "labs", "--model", model,
        "--epochs", "1", "--batch-size", "8",
        "--embedding-dim", "8", "--hidden-dim", "8",
        "--device", "cpu", "--seed", "2",
        "--no-cache-frozen-text",
        "--cache-dir", cache,
        "--num-workers", "1",
        *extra,
    ]


def _summaries(testcase, root, arms):
    summary = Path(root) / "summary.csv"
    testcase.assertTrue(summary.is_file())
    md = (Path(root) / "summary.md").read_text()
    with summary.open() as f:
        rows = list(csv.DictReader(f))
    testcase.assertEqual(len(rows), arms)
    # Same seed across arms; splits checked against metrics.json below.
    testcase.assertEqual({row["seed"] for row in rows}, {"2"})
    testcase.assertIn("identical across arms", md)
    return rows


def _check_same_split(testcase, root, rows):
    splits = set()
    for row in rows:
        for metrics_path in (Path(root) / row["arm"]).rglob("metrics.json"):
            metrics = json.loads(metrics_path.read_text())
            splits.add(json.dumps(metrics["split"], sort_keys=True))
            testcase.assertEqual(metrics["config"]["seed"], 2)
    testcase.assertEqual(len(splits), 1, f"splits differ: {splits}")


class TestDriverDemoData(unittest.TestCase):
    """One real demo-fixture case per mode (labs, CPU, seconds each)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        cls.cache = str(Path(cls._tmp.name) / "cache")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, mode_args, root, model="rnn", extra=()):
        driver = _driver()
        with TemporaryDirectory() as tmp:
            out = driver.main(
                _demo_argv(mode_args, tmp, root, self.cache, model, extra),
            )
            rows = _summaries(self, out, 2)
            _check_same_split(self, out, rows)
            return rows

    def test_ab_bidirectional(self):
        rows = self._run(["ab", "--flag", "bidirectional"], "ab", extra=())
        self.assertEqual([r["differs"] for r in rows],
                         ["bidirectional=False", "bidirectional=True"])

    def test_sweep_window(self):
        rows = self._run(
            ["sweep", "--flag", "window-hours", "--values", "0", "24"],
            "sweep", model="mlp")
        self.assertEqual([r["differs"] for r in rows],
                         ["observation_window_hours=0.0",
                          "observation_window_hours=24.0"])

    def test_ablate_off(self):
        rows = self._run(["ablate", "--off", "bidirectional"], "ablate",
                         extra=("--bidirectional",))
        self.assertEqual(rows[0]["differs"], "none (reference)")
        self.assertEqual(rows[1]["differs"], "bidirectional=False")


def _write_note_csvs(note_dir):
    import gzip

    def _write(path, rows):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["subject_id", "charttime", "note_id", "hadm_id", "note_type",
                 "note_seq", "storetime", "text"]
            )
            writer.writerows(rows)

    _write(
        note_dir / "discharge.csv.gz",
        [
            (10008, "2151-10-05", 1, 20012, "discharge", 1,
             "2151-10-25 12:00:00", "Chief Complaint:\n\nshortness of breath\n\n"),
            (10008, "2151-12-15", 2, 20013, "discharge", 1,
             "2151-12-20 12:00:00", "Chief Complaint:\n\nchest pain\n\n"),
        ],
    )
    _write(
        note_dir / "radiology.csv.gz",
        [
            (10008, "2151-10-06 09:00:00", 3, 20012, "radiology", 1,
             "2151-10-06 10:00:00", "Impression:\n\nno acute findings\n\n"),
        ],
    )


class TestDriverDemoAblate(unittest.TestCase):
    """Real same-task modality ablation on the demo fixture.

    notes_labs reference plus the arm with notes dropped: identical split
    sizes and sample ids across arms, and zero frozen-cache traffic in
    the drop arm. The labs drop is covered at the arm-construction level
    (`test_ablate_modalities`) and against the real cohort in the
    cluster ablation run: on this 13-sample toy fixture the notes-only
    arm leaves batches the RNN backbone cannot pack (pre-existing
    all-empty-batch limit, also present on the labs task), so the demo
    run keeps the arms that hold labs. Needs the text model in the local
    HF cache; tests never download.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001 -- any load failure means skip
            raise unittest.SkipTest(f"text model not in local HF cache: {exc}")
        cls._tmp = TemporaryDirectory()
        cls.cache = str(Path(cls._tmp.name) / "cache")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_ablate_modalities_same_task(self):
        driver = _driver()
        with TemporaryDirectory() as tmp:
            note_dir = Path(tmp) / "notes" / "note"
            note_dir.mkdir(parents=True, exist_ok=True)
            _write_note_csvs(note_dir)
            out = driver.main(
                ["--output-root", str(Path(tmp) / "ablate"),
                 "ablate", "--modalities", "notes", "--",
                 "--ehr-root", str(DEMO_ROOT),
                 "--note-root", str(Path(tmp) / "notes"),
                 "--task", "notes_labs", "--model", "rnn",
                 "--epochs", "1", "--batch-size", "8",
                 "--embedding-dim", "8", "--hidden-dim", "8",
                 "--device", "cpu", "--seed", "2",
                 "--freeze-encoder",
                 "--cache-dir", self.cache,
                 "--num-workers", "1"],
            )
            rows = _summaries(self, out, 2)
            self.assertEqual(
                [r["arm"] for r in rows],
                ["ablate_full", "ablate_no_notes"])
            by_arm = {}
            expected_drop = {"ablate_full": [],
                             "ablate_no_notes": ["notes"]}
            for row in rows:
                matches = list((Path(out) / row["arm"]).rglob("metrics.json"))
                self.assertEqual(len(matches), 1)
                metrics = json.loads(matches[0].read_text())
                by_arm[row["arm"]] = metrics
                self.assertEqual(metrics["config"]["seed"], 2)
                self.assertEqual(metrics["config"]["drop_modality"],
                                 expected_drop[row["arm"]])
            splits = {json.dumps(m["split"], sort_keys=True)
                      for m in by_arm.values()}
            self.assertEqual(len(splits), 1, f"splits differ: {splits}")
            ids = {}
            for row in rows:
                preds = list((Path(out) / row["arm"]).rglob("predictions*.csv"))
                self.assertEqual(len(preds), 1)
                with preds[0].open() as handle:
                    ids[row["arm"]] = sorted(
                        r["patient_id"] for r in csv.DictReader(handle))
            self.assertEqual(ids["ablate_full"], ids["ablate_no_notes"])
            full_cache = by_arm["ablate_full"]["frozen_text_cache"]
            self.assertGreater(full_cache["misses"] + full_cache["size"], 0)
            no_notes_cache = by_arm["ablate_no_notes"]["frozen_text_cache"]
            self.assertEqual(
                (no_notes_cache["hits"], no_notes_cache["misses"],
                 no_notes_cache["size"]), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
