"""Empirical proofs for the tranche-1 prep commits.

Each TestCase maps onto one proposed commit. These tests are the evidence
that commit is supposed to carry; cluster GPU jobs re-run the CUDA subset.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import unittest
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "examples" / "mortality_prediction" / "unified_embedding_e2e_mimic4.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("e2e_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(mod, *argv):
    with mock.patch.object(sys, "argv", ["e2e.py", "--ehr-root", "/tmp", *argv]):
        return mod.parse_args()


class TestScanParquetRestored(unittest.TestCase):
    """commit: restore BaseDataset._scan_table / _scan_parquet"""

    def test_meds_still_calls_scan_parquet(self):
        from pyhealth.datasets.base_dataset import BaseDataset
        from pyhealth.datasets.meds import MEDSDataset

        self.assertTrue(callable(getattr(BaseDataset, "_scan_table")))
        self.assertTrue(callable(getattr(BaseDataset, "_scan_parquet")))
        src = inspect.getsource(MEDSDataset._subset_patient_ids)
        self.assertIn("_scan_parquet", src)


class TestRnnLengthClamp(unittest.TestCase):
    """commit: clamp RNN packed lengths at 1"""

    def test_all_pad_mask_does_not_raise(self):
        from pyhealth.models.rnn import RNNLayer

        layer = RNNLayer(input_size=4, hidden_size=8, dropout=0.0).eval()
        x = torch.zeros(2, 5, 4)
        mask = torch.zeros(2, 5)
        with torch.no_grad():
            outputs, last = layer(x, mask)
        self.assertEqual(outputs.shape[0], 2)
        self.assertEqual(last.shape, (2, 8))
        self.assertTrue(torch.isfinite(last).all())


class TestPaddingIdx(unittest.TestCase):
    """commit: keep padding_idx=0 on nested code embeddings"""

    def test_nested_sequence_embedding_uses_padding_idx_zero(self):
        from pyhealth.datasets import create_sample_dataset
        from pyhealth.models.embedding.vanilla import EmbeddingModel

        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["A", "B"], ["C"]],
                "label": 0,
            },
            {
                "patient_id": "p1",
                "visit_id": "v0",
                "conditions": [["A"]],
                "label": 1,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
            in_memory=True,
        )
        model = EmbeddingModel(dataset, embedding_dim=8)
        emb = model.embedding_layers["conditions"]
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.padding_idx, 0)
        self.assertTrue(torch.equal(emb.weight[0], torch.zeros_like(emb.weight[0])))


class TestObservationWindow(unittest.TestCase):
    """protocol: full stay, no window_hours API"""

    def test_collection_end_is_discharge(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        task = LabsMIMIC4()
        admit = datetime(2020, 1, 1, 0, 0, 0)
        disch = datetime(2020, 1, 10, 0, 0, 0)
        self.assertEqual(task._admission_window_end(admit, disch), disch)

    def test_no_window_hours_attribute(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        self.assertFalse(hasattr(LabsMIMIC4(), "window_hours"))

    def test_cache_version_bumped(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        self.assertGreaterEqual(LabsMIMIC4().emitted_data_version, 5)


class TestFusedSdpaAndAmp(unittest.TestCase):
    """commit: fused SDPA, dtype-safe mask fill, validated AMP dtype"""

    def test_fp16_mask_fill_stays_finite(self):
        from pyhealth.models.transformer import Attention

        attn = Attention()
        q = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
        k = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
        v = torch.ones(1, 1, 2, 4, dtype=torch.float16)
        mask = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float16)
        out, weights = attn(q, k, v, mask=mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights[0, 0, 0, 1].item(), 0.0)

    def test_ordinary_forward_uses_sdpa(self):
        src = inspect.getsource(
            __import__(
                "pyhealth.models.transformer", fromlist=["MultiHeadedAttention"]
            ).MultiHeadedAttention.forward
        )
        self.assertIn("scaled_dot_product_attention", src)
        self.assertIn("register_hook", src)

    def test_resolve_amp_dtype_accepts_aliases_and_rejects_junk(self):
        from pyhealth.trainer import resolve_amp_dtype

        self.assertEqual(resolve_amp_dtype("bf16", use_amp=True), torch.bfloat16)
        self.assertEqual(resolve_amp_dtype("bfloat16", use_amp=True), torch.bfloat16)
        self.assertEqual(resolve_amp_dtype("fp16", use_amp=True), torch.float16)
        with self.assertRaises(ValueError):
            resolve_amp_dtype("fp32", use_amp=True)

    def test_sdpa_matches_explicit_path_on_cpu(self):
        from pyhealth.models.transformer import MultiHeadedAttention

        torch.manual_seed(0)
        mha = MultiHeadedAttention(h=2, d_model=16, dropout=0.0).eval()
        x = torch.randn(2, 5, 16, requires_grad=True)
        fused = mha(x, x, x, mask=None, register_hook=False)
        explicit = mha(x, x, x, mask=None, register_hook=True)
        self.assertTrue(torch.allclose(fused, explicit, atol=1e-5, rtol=1e-4))

        mask = torch.ones(2, 5)
        mask[1, 3:] = 0
        fused_m = mha(x, x, x, mask=mask, register_hook=False)
        self.assertTrue(torch.isfinite(fused_m).all())
        self.assertTrue(torch.allclose(fused_m[1, 3:], torch.zeros_like(fused_m[1, 3:]), atol=1e-6))


class TestPadMasks(unittest.TestCase):
    """commit: emit event pad masks and pad every ragged dimension"""

    def test_tuple_collate_emits_pad_mask_and_pads_both_dims(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

        batch = [
            {"notes": (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))},
            {"notes": (torch.ones(1, 5), torch.ones(1, 5, dtype=torch.long))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        ids = collated["notes"][0]
        self.assertEqual(tuple(ids.shape), (2, 2, 5))
        mask = collated[f"notes{PAD_MASK_SUFFIX}"]
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])

    def test_heads_thread_pad_mask(self):
        from pyhealth.models.bottleneck_transformer import BottleneckTransformer
        from pyhealth.models.ehrmamba import EHRMamba
        from pyhealth.models.jamba_ehr import JambaEHR
        from pyhealth.models.mlp import MLP
        from pyhealth.models.rnn import RNN
        from pyhealth.models.transformer import Transformer

        for cls in (MLP, RNN, Transformer, BottleneckTransformer, EHRMamba, JambaEHR):
            src = inspect.getsource(cls._build_unified_inputs)
            self.assertIn("PAD_MASK_SUFFIX", src, msg=cls.__name__)
            self.assertIn("pad_mask", src, msg=cls.__name__)


class TestUnifiedContentAndSort(unittest.TestCase):
    """commit: content LN, stable pad-last sort, skip observation-mask fields"""

    def test_padding_sorts_last_and_content_is_normalized(self):
        from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor()
        model = UnifiedMultimodalEmbeddingModel(
            processors={"labs": proc},
            embedding_dim=8,
            normalize_content=True,
        )
        model.encoders["labs"] = nn.Linear(2, 8)
        value = torch.tensor([[[10.0, 0.0], [0.0, 0.0]]])
        time = torch.tensor([[6.0, 0.0]])
        pad_mask = torch.tensor([[True, False]])
        out = model({"labs": {"value": value, "time": time, "pad_mask": pad_mask}})
        self.assertEqual(out["mask"].tolist(), [[1.0, 0.0]])
        self.assertAlmostEqual(out["time"][0, 0].item(), 6.0)
        self.assertTrue(torch.allclose(out["sequence"][0, 1], torch.zeros(8), atol=1e-6))

    def test_observation_mask_field_is_not_encoded(self):
        from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        model = UnifiedMultimodalEmbeddingModel(
            processors={
                "labs": StageNetTensorProcessor(),
                "labs_mask": StageNetTensorProcessor(),
            },
            embedding_dim=8,
        )
        model.encoders["labs"] = nn.Linear(2, 8)
        model.encoders["labs_mask"] = nn.Linear(2, 8)
        out = model(
            {
                "labs": {
                    "value": torch.ones(1, 1, 2),
                    "time": torch.tensor([[6.0]]),
                },
                "labs_mask": {
                    "value": torch.ones(1, 1, 2),
                    "time": torch.tensor([[6.0]]),
                },
            }
        )
        self.assertEqual(out["sequence"].shape[1], 1)


class TestLabStandardizer(unittest.TestCase):
    """commit: train-split lab z-score before the numeric projection"""

    def test_fit_ignores_unobserved_and_zscores_observed(self):
        from pyhealth.processors import fit_lab_standardizer

        standardizer = fit_lab_standardizer(
            [
                {
                    "labs": torch.tensor([[10.0, 100.0]]),
                    "labs_mask": torch.tensor([[True, False]]),
                },
                {
                    "labs": torch.tensor([[12.0, 200.0]]),
                    "labs_mask": torch.tensor([[True, False]]),
                },
            ]
        )
        x = torch.tensor([[[11.0, 150.0]]])
        obs = torch.tensor([[[True, True]]])
        z = standardizer(x, obs)
        self.assertAlmostEqual(z[0, 0, 0].item(), 0.0, places=5)
        self.assertEqual(z[0, 0, 1].item(), 0.0)

    def test_mask_processor_accepts_forward_fill_false(self):
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor(forward_fill=False)
        time, value = proc.process(([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(value.tolist(), [[1.0, 0.0], [0.0, 1.0]])


class TestTokenBudget(unittest.TestCase):
    """commit: default note token budget 512 with longest padding"""

    def test_processor_and_task_default_to_512_longest(self):
        from pyhealth.processors.tuple_time_text_processor import TupleTimeTextProcessor
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        proc = TupleTimeTextProcessor()
        self.assertEqual(proc.max_length, 512)
        src = inspect.getsource(TupleTimeTextProcessor.process)
        self.assertIn('padding="longest"', src)
        schema = NotesLabsMIMIC4._BASE_INPUT_SCHEMA["admission_note_times"][1]
        self.assertEqual(schema["max_length"], 512)


class TestMlpWired(unittest.TestCase):
    """commit: wire unified MLP into the e2e runner"""

    def test_cli_accepts_mlp_and_model_has_unified_path(self):
        from pyhealth.models import MLP

        self.assertTrue(hasattr(MLP, "_forward_unified"))
        mod = _load_runner()
        args = _parse(mod, "--model", "mlp")
        self.assertEqual(args.model, "mlp")


class TestSplitAndEvalFallback(unittest.TestCase):
    """commit: warn and record leaky split / eval fallbacks"""

    def test_split_warns_and_labels_leaky_fallback(self):
        from pyhealth.datasets import create_sample_dataset

        mod = _load_runner()
        samples = [
            {
                "patient_id": "only",
                "visit_id": "v0",
                "labs": [1.0, 2.0],
                "label": 0,
            },
            {
                "patient_id": "only",
                "visit_id": "v1",
                "labs": [3.0, 4.0],
                "label": 1,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"labs": "tensor"},
            output_schema={"label": "binary"},
            in_memory=True,
        )
        # The leaky fallback is now opt-in: by default a cohort that cannot be
        # split by patient stops the run instead of producing numbers where one
        # patient sits in both train and test.
        with self.assertRaises(RuntimeError):
            mod._split_dataset(dataset, seed=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train, val, test, mode = mod._split_dataset(
                dataset, seed=1, allow_leaky_split=True
            )
        self.assertEqual(mode, "by_sample_fallback_leaky")
        self.assertTrue(any("split_by_sample" in str(w.message) for w in caught))
        self.assertGreater(len(train) + len(val) + len(test), 0)

    def test_runner_records_eval_split(self):
        src = RUNNER.read_text()
        compact = "".join(src.split())
        self.assertIn('eval_split=test_loader,"test"', compact)
        self.assertIn("write_run_config", src)


class TestCliAndRunDirs(unittest.TestCase):
    """commit: CLI aliases, reject --patients, include task in run dir"""

    def test_underscore_aliases_bind_the_hyphenated_dests(self):
        mod = _load_runner()
        args = _parse(
            mod,
            "--learning_rate",
            "1e-4",
            "--batch_size",
            "32",
            "--num_layers",
            "2",
            # --amp_dtype needs --use-amp: passing it alone is refused rather
            # than running fp32 with bf16 recorded in the config.
            "--use-amp",
            "--amp_dtype",
            "bf16",
            "--conv_kernel",
            "4",
            "--mamba_state_size",
            "16",
            "--jamba_mamba_layers",
            "2",
        )
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.num_layers, 2)
        self.assertEqual(args.amp_dtype, "bf16")
        self.assertEqual(args.mamba_conv_kernel, 4)
        self.assertEqual(args.mamba_state_size, 16)
        self.assertEqual(args.jamba_mamba_layers, 2)
        self.assertEqual(args.embedding_dim, 128)
        self.assertTrue(args.use_amp)

    def test_amp_dtype_without_use_amp_is_rejected(self):
        # This combination used to parse fine and run fp32 while run_config
        # recorded amp_dtype "bf16" — the Tranche 1 flag list spells it exactly
        # that way, so it has to fail loudly rather than quietly.
        mod = _load_runner()
        with self.assertRaises(SystemExit):
            _parse(mod, "--amp_dtype", "bf16")

    def test_patients_is_rejected(self):
        mod = _load_runner()
        with self.assertRaises(SystemExit):
            _parse(mod, "--patients", "5")

    def test_exp_name_includes_task(self):
        src = RUNNER.read_text()
        self.assertIn('f"{args.task}_{args.model}_seed{args.seed}"', src)
        self.assertNotIn('f"{args.model}_seed{args.seed}"', src)


class TestProvenance(unittest.TestCase):
    """commit: persist run_config.json with git and source digest"""

    def test_write_run_config_records_resolved_fields(self):
        from pyhealth.utils import write_run_config
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = write_run_config(tmp, {"resolved_lr": 1e-4, "split_mode": "by_patient"})
            data = json.loads(Path(path).read_text())
        self.assertEqual(data["config"]["resolved_lr"], 1e-4)
        self.assertIn("git", data)
        self.assertIn("source_sha256", data)
        self.assertEqual(len(data["source_sha256"]), 64)


class TestResizedImages(unittest.TestCase):
    """commit: accept resized_images for the sunlab CXR layout"""

    def test_prepare_metadata_accepts_resized_images(self):
        from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

        src = inspect.getsource(MIMIC4CXRSunlabDataset.prepare_metadata)
        self.assertIn("resized_images", src)
        self.assertIn("images", src)


class TestDataloaderWorkers(unittest.TestCase):
    """commit: DataLoader worker / pin_memory / prefetch kwargs"""

    def test_get_dataloader_forwards_worker_kwargs(self):
        from pyhealth.datasets.utils import get_dataloader

        sig = inspect.signature(get_dataloader)
        for name in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
            self.assertIn(name, sig.parameters)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCudaSubset(unittest.TestCase):
    """GPU-only proofs. Run on the cluster; skipped on CPU laptops."""

    def test_fp16_attention_on_cuda(self):
        from pyhealth.models.transformer import Attention

        attn = Attention().cuda()
        q = torch.zeros(2, 2, 8, 16, dtype=torch.float16, device="cuda")
        k = torch.zeros(2, 2, 8, 16, dtype=torch.float16, device="cuda")
        v = torch.ones(2, 2, 8, 16, dtype=torch.float16, device="cuda")
        mask = torch.ones(2, 1, 8, 8, device="cuda")
        mask[:, :, :, -2:] = 0
        out, weights = attn(q, k, v, mask=mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(weights).all())

    def test_fused_sdpa_on_cuda(self):
        from pyhealth.models.transformer import MultiHeadedAttention

        mha = MultiHeadedAttention(h=4, d_model=64, dropout=0.0).cuda().eval()
        x = torch.randn(4, 32, 64, device="cuda", requires_grad=True)
        fused = mha(x, x, x, mask=None, register_hook=False)
        explicit = mha(x, x, x, mask=None, register_hook=True)
        self.assertTrue(torch.allclose(fused, explicit, atol=1e-4, rtol=1e-3))
        mask = torch.ones(4, 32, device="cuda")
        mask[:, 24:] = 0
        fused_m = mha(x, x, x, mask=mask, register_hook=False)
        self.assertTrue(torch.isfinite(fused_m).all())
        self.assertTrue(torch.allclose(fused_m[:, 24:], torch.zeros_like(fused_m[:, 24:]), atol=1e-5))

    def test_amp_bf16_autocast_dtype(self):
        from pyhealth.trainer import resolve_amp_dtype

        dtype = resolve_amp_dtype("bf16", use_amp=True)
        linear = nn.Linear(32, 32).cuda()
        x = torch.randn(8, 32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=dtype):
            y = linear(x)
        self.assertEqual(y.dtype, torch.bfloat16)


class TestNoMissingPlaceholder(unittest.TestCase):
    """commit: emit zero events instead of a fake [MISSING_TEXT] / empty-path row"""

    def test_empty_notes_are_not_missing_text_token(self):
        from pyhealth.processors import TupleTimeTextProcessor

        processor = TupleTimeTextProcessor()
        texts, time, tag = processor.process(([], []))
        self.assertEqual(texts, [])
        self.assertEqual(tuple(time.shape), (0,))
        self.assertNotIn("[MISSING_TEXT]", texts)

    def test_tokenized_empty_notes_skip_tokenizer(self):
        from pyhealth.processors.tuple_time_text_processor import (
            TupleTimeTextProcessor,
        )

        processor = TupleTimeTextProcessor.__new__(TupleTimeTextProcessor)
        processor.type_tag = "note"
        processor.tokenizer_model = "dummy"
        processor.max_length = 512
        processor.padding = True
        processor.truncation = True
        tokenizer = mock.Mock()
        processor.tokenizer = tokenizer
        ids, attn, types, time, tag = processor.process((["  ", ""], [0.0, 1.0]))
        tokenizer.assert_not_called()
        self.assertEqual(tuple(ids.shape), (0, 1))
        self.assertEqual(tuple(attn.shape), (0, 1))
        self.assertEqual(tuple(time.shape), (0,))

    def test_empty_labs_are_zero_events(self):
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        processor = StageNetTensorProcessor()
        processor.fit(
            [{"labs": ([0.0], [[1.0] * 10])}],
            "labs",
        )
        time, values = processor.process(([], []))
        self.assertEqual(tuple(values.shape), (0, 10))
        self.assertEqual(tuple(time.shape), (0,))

    def test_empty_images_are_zero_events(self):
        from pyhealth.processors.time_image_processor import TimeImageProcessor

        processor = TimeImageProcessor()
        images, timestamps, tag = processor.process(([], []))
        self.assertEqual(images.shape[0], 0)
        self.assertEqual(tuple(timestamps.shape), (0,))
        self.assertEqual(tag, "image")


class TestFrozenEncoderEval(unittest.TestCase):
    """commit: keep frozen BERT in eval when Trainer calls model.train()"""

    def test_train_keeps_frozen_text_encoder_in_eval(self):
        from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor()
        proc.fit([{"labs": ([0.0], [[1.0] * 10])}], "labs")
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": proc}, embedding_dim=8, freeze_text_encoder=True
        )

        class TinyEnc(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(p=0.9)
                self.config = type("C", (), {"hidden_size": 8})()

            def forward(self, input_ids, attention_mask=None):
                b, l = input_ids.shape
                h = self.drop(torch.ones(b, l, 8))
                return type("O", (), {"last_hidden_state": h})()

        enc = TinyEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(enc.training)
        model.eval()
        self.assertFalse(enc.training)
        model.train()
        self.assertFalse(enc.training)


class TestFrozenTextCache(unittest.TestCase):
    """commit: cache frozen [CLS] keyed on real tokens, not padded rows"""

    def test_cache_keys_ignore_padding_tokens(self):
        from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor()
        proc.fit([{"labs": ([0.0], [[1.0] * 10])}], "labs")
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": proc},
            embedding_dim=8,
            freeze_text_encoder=True,
            cache_frozen_text=True,
        )

        class CountingEnc(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.lin = nn.Linear(1, 8)
                self.config = type("C", (), {"hidden_size": 8})()

            def forward(self, input_ids, attention_mask=None):
                self.calls += 1
                b, l = input_ids.shape
                scale = input_ids[:, :1].float()
                h = torch.ones(b, l, 8) * scale
                return type("O", (), {"last_hidden_state": h})()

        enc = CountingEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")

        ids_a = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 9, 9]])
        mask_a = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        h1 = model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        h2 = model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        ids_b = torch.tensor([[1, 2, 3, 7, 7, 7]])
        mask_b = torch.tensor([[1, 1, 1, 0, 0, 0]])
        h3 = model._encode_text_cls("notes", enc, ids_b, mask_b)
        self.assertEqual(enc.calls, 1)
        self.assertEqual(h1.shape[0], 2)
        self.assertEqual(h3.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
