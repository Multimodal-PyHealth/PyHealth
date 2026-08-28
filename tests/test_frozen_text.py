"""Proofs for frozen-text encoder behaviour in UnifiedMultimodalEmbeddingModel."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
from pyhealth.processors.stagenet_processor import StageNetTensorProcessor


def _numeric_model(**kwargs) -> UnifiedMultimodalEmbeddingModel:
    proc = StageNetTensorProcessor()
    proc.fit([{"labs": ([0.0], [[1.0] * 10])}], "labs")
    return UnifiedMultimodalEmbeddingModel(
        {"labs": proc}, embedding_dim=8, freeze_text_encoder=True, **kwargs
    )


class TinyEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.9)
        self.lin = nn.Linear(1, 8)
        self.config = type("C", (), {"hidden_size": 8})()

    def forward(self, input_ids, attention_mask=None):
        b, l = input_ids.shape
        h = self.drop(torch.ones(b, l, 8))
        return type("O", (), {"last_hidden_state": h})()


class TestFrozenEncoderEval(unittest.TestCase):
    def test_train_keeps_frozen_text_encoder_in_eval(self):
        model = _numeric_model()
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


class TestFrozenTextCache(unittest.TestCase):
    def test_cache_keys_ignore_padding_tokens(self):
        model = _numeric_model(cache_frozen_text=True)
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

    def test_full_cache_recomputes_uncached_notes(self):
        model = _numeric_model(cache_frozen_text=True, max_frozen_text_cache=1)
        enc = CountingEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")

        ids_a = torch.tensor([[1, 2, 3]])
        mask_a = torch.tensor([[1, 1, 1]])
        model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        ids_b = torch.tensor([[4, 5, 6]])
        mask_b = torch.tensor([[1, 1, 1]])
        model._encode_text_cls("notes", enc, ids_b, mask_b)
        after_b = enc.calls
        self.assertGreater(after_b, 1)
        model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, after_b)

    def test_full_cache_encodes_each_miss_once_per_forward(self):
        model = _numeric_model(cache_frozen_text=True, max_frozen_text_cache=1)
        enc = CountingEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")

        model._encode_text_cls(
            "notes", enc, torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]])
        )
        self.assertEqual(enc.calls, 1)
        # CountingEnc broadcasts (b, l, 8) * (b, 1), so keep b == l for the
        # two unique misses it will see.
        ids = torch.tensor([[4, 5], [7, 8], [4, 5]])
        mask = torch.ones_like(ids)
        out = model._encode_text_cls("notes", enc, ids, mask)
        # One batched call for the two unique misses; no per-row re-encode.
        self.assertEqual(enc.calls, 2)
        self.assertEqual(out.shape[0], 3)
        torch.testing.assert_close(out[0], out[2])

    def test_uncapped_cache_keeps_every_unique_note(self):
        model = _numeric_model(cache_frozen_text=True, max_frozen_text_cache=None)
        enc = CountingEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")

        ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mask = torch.ones_like(ids)
        model._encode_text_cls("notes", enc, ids, mask)
        first = enc.calls
        self.assertEqual(first, 1)
        model._encode_text_cls("notes", enc, ids, mask)
        self.assertEqual(enc.calls, first)


if __name__ == "__main__":
    unittest.main()

