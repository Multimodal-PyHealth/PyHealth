"""Chunked, checkpointed text encoding for a trainable encoder is exact."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
from pyhealth.processors.stagenet_processor import StageNetTensorProcessor


def _model(**kwargs) -> UnifiedMultimodalEmbeddingModel:
    proc = StageNetTensorProcessor()
    proc.fit([{"labs": ([0.0], [[1.0] * 10])}], "labs")
    return UnifiedMultimodalEmbeddingModel(
        {"labs": proc}, embedding_dim=8, freeze_text_encoder=False, **kwargs
    )


class TrainableEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.lin = nn.Linear(8, 8)
        self.calls = 0
        self.config = type("C", (), {"hidden_size": 8})()

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        h = self.lin(self.emb(input_ids))
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        return type("O", (), {"last_hidden_state": h})()


class TestTextGradCheckpoint(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(0)
        ids = torch.randint(1, 16, (7, 5))
        mask = torch.ones_like(ids)
        mask[:, -1] = 0
        return ids, mask

    def test_chunked_matches_unchunked_and_trains(self):
        ids, mask = self._inputs()
        plain = _model(text_grad_checkpoint_rows=0)
        chunked = _model(text_grad_checkpoint_rows=3)
        enc = TrainableEnc()
        plain.encoders["notes"] = enc
        chunked.encoders["notes"] = enc
        plain.train()
        chunked.train()

        enc.calls = 0
        h_plain = plain._encode_text_cls("notes", enc, ids, mask)
        self.assertEqual(enc.calls, 1)

        enc.calls = 0
        h_chunk = chunked._encode_text_cls("notes", enc, ids, mask)
        self.assertEqual(enc.calls, 3)  # ceil(7 / 3) forward chunks
        self.assertEqual(h_chunk.shape, (7, 8))
        torch.testing.assert_close(h_chunk, h_plain)

        enc.zero_grad()
        h_chunk.sum().backward()
        self.assertIsNotNone(enc.lin.weight.grad)
        self.assertGreater(enc.lin.weight.grad.abs().sum().item(), 0.0)

    def test_no_grad_path_does_not_chunk(self):
        ids, mask = self._inputs()
        chunked = _model(text_grad_checkpoint_rows=3)
        enc = TrainableEnc()
        chunked.encoders["notes"] = enc
        chunked.eval()
        with torch.no_grad():
            out = chunked._encode_text_cls("notes", enc, ids, mask)
        self.assertEqual(enc.calls, 1)
        self.assertEqual(out.shape, (7, 8))

    def test_frozen_field_ignores_chunking(self):
        ids, mask = self._inputs()
        model = _model(text_grad_checkpoint_rows=3, cache_frozen_text=False)
        enc = TrainableEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")
        model.train()
        out = model._encode_text_cls("notes", enc, ids, mask)
        self.assertEqual(enc.calls, 1)
        self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    unittest.main()
