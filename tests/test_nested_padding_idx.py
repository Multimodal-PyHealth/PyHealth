"""Proof that nested code embeddings freeze the pad row.

padding_idx=None let gradients update index 0, so padded visits moved the
representation of a real code that happened to share that slot.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn


class TestNestedPaddingIdx(unittest.TestCase):
    def test_nested_embedding_pad_row_is_zeros_and_frozen(self):
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
        x = torch.zeros(2, 3, dtype=torch.long)
        y = emb(x).sum()
        y.backward()
        self.assertIsNotNone(emb.weight.grad)
        self.assertTrue(
            torch.equal(emb.weight.grad[0], torch.zeros_like(emb.weight.grad[0]))
        )
