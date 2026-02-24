# =============================================================================
# Tests for MultimodalJambaEHR and UnifiedMultimodalEmbedding
# Run: python -m unittest tests/core/test_multimodal_jamba.py -v
# =============================================================================

import unittest

import torch

from pyhealth.models.multimodal_jamba import (
    ModalityType,
    MultimodalJambaEHR,
    SinusoidalTimeEmbedding,
    UnifiedMultimodalEmbedding,
)
from pyhealth.models.jamba_ehr import JambaLayer, build_layer_schedule


class TestBuildLayerSchedule(unittest.TestCase):
    """Tests for the interleaved layer schedule builder."""

    def test_pure_mamba(self):
        schedule = build_layer_schedule(0, 4)
        self.assertEqual(schedule, ["mamba"] * 4)

    def test_pure_transformer(self):
        schedule = build_layer_schedule(4, 0)
        self.assertEqual(schedule, ["transformer"] * 4)

    def test_jamba_default(self):
        schedule = build_layer_schedule(2, 6)
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule.count("transformer"), 2)
        self.assertEqual(schedule.count("mamba"), 6)

    def test_even_split(self):
        schedule = build_layer_schedule(4, 4)
        self.assertEqual(schedule.count("transformer"), 4)
        self.assertEqual(schedule.count("mamba"), 4)

    def test_zero_total(self):
        schedule = build_layer_schedule(0, 0)
        self.assertEqual(schedule, [])


class TestJambaLayer(unittest.TestCase):
    """Tests for the JambaLayer backbone."""

    def test_output_shape(self):
        layer = JambaLayer(
            feature_size=64,
            num_transformer_layers=1,
            num_mamba_layers=3,
            heads=2,
        )
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10)
        emb, cls_emb = layer(x, mask=mask)
        self.assertEqual(emb.shape, (2, 10, 64))
        self.assertEqual(cls_emb.shape, (2, 64))

    def test_pure_transformer(self):
        layer = JambaLayer(
            feature_size=64,
            num_transformer_layers=4,
            num_mamba_layers=0,
            heads=2,
        )
        emb, cls_emb = layer(torch.randn(2, 10, 64))
        self.assertEqual(emb.shape, (2, 10, 64))

    def test_pure_mamba(self):
        layer = JambaLayer(
            feature_size=64,
            num_transformer_layers=0,
            num_mamba_layers=4,
        )
        emb, cls_emb = layer(torch.randn(2, 10, 64))
        self.assertEqual(emb.shape, (2, 10, 64))

    def test_gradient_flow(self):
        layer = JambaLayer(
            feature_size=64,
            num_transformer_layers=1,
            num_mamba_layers=2,
            heads=2,
        )
        x = torch.randn(2, 10, 64, requires_grad=True)
        emb, cls_emb = layer(x)
        cls_emb.sum().backward()
        self.assertIsNotNone(x.grad)


class TestSinusoidalTimeEmbedding(unittest.TestCase):
    """Tests for the sinusoidal time embedding module."""

    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(64)
        t = torch.rand(2, 10)
        out = emb(t)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_odd_embed_dim(self):
        emb = SinusoidalTimeEmbedding(63)
        t = torch.rand(2, 10)
        out = emb(t)
        self.assertEqual(out.shape, (2, 10, 63))

    def test_deterministic(self):
        emb = SinusoidalTimeEmbedding(32)
        t = torch.tensor([[0.0, 1.0, 2.0]])
        out1 = emb(t)
        out2 = emb(t)
        self.assertTrue(torch.allclose(out1, out2))

    def test_different_times_different_embeddings(self):
        emb = SinusoidalTimeEmbedding(32)
        t = torch.tensor([[0.0, 100.0]])
        out = emb(t)
        self.assertFalse(torch.allclose(out[:, 0, :], out[:, 1, :]))


class TestUnifiedMultimodalEmbedding(unittest.TestCase):
    """Tests for the unified multimodal embedding module."""

    def setUp(self):
        self.B = 2
        self.E = 64
        self.ume = UnifiedMultimodalEmbedding(
            embed_dim=self.E, use_cls_token=True,
        )

    def test_all_modalities(self):
        inputs = {
            ModalityType.IMAGE: (
                torch.randn(self.B, 5, self.E),
                torch.rand(self.B, 5),
            ),
            ModalityType.TEXT: (
                torch.randn(self.B, 10, self.E),
                torch.rand(self.B, 10),
            ),
            ModalityType.TIMESERIES: (
                torch.randn(self.B, 8, self.E),
                torch.rand(self.B, 8),
            ),
            ModalityType.SEQUENCE: (
                torch.randn(self.B, 3, self.E),
                torch.rand(self.B, 3),
            ),
        }
        emb, mask = self.ume(inputs)
        # CLS + 5 + 10 + 8 + 3 = 27
        self.assertEqual(emb.shape, (self.B, 27, self.E))
        self.assertEqual(mask.shape, (self.B, 27))

    def test_missing_modalities(self):
        inputs = {
            ModalityType.TEXT: (
                torch.randn(self.B, 10, self.E),
                torch.rand(self.B, 10),
            ),
        }
        emb, mask = self.ume(inputs)
        # CLS + 1(missing) + 10(text) + 1(missing) + 1(missing)
        self.assertEqual(emb.shape, (self.B, 14, self.E))

    def test_no_modalities_raises(self):
        with self.assertRaises(ValueError):
            self.ume({})

    def test_no_cls_token(self):
        ume = UnifiedMultimodalEmbedding(
            embed_dim=self.E, use_cls_token=False,
        )
        inputs = {
            ModalityType.IMAGE: (
                torch.randn(self.B, 5, self.E),
                torch.rand(self.B, 5),
            ),
        }
        emb, mask = ume(inputs)
        # 5(img) + 1(miss text) + 1(miss ts) + 1(miss seq) = 8
        self.assertEqual(emb.shape, (self.B, 8, self.E))

    def test_mask_all_valid(self):
        inputs = {
            ModalityType.TEXT: (
                torch.randn(self.B, 5, self.E),
                torch.rand(self.B, 5),
            ),
        }
        _, mask = self.ume(inputs)
        self.assertTrue((mask == 1.0).all())

    def test_mask_is_float(self):
        """JambaLayer expects float mask, not bool."""
        inputs = {
            ModalityType.TEXT: (
                torch.randn(self.B, 5, self.E),
                torch.rand(self.B, 5),
            ),
        }
        _, mask = self.ume(inputs)
        self.assertEqual(mask.dtype, torch.float)

    def test_gradients_through_missing_tokens(self):
        inputs = {
            ModalityType.TEXT: (
                torch.randn(self.B, 5, self.E),
                torch.rand(self.B, 5),
            ),
        }
        emb, _ = self.ume(inputs)
        emb.sum().backward()
        self.assertIsNotNone(self.ume.missing_tokens.grad)


class TestMultimodalJambaEHR(unittest.TestCase):
    """Integration tests for the full model."""

    def setUp(self):
        self.B = 4
        self.E = 64
        self.model = MultimodalJambaEHR(
            embedding_dim=self.E,
            num_transformer_layers=1,
            num_mamba_layers=3,
            heads=2,
            num_classes=2,
        )

    def _make_inputs(self, modalities: dict):
        return {
            mod: (
                torch.randn(self.B, s, self.E),
                torch.rand(self.B, s),
            )
            for mod, s in modalities.items()
        }

    def test_forward_all_modalities(self):
        inputs = self._make_inputs({
            ModalityType.IMAGE: 5,
            ModalityType.TEXT: 10,
            ModalityType.TIMESERIES: 20,
            ModalityType.SEQUENCE: 8,
        })
        out = self.model(
            inputs, labels=torch.randint(0, 2, (self.B,)),
        )
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        self.assertEqual(out["logit"].shape, (self.B, 2))

    def test_forward_missing_modalities(self):
        inputs = self._make_inputs({
            ModalityType.TIMESERIES: 20,
        })
        out = self.model(
            inputs, labels=torch.randint(0, 2, (self.B,)),
        )
        self.assertEqual(out["logit"].shape, (self.B, 2))

    def test_backward_all_params_have_grad(self):
        inputs = self._make_inputs({
            ModalityType.IMAGE: 5,
            ModalityType.TEXT: 10,
        })
        out = self.model(
            inputs, labels=torch.randint(0, 2, (self.B,)),
        )
        out["loss"].backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad, f"No gradient for {name}"
                )

    def test_no_labels_no_loss(self):
        inputs = self._make_inputs({ModalityType.TEXT: 10})
        out = self.model(inputs)
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)

    def test_cls_pooling(self):
        model = MultimodalJambaEHR(
            embedding_dim=self.E,
            num_transformer_layers=1,
            num_mamba_layers=2,
            heads=2,
            num_classes=2,
            pool="cls",
        )
        inputs = self._make_inputs({ModalityType.TEXT: 10})
        out = model(inputs)
        self.assertEqual(out["logit"].shape, (self.B, 2))

    def test_mean_pooling(self):
        model = MultimodalJambaEHR(
            embedding_dim=self.E,
            num_transformer_layers=1,
            num_mamba_layers=2,
            heads=2,
            num_classes=3,
            pool="mean",
        )
        inputs = self._make_inputs({
            ModalityType.TEXT: 10,
            ModalityType.SEQUENCE: 5,
        })
        out = model(inputs)
        self.assertEqual(out["logit"].shape, (self.B, 3))

    def test_last_pooling_default(self):
        """Default pool='last' uses JambaLayer's get_last_visit."""
        inputs = self._make_inputs({ModalityType.IMAGE: 5})
        out = self.model(inputs)
        self.assertEqual(out["logit"].shape, (self.B, 2))

    def test_single_sample_batch(self):
        model = MultimodalJambaEHR(
            embedding_dim=self.E,
            num_transformer_layers=1,
            num_mamba_layers=1,
            heads=2,
            num_classes=2,
        )
        inputs = {
            ModalityType.TEXT: (
                torch.randn(1, 5, self.E),
                torch.rand(1, 5),
            ),
        }
        out = model(inputs, labels=torch.tensor([0]))
        self.assertEqual(out["logit"].shape, (1, 2))
        out["loss"].backward()

    def test_probabilities_sum_to_one(self):
        inputs = self._make_inputs({ModalityType.TEXT: 10})
        out = self.model(inputs)
        sums = out["y_prob"].sum(dim=-1)
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        )

    def test_long_sequence(self):
        model = MultimodalJambaEHR(
            embedding_dim=self.E,
            num_transformer_layers=1,
            num_mamba_layers=3,
            heads=2,
            num_classes=2,
        )
        inputs = self._make_inputs({
            ModalityType.TIMESERIES: 200,
            ModalityType.SEQUENCE: 200,
        })
        out = model(
            inputs, labels=torch.randint(0, 2, (self.B,)),
        )
        self.assertEqual(out["logit"].shape, (self.B, 2))
        out["loss"].backward()


if __name__ == "__main__":
    unittest.main()