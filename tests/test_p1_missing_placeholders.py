"""Proof that empty modalities are zero events, with a Will-on placeholder flag.

Default: no notes/labs/CXR → shape[0] == 0.
--emit-missing-placeholders: one fake event ([MISSING_TEXT], zero lab row,
empty-path CXR → black RGB frame).

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_missing_placeholders.py -q
"""

from __future__ import annotations

import unittest

import torch


class TestP1MissingPlaceholders(unittest.TestCase):
    def test_default_leaves_empty_lists_empty(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4()
        self.assertFalse(task.emit_missing_placeholders)
        notes, times = [], []
        labs, masks, lab_times = [], [], []
        task._apply_missing_placeholders(
            note_texts=notes,
            note_times=times,
            lab_values=labs,
            lab_masks=masks,
            lab_times=lab_times,
        )
        self.assertEqual(notes, [])
        self.assertEqual(labs, [])

    def test_will_on_invents_one_fake_event_per_empty_modality(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsCXRMIMIC4

        task = NotesLabsCXRMIMIC4(emit_missing_placeholders=True)
        notes, times = [], []
        labs, masks, lab_times = [], [], []
        cxr, cxr_times = [], []
        task._apply_missing_placeholders(
            note_texts=notes,
            note_times=times,
            lab_values=labs,
            lab_masks=masks,
            lab_times=lab_times,
            cxr_paths=cxr,
            cxr_times=cxr_times,
        )
        self.assertEqual(notes, ["[MISSING_TEXT]"])
        self.assertEqual(times, [0.0])
        self.assertEqual(len(labs[0]), 10)
        self.assertEqual(labs[0], [0.0] * 10)
        self.assertEqual(masks[0], [False] * 10)
        self.assertEqual(cxr, [""])
        self.assertEqual(cxr_times, [0.0])

    def test_processor_empty_notes_are_zero_events(self):
        from pyhealth.processors import TupleTimeTextProcessor

        texts, time, _ = TupleTimeTextProcessor().process(([], []))
        self.assertEqual(texts, [])
        self.assertEqual(tuple(time.shape), (0,))
        self.assertNotIn("[MISSING_TEXT]", texts)

    def test_processor_missing_text_token_is_one_event(self):
        from pyhealth.processors import TupleTimeTextProcessor

        texts, time, _ = TupleTimeTextProcessor().process(
            (["[MISSING_TEXT]"], [0.0])
        )
        self.assertEqual(texts, ["[MISSING_TEXT]"])
        self.assertEqual(tuple(time.shape), (1,))

    def test_empty_image_list_is_zero_events(self):
        from pyhealth.processors.time_image_processor import TimeImageProcessor

        images, timestamps, tag = TimeImageProcessor(image_size=8, mode="RGB").process(
            ([], [])
        )
        self.assertEqual(images.shape[0], 0)
        self.assertEqual(tuple(timestamps.shape), (0,))
        self.assertEqual(tag, "image")

    def test_empty_path_is_a_black_frame(self):
        from pyhealth.processors.time_image_processor import TimeImageProcessor

        proc = TimeImageProcessor(image_size=8, mode="RGB", padding="")
        images, timestamps, _ = proc.process(([""], [0.0]))
        self.assertEqual(tuple(images.shape), (1, 3, 8, 8))
        self.assertTrue(torch.equal(images[0], torch.zeros(3, 8, 8)))
        self.assertEqual(float(timestamps[0]), 0.0)

    def test_unified_forwards_an_empty_image_batch(self):
        from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.time_image_processor import TimeImageProcessor

        proc = TimeImageProcessor(image_size=8, mode="RGB")
        proc.fit([{"cxr": ([], [])}], "cxr")
        model = UnifiedMultimodalEmbeddingModel(
            {"cxr": proc},
            embedding_dim=8,
            image_size=8,
            patch_size=8,
        ).eval()
        images = torch.zeros(2, 0, 3, 8, 8)
        times = torch.zeros(2, 0)
        out = model(
            {"cxr": {"value": images, "time": times, "mask": None}}
        )
        self.assertEqual(tuple(out["sequence"].shape), (2, 0, 8))


if __name__ == "__main__":
    unittest.main()
