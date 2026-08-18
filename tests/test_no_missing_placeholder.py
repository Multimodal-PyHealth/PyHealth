"""Proof that empty notes/labs/CXR/ICD are zero events, not fake rows.

The previous fallback stuffed ``[MISSING_TEXT]`` / empty-string notes / a pad
ICD visit / a black image / a zero lab row so the fast tokenizer would not
crash. BERT then embedded a constant, and note presence tracked mortality.
"""

from __future__ import annotations

import inspect
import unittest
from datetime import datetime


class TestProcessorsEmitZeroEvents(unittest.TestCase):
    def test_empty_note_list_is_not_missing_text(self):
        from pyhealth.processors.tuple_time_text_processor import TupleTimeTextProcessor

        texts, times, tag = TupleTimeTextProcessor().process(([], []))
        self.assertEqual(texts, [])
        self.assertEqual(tuple(times.shape), (0,))
        self.assertEqual(tag, "note")

    def test_empty_codes_are_not_a_pad_visit(self):
        from pyhealth.processors.stagenet_processor import StageNetProcessor

        proc = StageNetProcessor()
        proc.fit([{"data": ([0.0], ["A"])}], "data")
        _, values = proc.process((None, []))
        self.assertEqual(tuple(values.shape), (0,))

    def test_empty_images_are_not_a_black_frame(self):
        from pyhealth.processors.time_image_processor import TimeImageProcessor

        images, times, tag = TimeImageProcessor(image_size=16, mode="L").process(
            ([], [])
        )
        self.assertEqual(tuple(images.shape), (0, 1, 16, 16))
        self.assertEqual(tuple(times.shape), (0,))
        self.assertEqual(tag, "image")


class TestTasksDoNotInjectPlaceholders(unittest.TestCase):
    def test_task_bodies_do_not_stuff_missing_text(self):
        from pyhealth.tasks import multimodal_mimic4 as m

        for name in (
            "ClinicalNotesMIMIC4",
            "ClinicalNotesICDLabsMIMIC4",
            "ClinicalNotesICDLabsCXRMIMIC4",
            "ICDLabsMIMIC4",
            "NotesLabsMIMIC4",
            "LabsOnlyMIMIC4",
        ):
            src = inspect.getsource(getattr(m, name).__call__)
            self.assertNotIn(
                "MISSING_TEXT_TOKEN",
                src,
                msg=f"{name} still injects a fake missing-text event",
            )
            self.assertNotIn(
                "MISSING_CODE_TOKEN",
                src,
                msg=f"{name} still injects a fake missing-code visit",
            )

    def test_notes_labs_empty_patient_emits_empty_lists(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        class _Event:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _Patient:
            patient_id = "p-1"

            def get_events(
                self, event_type, start=None, end=None, filters=None, return_df=False
            ):
                if event_type == "patients":
                    return [_Event(anchor_age=55)]
                if event_type == "admissions":
                    return [
                        _Event(
                            timestamp=datetime(2020, 1, 1, 0, 0, 0),
                            dischtime="2020-01-03 12:00:00",
                            hadm_id=101,
                            hospital_expire_flag=0,
                        )
                    ]
                if return_df:
                    return pl.DataFrame(
                        {
                            "timestamp": [],
                            "labevents/itemid": [],
                            "labevents/storetime": [],
                            "labevents/valuenum": [],
                        }
                    )
                return []

        samples = NotesLabsMIMIC4(window_hours=24)(_Patient())
        self.assertEqual(len(samples), 1)
        notes, note_times = samples[0]["admission_note_times"]
        lab_times, lab_values = samples[0]["labs"]
        self.assertEqual(notes, [])
        self.assertEqual(note_times, [])
        self.assertEqual(lab_times, [])
        self.assertEqual(lab_values, [])

    def test_notes_labs_call_uses_per_admission_window(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        src = inspect.getsource(NotesLabsMIMIC4.__call__)
        self.assertIn("_admission_window_end", src)
